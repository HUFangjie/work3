# defenses/defense_fedgraphguard.py
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch

from defenses.base_defense import BaseDefense


class FedGraphGuardDefense(BaseDefense):
    """FedGraphGuard aggregation for federated distillation logits.

    The original warning reported on Tiny-ImageNet comes from spectral methods
    receiving a disconnected affinity graph.  This implementation keeps the
    graph-based FedGraphGuard design (client-logit graph construction, graph
    connectivity repair, trust scoring, client filtering, weighted aggregation)
    but does **not** call sklearn's spectral embedding on a disconnected graph.

    Key points:
      1) Build one feature vector per client from its public logits.
      2) Compute cosine affinities between clients.
      3) Build either a dense graph or a sparse k-NN graph.
      4) If sparse graph components are disconnected, deterministically connect
         them with the strongest cross-component edges before scoring.
      5) Score clients by graph degree / medoid proximity, keep the most trusted
         clients, and return a trust-weighted logit aggregate.

    This keeps the defense behavior graph-based while removing the repeated
    "Graph is not fully connected" sklearn warning at the source.
    """

    def __init__(
        self,
        device: torch.device,
        keep_ratio: float = 0.7,
        min_clients_kept: int = 2,
        similarity_temperature: float = 0.5,
        affinity_floor: float = 1e-6,
        weight_temperature: float = 0.5,
        normalize_logits: bool = True,
        graph_mode: str = "knn",
        knn_k: int = 0,
        connect_components: bool = True,
        component_floor: float = 1e-3,
        medoid_mix: float = 0.25,
    ) -> None:
        super().__init__(device=device)
        self.keep_ratio = float(keep_ratio)
        self.min_clients_kept = int(min_clients_kept)
        self.similarity_temperature = float(similarity_temperature)
        self.affinity_floor = float(affinity_floor)
        self.weight_temperature = float(weight_temperature)
        self.normalize_logits = bool(normalize_logits)
        self.graph_mode = str(graph_mode).lower()
        self.knn_k = int(knn_k)
        self.connect_components = bool(connect_components)
        self.component_floor = float(component_floor)
        self.medoid_mix = float(medoid_mix)

    # ------------------------------------------------------------------
    # Feature / similarity helpers
    # ------------------------------------------------------------------
    def _stack_client_logits(
        self,
        client_logits: Dict[int, torch.Tensor],
    ) -> Tuple[List[int], torch.Tensor]:
        client_ids = [int(cid) for cid in client_logits.keys()]
        if len(client_ids) == 0:
            raise ValueError("FedGraphGuardDefense received empty client_logits.")
        stacked = torch.stack(
            [client_logits[cid].detach().float().to(self.device) for cid in client_ids],
            dim=0,
        )
        return client_ids, stacked

    def _flatten_features(self, stacked_logits: torch.Tensor) -> torch.Tensor:
        features = stacked_logits.reshape(stacked_logits.shape[0], -1)
        if self.normalize_logits:
            features = features - features.mean(dim=1, keepdim=True)
            features = features / features.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
        return features

    @staticmethod
    def _cosine_similarity(features: torch.Tensor) -> torch.Tensor:
        normalized = features - features.mean(dim=1, keepdim=True)
        normalized = normalized / normalized.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
        return (normalized @ normalized.t()).clamp(min=-1.0, max=1.0)

    def _similarity_to_affinity(self, sim: torch.Tensor) -> torch.Tensor:
        temp = max(self.similarity_temperature, 1e-12)
        affinity = torch.exp((sim - 1.0) / temp)
        floor = max(self.affinity_floor, 0.0)
        if floor > 0.0:
            affinity = affinity.clamp_min(floor)
        affinity.fill_diagonal_(0.0)
        return affinity

    # ------------------------------------------------------------------
    # Graph construction / connectivity repair
    # ------------------------------------------------------------------
    def _default_knn_k(self, num_clients: int) -> int:
        if self.knn_k > 0:
            return min(max(1, self.knn_k), max(1, num_clients - 1))
        # A small but not too sparse k.  For 10 clients this gives k=4, which
        # is robust on Tiny-ImageNet without making the graph fully uniform.
        return min(max(1, int(math.ceil(math.sqrt(num_clients))) + 1), max(1, num_clients - 1))

    def _build_knn_graph(self, affinity: torch.Tensor) -> torch.Tensor:
        num_clients = int(affinity.shape[0])
        if num_clients <= 2:
            return affinity.clone()

        k = self._default_knn_k(num_clients)
        graph = torch.zeros_like(affinity)
        topk_idx = torch.topk(affinity, k=k, dim=1, largest=True).indices
        graph.scatter_(1, topk_idx, affinity.gather(1, topk_idx))
        # Symmetrize because graph defenses usually treat client similarity as
        # undirected; max preserves the stronger directed k-NN edge.
        graph = torch.maximum(graph, graph.t())
        graph.fill_diagonal_(0.0)
        return graph

    def _connected_components(self, graph: torch.Tensor) -> List[List[int]]:
        num_clients = int(graph.shape[0])
        adjacency = graph > 0
        visited = torch.zeros(num_clients, dtype=torch.bool, device=graph.device)
        components: List[List[int]] = []

        for start in range(num_clients):
            if bool(visited[start].item()):
                continue
            stack = [start]
            visited[start] = True
            comp: List[int] = []
            while stack:
                node = stack.pop()
                comp.append(node)
                neighbors = torch.nonzero(adjacency[node], as_tuple=False).view(-1).tolist()
                for nb in neighbors:
                    nb_i = int(nb)
                    if not bool(visited[nb_i].item()):
                        visited[nb_i] = True
                        stack.append(nb_i)
            components.append(comp)
        return components

    def _connect_components(self, graph: torch.Tensor, dense_affinity: torch.Tensor) -> torch.Tensor:
        if not self.connect_components:
            return graph

        connected = graph.clone()
        components = self._connected_components(connected)
        if len(components) <= 1:
            return connected

        # Iteratively attach the closest remaining component to the first
        # component using the strongest cross-component affinity.  This is small
        # (num clients per round), deterministic, and avoids sklearn's warning.
        while len(components) > 1:
            base = components[0]
            best_edge: Optional[Tuple[int, int, float, int]] = None
            base_idx = torch.tensor(base, dtype=torch.long, device=graph.device)

            for comp_pos, comp in enumerate(components[1:], start=1):
                comp_idx = torch.tensor(comp, dtype=torch.long, device=graph.device)
                block = dense_affinity.index_select(0, base_idx).index_select(1, comp_idx)
                flat_idx = int(torch.argmax(block).item())
                row = flat_idx // len(comp)
                col = flat_idx % len(comp)
                weight = float(block[row, col].item())
                if best_edge is None or weight > best_edge[2]:
                    best_edge = (base[row], comp[col], weight, comp_pos)

            if best_edge is None:
                break

            i, j, weight, comp_pos = best_edge
            repaired_weight = max(weight, max(self.component_floor, self.affinity_floor, 0.0))
            connected[i, j] = repaired_weight
            connected[j, i] = repaired_weight
            components[0] = components[0] + components[comp_pos]
            del components[comp_pos]

        return connected

    def _build_connected_graph(self, features: torch.Tensor) -> torch.Tensor:
        sim = self._cosine_similarity(features)
        dense_affinity = self._similarity_to_affinity(sim)

        if self.graph_mode == "dense":
            graph = dense_affinity
        else:
            graph = self._build_knn_graph(dense_affinity)

        return self._connect_components(graph=graph, dense_affinity=dense_affinity)

    # ------------------------------------------------------------------
    # Trust scoring / aggregation
    # ------------------------------------------------------------------
    def _trust_scores(self, graph: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        num_clients = int(graph.shape[0])
        degree = graph.sum(dim=1) / max(num_clients - 1, 1)

        # Medoid proximity stabilizes degree-only scores when k-NN degrees tie.
        pair_dist = torch.cdist(features, features, p=2)
        medoid = int(torch.argmin(pair_dist.mean(dim=1)).item())
        medoid_sim = graph[medoid]
        medoid_sim = medoid_sim / medoid_sim.max().clamp_min(1e-12)

        mix = min(max(self.medoid_mix, 0.0), 1.0)
        return (1.0 - mix) * degree + mix * medoid_sim

    def _num_clients_to_keep(self, num_clients: int) -> int:
        ratio = min(max(self.keep_ratio, 0.0), 1.0)
        keep_n = int(math.ceil(num_clients * ratio))
        keep_n = max(keep_n, min(self.min_clients_kept, num_clients))
        return min(max(1, keep_n), num_clients)

    def _aggregate_kept_logits(
        self,
        stacked_logits: torch.Tensor,
        trust: torch.Tensor,
        keep_idx: torch.Tensor,
    ) -> torch.Tensor:
        kept_trust = trust[keep_idx]
        temp = max(self.weight_temperature, 1e-12)
        weights = torch.softmax((kept_trust - kept_trust.max()) / temp, dim=0)
        return (weights.view(-1, 1, 1) * stacked_logits[keep_idx]).sum(dim=0)

    def aggregate(
        self,
        client_logits: Dict[int, torch.Tensor],
        y_public: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        _, stacked_logits = self._stack_client_logits(client_logits)
        num_clients = int(stacked_logits.shape[0])
        if num_clients == 1:
            return stacked_logits[0]

        features = self._flatten_features(stacked_logits)
        graph = self._build_connected_graph(features)
        trust = self._trust_scores(graph=graph, features=features)

        keep_n = self._num_clients_to_keep(num_clients)
        keep_idx = torch.topk(trust, k=keep_n, largest=True).indices
        return self._aggregate_kept_logits(
            stacked_logits=stacked_logits,
            trust=trust,
            keep_idx=keep_idx,
        )
