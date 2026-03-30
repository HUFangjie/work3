from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from sklearn.cluster import SpectralClustering

from defenses.base_defense import BaseDefense


class FedGraphGuardDefense(BaseDefense):
    """FedGraphGuard: graph-based trust-aware robust aggregation for FD logits."""

    def __init__(
        self,
        device: torch.device,
        temperature: float = 1.0,
        topk: int = 3,
        winsor_q: float = 0.1,
        rn: int = 3,
        gnn_layers: int = 2,
        gnn_gamma: float = 0.5,
        lrr_lambda: float = 0.05,
        lrr_gamma: float = 0.01,
        lrr_iters: int = 40,
        lrr_lr: float = 0.1,
        alpha: float = 0.6,
        n_clusters: int = 2,
        tau: float = 1.0,
        phi_min: float = 0.02,
        ppr_beta: float = 0.85,
        ppr_max_iter: int = 100,
        ppr_tol: float = 1e-6,
        trust_threshold: float = 0.1,
        trim_ratio: float = 0.2,
        eps: float = 1e-12,
    ) -> None:
        super().__init__(device=device)
        self.temperature = float(temperature)
        self.topk = int(topk)
        self.winsor_q = float(winsor_q)
        self.rn = int(rn)
        self.gnn_layers = int(gnn_layers)
        self.gnn_gamma = float(gnn_gamma)
        self.lrr_lambda = float(lrr_lambda)
        self.lrr_gamma = float(lrr_gamma)
        self.lrr_iters = int(lrr_iters)
        self.lrr_lr = float(lrr_lr)
        self.alpha = float(alpha)
        self.n_clusters = int(n_clusters)
        self.tau = float(tau)
        self.phi_min = float(phi_min)
        self.ppr_beta = float(ppr_beta)
        self.ppr_max_iter = int(ppr_max_iter)
        self.ppr_tol = float(ppr_tol)
        self.trust_threshold = float(trust_threshold)
        self.trim_ratio = float(trim_ratio)
        self.eps = float(eps)

        self.last_debug: Dict[str, Any] = {}

    def build_reference_free_graph(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n, _, c = logits.shape
        probs = torch.softmax(logits / max(self.temperature, self.eps), dim=-1)

        k = max(1, min(self.topk, c))
        top_idx = torch.topk(probs, k=k, dim=-1).indices
        top_mask = torch.zeros_like(probs, dtype=torch.bool)
        top_mask.scatter_(-1, top_idx, True)

        WC = torch.eye(n, dtype=torch.float32, device=logits.device)
        low_q = max(0.0, min(0.5, self.winsor_q))
        high_q = 1.0 - low_q

        for i in range(n):
            for j in range(i + 1, n):
                inter = (top_mask[i] & top_mask[j]).sum(dim=-1).float()
                union = (top_mask[i] | top_mask[j]).sum(dim=-1).float().clamp_min(1.0)
                jac = inter / union
                lo = torch.quantile(jac, low_q)
                hi = torch.quantile(jac, high_q)
                sim = torch.clamp(jac, lo, hi).mean()
                WC[i, j] = sim
                WC[j, i] = sim

        Wc = torch.zeros_like(WC)
        k_nn = max(1, min(self.rn, max(1, n - 1)))
        for i in range(n):
            scores = WC[i].clone()
            scores[i] = -1.0
            nn_idx = torch.topk(scores, k=k_nn, largest=True).indices
            Wc[i, nn_idx] = WC[i, nn_idx]

        Wc = torch.maximum(Wc, Wc.t())
        row_sum = Wc.sum(dim=1, keepdim=True).clamp_min(self.eps)
        P = Wc / row_sum
        return probs, WC, Wc, P

    def local_purification_gnn(self, probs: torch.Tensor, Wc: torch.Tensor) -> torch.Tensor:
        h = probs.mean(dim=1)  # [n, c]
        n = h.shape[0]

        for _ in range(max(1, self.gnn_layers)):
            h_new = torch.empty_like(h)
            for i in range(n):
                nei = torch.where(Wc[i] > 0)[0]
                if nei.numel() == 0:
                    med = h[i]
                else:
                    neigh_feat = torch.cat([h[i : i + 1], h[nei]], dim=0)
                    med = torch.median(neigh_feat, dim=0).values
                h_new[i] = (1.0 - self.gnn_gamma) * h[i] + self.gnn_gamma * med
            h = h_new
        return h

    def global_subspace_purification(self, X: torch.Tensor, Wc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = X.shape[0]
        Xt = X.t()  # [d, n]
        G = X @ X.t()  # [n, n]
        Z = torch.relu(G.clone())
        Z.fill_diagonal_(0.0)

        I = torch.eye(n, dtype=X.dtype, device=X.device)
        for _ in range(max(1, self.lrr_iters)):
            grad_rec = 2.0 * (G @ (Z - I))
            grad_l1 = self.lrr_gamma * torch.sign(Z)
            Z = Z - self.lrr_lr * (grad_rec + grad_l1)

            try:
                u, s, vh = torch.linalg.svd(Z, full_matrices=False)
                s = torch.relu(s - self.lrr_lr * self.lrr_lambda)
                Z = (u * s.unsqueeze(0)) @ vh
            except Exception:
                pass

            Z = torch.clamp(Z, min=0.0)
            Z.fill_diagonal_(0.0)

        W_tilde = 0.5 * (torch.abs(Z) + torch.abs(Z.t()))
        W_final = self.alpha * W_tilde + (1.0 - self.alpha) * Wc
        P_final = W_final / W_final.sum(dim=1, keepdim=True).clamp_min(self.eps)

        _ = Xt  # keep explicit use consistent with formulation
        return Z, W_tilde, P_final

    def evaluate_cluster_trust(self, X: torch.Tensor, Z: torch.Tensor, W_tilde: torch.Tensor, Wc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        n = X.shape[0]
        if n <= 2:
            clusters = torch.zeros(n, dtype=torch.long, device=X.device)
        else:
            k = max(2, min(self.n_clusters, n - 1))
            A = W_tilde.detach().cpu().numpy()
            try:
                sc = SpectralClustering(n_clusters=k, affinity="precomputed", random_state=0, assign_labels="kmeans")
                labels_np = sc.fit_predict(A)
            except Exception:
                labels_np = np.zeros(n, dtype=np.int64)
            clusters = torch.tensor(labels_np, dtype=torch.long, device=X.device)

        Xt = X.t()
        X_hat = (Xt @ Z).t()
        unique = torch.unique(clusters)

        t0 = torch.zeros(n, dtype=torch.float32, device=X.device)
        for cid in unique.tolist():
            mask = clusters == int(cid)
            idx = torch.where(mask)[0]
            if idx.numel() == 0:
                continue

            err = torch.norm(X[idx] - X_hat[idx], p="fro").pow(2)

            in_mask = mask.float().view(-1, 1)
            out_mask = (1.0 - mask.float()).view(1, -1)
            cut = (Wc * (in_mask @ out_mask)).sum()
            vol = Wc[idx].sum().clamp_min(self.eps)
            phi = cut / vol

            conf = torch.exp(-err / max(self.tau, self.eps)) * (phi > self.phi_min).float()
            t0[idx] = conf.float()

        if float(t0.sum().item()) <= self.eps:
            t0 = torch.ones_like(t0) / max(1, n)
        else:
            t0 = t0 / t0.sum().clamp_min(self.eps)
        return clusters, t0

    def propagate_trust(self, P_final: torch.Tensor, t0: torch.Tensor) -> torch.Tensor:
        t = t0.clone()
        for _ in range(max(1, self.ppr_max_iter)):
            t_next = self.ppr_beta * (P_final.t() @ t) + (1.0 - self.ppr_beta) * t0
            if torch.norm(t_next - t, p=1) < self.ppr_tol:
                t = t_next
                break
            t = t_next

        t = torch.clamp(t, min=0.0)
        if float(t.sum().item()) <= self.eps:
            t = torch.ones_like(t) / max(1, t.numel())
        else:
            t = t / t.sum().clamp_min(self.eps)
        return t

    def aggregate_logits(self, logits: torch.Tensor, trust: torch.Tensor) -> torch.Tensor:
        n, b, c = logits.shape
        keep = torch.where(trust >= self.trust_threshold)[0]
        if keep.numel() == 0:
            keep = torch.topk(trust, k=1).indices

        x = logits[keep]
        w = trust[keep]
        m = x.shape[0]

        trim_each = int(max(0, min(m // 2, int(np.floor(m * self.trim_ratio / 2.0)))))
        agg = torch.empty((b, c), dtype=logits.dtype, device=logits.device)

        for j in range(b):
            for cls in range(c):
                vals = x[:, j, cls]
                ord_idx = torch.argsort(vals)
                if m - 2 * trim_each <= 0:
                    keep_idx = ord_idx
                else:
                    keep_idx = ord_idx[trim_each : m - trim_each]

                v = vals[keep_idx]
                ww = w[keep_idx]
                agg[j, cls] = (v * ww).sum() / ww.sum().clamp_min(self.eps)

        return agg

    def aggregate(
        self,
        client_logits: Dict[int, torch.Tensor],
        y_public: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if not client_logits:
            raise ValueError("FedGraphGuardDefense received empty client_logits.")

        cids = list(client_logits.keys())
        logits = torch.stack([client_logits[cid].detach().float() for cid in cids], dim=0)

        probs, WC, Wc, _ = self.build_reference_free_graph(logits)
        X = self.local_purification_gnn(probs, Wc)
        Z, W_tilde, P_final = self.global_subspace_purification(X, Wc)
        clusters, t0 = self.evaluate_cluster_trust(X, Z, W_tilde, Wc)
        trust = self.propagate_trust(P_final, t0)
        agg = self.aggregate_logits(logits, trust)

        self.last_debug = {
            "client_ids": cids,
            "trust_scores": {cid: float(trust[i].item()) for i, cid in enumerate(cids)},
            "cluster_labels": {cid: int(clusters[i].item()) for i, cid in enumerate(cids)},
            "Wc": Wc.detach().cpu(),
            "W_tilde": W_tilde.detach().cpu(),
            "Z": Z.detach().cpu(),
        }
        return agg
