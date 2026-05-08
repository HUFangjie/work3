from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))
    from obs_utils import resolve_out_dir, save_json
else:
    from .obs_utils import resolve_out_dir, save_json


def load_logits(path: str) -> np.ndarray:
    logits = np.load(path)
    if logits.ndim != 3:
        raise ValueError(f"Expected logits shape (K,N_pub,C), got {logits.shape}")
    return np.asarray(logits, dtype=np.float64)


def validate_inputs(logits: np.ndarray, byz_ids: Sequence[int]) -> Tuple[List[int], List[int]]:
    K = logits.shape[0]
    byz = sorted(set(byz_ids))
    if len(byz) == 0:
        raise ValueError("byzantine_ids cannot be empty")
    if min(byz) < 0 or max(byz) >= K:
        raise ValueError("byzantine_ids out of range")
    if (len(byz) / K) >= 0.5:
        warnings.warn("Byzantine ratio >= 0.5 may weaken separation in Observation-3", RuntimeWarning)
    return byz, [i for i in range(K) if i not in byz]


def jaccard_similarity_matrix(logits: np.ndarray, kappa: int = 5) -> np.ndarray:
    K, _, C = logits.shape
    kk = min(kappa, C)
    topk = np.argpartition(logits, kth=C - kk, axis=2)[:, :, -kk:]
    S = np.eye(K, dtype=np.float64)
    for i in range(K):
        for j in range(i + 1, K):
            vals = []
            for a, b in zip(topk[i], topk[j]):
                sa, sb = set(a.tolist()), set(b.tolist())
                vals.append(len(sa & sb) / len(sa | sb))
            S[i, j] = S[j, i] = float(np.mean(vals))
    return S


def apply_gaussian_attack(logits: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return logits + rng.normal(0.0, sigma, size=logits.shape)


def apply_label_flip_swap_attack(logits: np.ndarray) -> np.ndarray:
    """Top-bottom logit swap per sample (label_flip_swap), not retraining a flipped-label teacher."""
    adv = logits.copy()
    top = np.argmax(logits, axis=1)
    bot = np.argmin(logits, axis=1)
    rows = np.arange(logits.shape[0])
    adv[rows, top], adv[rows, bot] = logits[rows, bot], logits[rows, top]
    return adv


def apply_targeted_attack(logits: np.ndarray, target_class: int) -> np.ndarray:
    adv = np.zeros_like(logits)
    adv[:, target_class] = 10.0
    return adv


def apply_alie_attack(benign_logits_all: np.ndarray, benign_ids: Sequence[int], z_scale: float) -> np.ndarray:
    mu = np.mean(benign_logits_all[benign_ids], axis=0)
    std = np.std(benign_logits_all[benign_ids], axis=0)
    return mu + z_scale * std


def build_observed_logits(benign_logits: np.ndarray, byz_ids: Sequence[int], attack: str, args: argparse.Namespace) -> np.ndarray:
    obs = benign_logits.copy()
    K = benign_logits.shape[0]
    benign_ids = [i for i in range(K) if i not in byz_ids]
    rng = np.random.default_rng(args.seed)
    for b in byz_ids:
        if attack == "gaussian":
            obs[b] = apply_gaussian_attack(obs[b], args.gaussian_sigma, rng)
        elif attack == "label_flip":
            obs[b] = apply_label_flip_swap_attack(obs[b])
        elif attack == "targeted":
            obs[b] = apply_targeted_attack(obs[b], args.target_class)
        elif attack == "alie":
            obs[b] = apply_alie_attack(benign_logits, benign_ids, args.alie_z_scale)
        else:
            raise ValueError(f"Unsupported attack: {attack}")
    return obs


def _soft_threshold(X: np.ndarray, tau: float) -> np.ndarray:
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0.0)


def solve_rpca_admm(S: np.ndarray, lam: Optional[float], rho: float, max_iter: int, tol: float):
    K = S.shape[0]
    lam_eff = (1.0 / np.sqrt(K)) if lam is None else float(lam)
    L = np.zeros_like(S)
    E = np.zeros_like(S)
    Y = np.zeros_like(S)
    converged, iters = False, 0
    for it in range(1, max_iter + 1):
        U, s, Vt = np.linalg.svd(S - E + (1.0 / rho) * Y, full_matrices=False)
        s_thr = np.maximum(s - 1.0 / rho, 0.0)
        L = (U * s_thr) @ Vt
        E = _soft_threshold(S - L + (1.0 / rho) * Y, lam_eff / rho)
        residual = S - L - E
        Y = Y + rho * residual
        rel = np.linalg.norm(residual, ord="fro") / (np.linalg.norm(S, ord="fro") + 1e-12)
        iters = it
        if rel < tol:
            converged = True
            break
    return L, E, {"lambda": lam_eff, "converged": converged, "iterations": iters}


def postprocess_graph(W: np.ndarray) -> np.ndarray:
    Z = np.maximum(W, 0.0)
    Z = 0.5 * (Z + Z.T)
    np.fill_diagonal(Z, 0.0)
    m = Z.max()
    if m > 0:
        Z = Z / m
    return Z


def spectral_gap(W: np.ndarray) -> float:
    S = 0.5 * (W + W.T)
    np.fill_diagonal(S, 0.0)
    deg = S.sum(axis=1)
    inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    D_inv = np.diag(inv_sqrt)
    L_norm = np.eye(S.shape[0]) - D_inv @ S @ D_inv
    eigvals = np.linalg.eigvalsh(L_norm)
    return float(eigvals[1]) if S.shape[0] > 1 else 0.0


def connectivity_stats(W: np.ndarray, benign_ids: Sequence[int], byz_ids: Sequence[int]) -> Dict[str, float]:
    bb_vals = [W[i, j] for i in benign_ids for j in benign_ids if i != j]
    cross_vals = [W[i, j] for i in byz_ids for j in benign_ids]
    w_b = float(np.mean(bb_vals)) if bb_vals else 0.0
    w_c = float(np.mean(cross_vals)) if cross_vals else 0.0
    return {"W_benign": w_b, "W_cross": w_c, "connectivity_gap": w_b - w_c}


def personalized_pagerank(W: np.ndarray, beta: float = 0.85, seed: Optional[np.ndarray] = None, tol: float = 1e-6, max_iter: int = 200, return_history: bool = False):
    K = W.shape[0]
    if seed is None:
        seed = np.ones(K, dtype=np.float64) / K
    else:
        seed = np.asarray(seed, dtype=np.float64)
        seed = seed / (seed.sum() + 1e-12)
    deg = W.sum(axis=1, keepdims=True)
    P = np.divide(W, deg, out=np.zeros_like(W), where=deg > 0)
    zero_rows = np.where(deg.squeeze() <= 1e-12)[0]
    if len(zero_rows) > 0:
        P[zero_rows, :] = 1.0 / K
    v = seed.copy()
    hist = [v.copy()]
    for _ in range(max_iter):
        v_new = (1 - beta) * seed + beta * (P.T @ v)
        if return_history:
            hist.append(v_new.copy())
        if np.linalg.norm(v_new - v, ord=1) < tol:
            v = v_new
            break
        v = v_new
    v = v / (v.sum() + 1e-12)
    return (v, hist) if return_history else v


def node_strength(W: np.ndarray) -> np.ndarray:
    return W.sum(axis=1)


def _save(fig, base: Path) -> None:
    fig.savefig(str(base) + ".png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(str(base) + ".pdf", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _build_graph(W: np.ndarray, qtop: float, qweak: float):
    K = W.shape[0]
    vals = np.array([W[i, j] for i in range(K) for j in range(i + 1, K)])
    top_thr = float(np.quantile(vals, qtop)) if len(vals) else 0.0
    weak_thr = float(np.quantile(vals, qweak)) if len(vals) else 0.0
    G = nx.Graph()
    G.add_nodes_from(range(K))
    for i in range(K):
        for j in range(i + 1, K):
            w = float(W[i, j])
            if w >= weak_thr:
                G.add_edge(i, j, weight=w, strong=(w >= top_thr))
    return G


def plot_graph_before(S_graph, byz_ids, attack, out_dir, layout_seed, top_edge_quantile, weak_edge_quantile):
    K = S_graph.shape[0]
    benign_ids = [i for i in range(K) if i not in byz_ids]
    G = _build_graph(S_graph, top_edge_quantile, weak_edge_quantile)
    pos = nx.spring_layout(G, seed=layout_seed, weight="weight")
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    strong = [(u, v) for u, v, d in G.edges(data=True) if d["strong"]]
    weak = [(u, v) for u, v, d in G.edges(data=True) if not d["strong"]]
    nx.draw_networkx_edges(G, pos, edgelist=weak, edge_color="gray", style="dashed", alpha=0.45, width=1.1, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=strong, edge_color="#4C78A8", style="solid", alpha=0.65, width=1.8, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=benign_ids, node_color="#377eb8", node_size=450, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=list(byz_ids), node_color="#d62728", node_size=500, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=9, font_color="white", ax=ax)
    ax.set_title("(a) Observed graph before purification", fontsize=13)
    ax.text(0.01, -0.08, "Before purification, noisy edges obscure the trust structure.", transform=ax.transAxes, fontsize=10)
    ax.axis("off")
    _save(fig, out_dir / f"obs3a_graph_before_purification_{attack}")


def plot_graph_after(L_hat, trust, byz_ids, attack, out_dir, layout_seed, top_edge_quantile, weak_edge_quantile):
    K = L_hat.shape[0]
    benign_ids = [i for i in range(K) if i not in byz_ids]
    G = _build_graph(L_hat, top_edge_quantile, weak_edge_quantile)
    pos = nx.spring_layout(G, seed=layout_seed, weight="weight", k=1.15)
    sizes = 380 + 620 * (trust / (trust.max() + 1e-12))
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    bb_edges, byz_edges = [], []
    for u, v, d in G.edges(data=True):
        (bb_edges if (u in benign_ids and v in benign_ids and d["strong"]) else byz_edges).append((u, v))
    nx.draw_networkx_edges(G, pos, edgelist=bb_edges, edge_color="#4C78A8", width=2.0, alpha=0.72, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=byz_edges, edge_color="gray", style="dashed", width=1.2, alpha=0.45, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=benign_ids, node_color="#377eb8", node_size=[sizes[i] for i in benign_ids], ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=list(byz_ids), node_color="#d62728", node_size=[sizes[i] for i in byz_ids], ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=9, font_color="white", ax=ax)
    ax.set_title("(b) Purified graph $L_{hat}$", fontsize=13)
    ax.text(0.01, -0.08, "Purification amplifies the separation between the benign core and Byzantine fringe.", transform=ax.transAxes, fontsize=10)
    ax.text(0.02, 0.97, "Benign core: high connectivity", transform=ax.transAxes, fontsize=9, color="#1f4f88", va="top")
    ax.text(0.02, 0.92, "Byzantine fringe: weak integration", transform=ax.transAxes, fontsize=9, color="#a61c1c", va="top")
    ax.axis("off")
    _save(fig, out_dir / f"obs3b_graph_after_purification_{attack}")


def plot_trust_alignment(trust, strength, byz_ids, attack, out_dir):
    K = len(trust)
    byz_set = set(byz_ids)
    order = np.argsort(-trust)
    trust_n = trust / (trust.max() + 1e-12)
    strength_n = strength / (strength.max() + 1e-12)
    fig, ax1 = plt.subplots(figsize=(7.6, 4.6))
    bar_colors = ["#d62728" if i in byz_set else "#377eb8" for i in order]
    ax1.bar(np.arange(K), trust_n[order], color=bar_colors, alpha=0.9)
    ax1.set_ylabel("Normalized PPR trust score", fontsize=11)
    ax1.set_xlabel("Clients sorted by trust score", fontsize=11)
    ax2 = ax1.twinx()
    ax2.plot(np.arange(K), strength_n[order], "o--", color="gray", linewidth=1.6, markersize=4)
    ax2.set_ylabel("Normalized node strength", fontsize=11)
    labels = [str(i) for i in order]
    ax1.set_xticks(np.arange(K))
    ax1.set_xticklabels(labels, fontsize=9)
    for tick, idx in zip(ax1.get_xticklabels(), order):
        tick.set_color("#d62728" if idx in byz_set else "#377eb8")
    byz_positions = [p for p, idx in enumerate(order) if idx in byz_set]
    if byz_positions == list(range(min(byz_positions), max(byz_positions) + 1)) and len(byz_positions) > 0:
        if min(byz_positions) > 0:
            ax1.axvline(min(byz_positions) - 0.5, color="black", linestyle="--", alpha=0.5)
    benign_ids = [i for i in range(K) if i not in byz_set]
    bmean, ymean = float(np.mean(trust[benign_ids])), float(np.mean(trust[list(byz_set)]))
    ax1.text(0.01, 0.95, f"Benign mean trust: {bmean:.4f}\nByzantine mean trust: {ymean:.4f}\nSeparation gap: {bmean - ymean:.4f}", transform=ax1.transAxes, va="top", fontsize=9)
    ax1.set_title("(c) Trust scores align with structural support", fontsize=13)
    ax1.text(0.62, 0.90, "Higher connectivity aligns with higher trust", transform=ax1.transAxes, fontsize=9, color="#444444")
    _save(fig, out_dir / f"obs3c_trust_scores_alignment_{attack}")


def plot_structural_improvement(gap_before, gap_after, conn_before, conn_after, attack, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.9))
    axes[0].bar(["Before", "After"], [gap_before, gap_after], color=["#bdbdbd", "#377eb8"])
    axes[0].set_title("Spectral gap", fontsize=12)
    axes[0].set_ylabel("Value", fontsize=10)
    axes[0].annotate("increase", xy=(1, gap_after), xytext=(0.55, max(gap_before, gap_after) * 1.05), arrowprops=dict(arrowstyle="->", color="#377eb8"), color="#377eb8", fontsize=9)
    c_before = conn_before["connectivity_gap"]
    c_after = conn_after["connectivity_gap"]
    axes[1].bar(["Before", "After"], [c_before, c_after], color=["#bdbdbd", "#377eb8"])
    axes[1].set_title("Connectivity gap", fontsize=12)
    axes[1].annotate("increase", xy=(1, c_after), xytext=(0.52, max(c_before, c_after) * 1.05 + 1e-9), arrowprops=dict(arrowstyle="->", color="#377eb8"), color="#377eb8", fontsize=9)
    fig.suptitle("(d) Structural improvement after purification", fontsize=13)
    fig.text(0.15, -0.02, "Graph purification increases spectral and connectivity separation.", fontsize=10)
    _save(fig, out_dir / f"obs3d_structural_improvement_{attack}")


def plot_convergence(history, byz_ids, attack, out_dir):
    picks = [0, 5, 10, 20, len(history) - 1]
    picks = sorted(set([p for p in picks if p < len(history)]))
    K = len(history[0])
    x = np.arange(K)
    byz_set = set(byz_ids)
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    for p in picks:
        h = history[p]
        col = "#377eb8" if p != picks[-1] else "#1b4f72"
        ax.plot(x, h, marker="o", linewidth=1.2, alpha=0.8, label=f"iter {p}", color=col)
    for i in x:
        if i in byz_set:
            ax.axvspan(i - 0.5, i + 0.5, color="#fddede", alpha=0.3)
    ax.set_title("(e) PPR trust convergence (appendix)")
    ax.set_xlabel("Client ID")
    ax.set_ylabel("Trust score")
    ax.legend(fontsize=8)
    _save(fig, out_dir / f"obs3e_ppr_convergence_{attack}")


def main() -> None:
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 13, "axes.labelsize": 11, "xtick.labelsize": 9, "ytick.labelsize": 9})
    p = argparse.ArgumentParser()
    p.add_argument("--logits-npy", required=True)
    p.add_argument("--out-dir", default="observations/outputs/obs3")
    p.add_argument("--kappa", type=int, default=5)
    p.add_argument("--byzantine-ids", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--attack", default="alie", choices=["gaussian", "label_flip", "targeted", "alie", "all"])
    p.add_argument("--target-class", type=int, default=0)
    p.add_argument("--gaussian-sigma", type=float, default=1.0)
    p.add_argument("--alie-z-scale", type=float, default=1.5)
    p.add_argument("--ppr-beta", type=float, default=0.85)
    p.add_argument("--rpca-lambda", type=float, default=None)
    p.add_argument("--rpca-rho", type=float, default=1.0)
    p.add_argument("--rpca-max-iter", type=int, default=500)
    p.add_argument("--rpca-tol", type=float, default=1e-4)
    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--layout-seed", type=int, default=0)
    p.add_argument("--top-edge-quantile", type=float, default=0.65)
    p.add_argument("--weak-edge-quantile", type=float, default=0.40)
    p.add_argument("--plot-convergence", action="store_true")
    args = p.parse_args()

    benign_logits = load_logits(args.logits_npy)
    K, N, C = benign_logits.shape
    byz_ids, benign_ids = validate_inputs(benign_logits, args.byzantine_ids)
    out_dir = resolve_out_dir(args.out_dir)
    attacks = ["gaussian", "label_flip", "targeted", "alie"] if args.attack == "all" else [args.attack]

    for attack in attacks:
        observed_logits = build_observed_logits(benign_logits, byz_ids, attack, args)
        S_observed = jaccard_similarity_matrix(observed_logits, kappa=args.kappa)
        S_graph = S_observed.copy()
        np.fill_diagonal(S_graph, 0.0)

        L_raw, _, rpca_info = solve_rpca_admm(S_graph, args.rpca_lambda, args.rpca_rho, args.rpca_max_iter, args.rpca_tol)
        L_hat = postprocess_graph(L_raw)

        degree_before = node_strength(S_graph)
        degree_after = node_strength(L_hat)
        ppr_out = personalized_pagerank(
            L_hat,
            beta=args.ppr_beta,
            seed=None,
            tol=args.eps,
            max_iter=200,
            return_history=args.plot_convergence,
        )
        if args.plot_convergence:
            trust, history = ppr_out
        else:
            trust = ppr_out
            history = None

        conn_before = connectivity_stats(S_graph, benign_ids, byz_ids)
        conn_after = connectivity_stats(L_hat, benign_ids, byz_ids)
        gap_before = spectral_gap(S_graph)
        gap_after = spectral_gap(L_hat)

        plot_graph_before(S_graph, byz_ids, attack, out_dir, args.layout_seed, args.top_edge_quantile, args.weak_edge_quantile)
        plot_graph_after(L_hat, trust, byz_ids, attack, out_dir, args.layout_seed, args.top_edge_quantile, args.weak_edge_quantile)
        plot_trust_alignment(trust, degree_after, byz_ids, attack, out_dir)
        plot_structural_improvement(gap_before, gap_after, conn_before, conn_after, attack, out_dir)
        if args.plot_convergence:
            plot_convergence(history, byz_ids, attack, out_dir)

        trust_n = trust / (trust.max() + 1e-12)
        strength_n = degree_after / (degree_after.max() + 1e-12)
        metrics = {
            "logits_npy": args.logits_npy,
            "K": K,
            "N_pub": N,
            "C": C,
            "kappa": args.kappa,
            "attack": "label_flip_swap" if attack == "label_flip" else attack,
            "byzantine_ids": byz_ids,
            "benign_ids": benign_ids,
            "rpca_lambda": rpca_info["lambda"],
            "rpca_converged": rpca_info["converged"],
            "rpca_iterations": rpca_info["iterations"],
            "spectral_gap_before": gap_before,
            "spectral_gap_after": gap_after,
            "spectral_gap_gain": gap_after - gap_before,
            "connectivity_before": conn_before,
            "connectivity_after": conn_after,
            "connectivity_gap_gain": conn_after["connectivity_gap"] - conn_before["connectivity_gap"],
            "trust_scores": trust.tolist(),
            "trust_scores_normalized": trust_n.tolist(),
            "node_strength": degree_after.tolist(),
            "node_strength_normalized": strength_n.tolist(),
            "benign_mean_trust": float(np.mean(trust[benign_ids])),
            "byzantine_mean_trust": float(np.mean(trust[byz_ids])),
            "trust_separation_gap": float(np.mean(trust[benign_ids]) - np.mean(trust[byz_ids])),
            "benign_mean_strength": float(np.mean(degree_after[benign_ids])),
            "byzantine_mean_strength": float(np.mean(degree_after[byz_ids])),
            "strength_separation_gap": float(np.mean(degree_after[benign_ids]) - np.mean(degree_after[byz_ids])),
            "degree_before": degree_before.tolist(),
        }
        save_json(out_dir / f"obs3_metrics_{attack}.json", metrics)


if __name__ == "__main__":
    main()
