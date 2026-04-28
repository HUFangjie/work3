"""Observation-3 for FedGraphGuard (ReG-Trust): purification, connectivity, and PPR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

from .common_metrics import (
    apply_attack,
    connectivity_stats,
    jaccard_similarity_matrix,
    low_rank_rpca,
    personalized_pagerank,
    spectral_gap,
)
from .plotting import plot_exp3_trust_scores


def run_observation3(
    benign_logits_npy: str,
    byzantine_ids: Sequence[int],
    out_dir: str,
    kappa: int = 5,
    beta: float = 0.85,
    seed: int = 123,
) -> Dict[str, float]:
    """Run Observation-3 with ALIE attack and graph purification.

    Input npy shape: (K, N_pub, C).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    benign = np.asarray(np.load(benign_logits_npy), dtype=np.float64)
    if benign.ndim != 3:
        raise ValueError(f"Expected (K,N_pub,C), got {benign.shape}")

    k = benign.shape[0]
    all_ids = np.arange(k)
    byz = np.array(sorted(set(int(i) for i in byzantine_ids)), dtype=np.int64)
    benign_ids = np.array([i for i in all_ids if i not in set(byz)], dtype=np.int64)

    benign_mean = np.mean(benign, axis=0)
    benign_std = np.std(benign, axis=0)

    mixed = benign.copy()
    for bid in byz:
        mixed[bid] = apply_attack(
            mixed[bid],
            attack_type="alie",
            rng=rng,
            benign_mean=benign_mean,
            benign_std=benign_std,
            z_scale=1.5,
        )

    s_obs = jaccard_similarity_matrix([mixed[i] for i in range(k)], kappa=kappa)

    lam = 1.0 / np.sqrt(k)
    l_hat, e_hat = low_rank_rpca(s_obs, lam=lam, rho=1.0, max_iter=500, tol=1e-4)
    l_hat = np.clip(l_hat, a_min=0.0, a_max=None)
    if np.max(l_hat) > 0:
        l_hat = l_hat / np.max(l_hat)

    gap_before = spectral_gap(s_obs)
    gap_after = spectral_gap(l_hat)

    conn_before = connectivity_stats(s_obs, benign_ids=benign_ids, byzantine_ids=byz)
    conn_after = connectivity_stats(l_hat, benign_ids=benign_ids, byzantine_ids=byz)

    trust = personalized_pagerank(l_hat, beta=beta, tol=1e-6, max_iter=200)

    plot_exp3_trust_scores(
        trust_scores=trust,
        byzantine_ids=byz.tolist(),
        out_file=str(out / "obs3_ppr_trust_scores.png"),
    )

    result = {
        "note": "FedGraphGuard is ReG-Trust; this file implements Observation-3 independently.",
        "kappa": kappa,
        "beta": beta,
        "byzantine_ids": byz.tolist(),
        "spectral_gap": {
            "before_purification": float(gap_before),
            "after_purification": float(gap_after),
            "relative_gain": float((gap_after - gap_before) / (abs(gap_before) + 1e-12)),
        },
        "connectivity_before": conn_before,
        "connectivity_after": conn_after,
        "trust_scores": trust.tolist(),
        "sparse_component_l1": float(np.sum(np.abs(e_hat))),
    }
    (out / "obs3_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    return {"clients": float(k), "spectral_gap_after": float(gap_after)}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Observation-3 (purification + PPR trust).")
    parser.add_argument("--benign-logits-npy", required=True, help="Path to benign logits tensor (K,N_pub,C)")
    parser.add_argument("--out-dir", default="observations/outputs/obs3", help="Output directory")
    parser.add_argument("--kappa", type=int, default=5, help="Top-k classes for Jaccard")
    parser.add_argument("--beta", type=float, default=0.85, help="PPR damping")
    parser.add_argument("--byzantine-ids", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5], help="Byzantine ids")
    parser.add_argument("--seed", type=int, default=123, help="RNG seed")
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    run_observation3(
        benign_logits_npy=args.benign_logits_npy,
        byzantine_ids=args.byzantine_ids,
        out_dir=args.out_dir,
        kappa=args.kappa,
        beta=args.beta,
        seed=args.seed,
    )
