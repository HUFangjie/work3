"""Observation-1 for FedGraphGuard (ReG-Trust): low-rank benign similarity.

This script expects precomputed per-client logits on a shared public dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))
    from common_metrics import jaccard_similarity_matrix, singular_spectrum_metrics
    from plotting import plot_exp1_energy_curves
else:
    from .common_metrics import jaccard_similarity_matrix, singular_spectrum_metrics
    from .plotting import plot_exp1_energy_curves


def run_observation1(logits_npz: str, out_dir: str, kappa: int = 5) -> Dict[str, float]:
    """Run Observation-1 on one benign run and dump metrics/figures.

    npz format:
      - keys like alpha_0.1_seed_0 ... alpha_1.0_seed_4
      - each array shape: (K, N_pub, C)
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    payload = np.load(logits_npz)
    alpha_to_curves: Dict[float, List[np.ndarray]] = {}
    alpha_to_ranks: Dict[float, List[float]] = {}

    for key in payload.files:
        arr = np.asarray(payload[key], dtype=np.float64)
        if arr.ndim != 3:
            raise ValueError(f"{key}: expected (K,N_pub,C), got {arr.shape}")

        alpha_str = key.split("_")[1]
        alpha = float(alpha_str)

        logits_list = [arr[i] for i in range(arr.shape[0])]
        s = jaccard_similarity_matrix(logits_list, kappa=kappa)
        _, cum_energy, eff_rank = singular_spectrum_metrics(s)

        alpha_to_curves.setdefault(alpha, []).append(cum_energy)
        alpha_to_ranks.setdefault(alpha, []).append(eff_rank)

    alpha_to_mean = {a: np.mean(np.stack(v), axis=0) for a, v in alpha_to_curves.items()}
    alpha_to_std = {a: np.std(np.stack(v), axis=0) for a, v in alpha_to_curves.items()}
    rank_stats = {
        str(a): {
            "effective_rank_mean": float(np.mean(vals)),
            "effective_rank_std": float(np.std(vals)),
        }
        for a, vals in alpha_to_ranks.items()
    }

    plot_exp1_energy_curves(
        alpha_to_curve=alpha_to_mean,
        alpha_to_std=alpha_to_std,
        out_file=str(out / "obs1_cumulative_energy.png"),
    )

    result = {
        "note": "FedGraphGuard is ReG-Trust; this file implements Observation-1 independently.",
        "kappa": kappa,
        "effective_rank_stats": rank_stats,
    }
    (out / "obs1_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return {"alphas": len(alpha_to_mean), "kappa": float(kappa)}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Observation-1 (low-rank benign similarity).")
    parser.add_argument("--logits-npz", required=True, help="NPZ with benign logits grouped by alpha/seed")
    parser.add_argument("--out-dir", default="observations/outputs/obs1", help="Output directory")
    parser.add_argument("--kappa", type=int, default=5, help="Top-k classes for Jaccard")
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    run_observation1(logits_npz=args.logits_npz, out_dir=args.out_dir, kappa=args.kappa)
