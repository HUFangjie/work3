"""Observation-2 for FedGraphGuard (ReG-Trust): Byzantine row-sparse perturbation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))
    from common_metrics import (
        apply_attack,
        byzantine_energy_concentration,
        jaccard_similarity_matrix,
        row_energy,
        sparsity_ratio,
    )
    from plotting import plot_exp2_row_energy
else:
    from .common_metrics import (
        apply_attack,
        byzantine_energy_concentration,
        jaccard_similarity_matrix,
        row_energy,
        sparsity_ratio,
    )
    from .plotting import plot_exp2_row_energy


ATTACKS = ("gaussian", "label_flip", "targeted", "alie")


def run_observation2(
    benign_logits_npy: str,
    byzantine_ids: Sequence[int],
    out_dir: str,
    kappa: int = 5,
    seed: int = 42,
) -> Dict[str, float]:
    """Run Observation-2 using one benign reference tensor.

    Input npy shape: (K, N_pub, C) for benign logits.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    benign = np.asarray(np.load(benign_logits_npy), dtype=np.float64)
    if benign.ndim != 3:
        raise ValueError(f"Expected (K,N_pub,C), got {benign.shape}")

    k_clients = benign.shape[0]
    benign_list = [benign[i] for i in range(k_clients)]
    s_benign = jaccard_similarity_matrix(benign_list, kappa=kappa)

    benign_mean = np.mean(benign, axis=0)
    benign_std = np.std(benign, axis=0)

    atk_to_energy = {}
    atk_to_concentration = {}
    table = {}

    for attack in ATTACKS:
        mixed = benign.copy()
        for bid in byzantine_ids:
            mixed[bid] = apply_attack(
                mixed[bid],
                attack_type=attack,
                rng=rng,
                target_class=0,
                sigma=1.0,
                benign_mean=benign_mean,
                benign_std=benign_std,
                z_scale=1.5,
            )

        s_obs = jaccard_similarity_matrix([mixed[i] for i in range(k_clients)], kappa=kappa)
        e = s_obs - s_benign

        re_norm = row_energy(e, normalize=True)
        conc = byzantine_energy_concentration(e, byzantine_ids=byzantine_ids)
        spr = sparsity_ratio(e, threshold=1e-3)

        atk_to_energy[attack] = re_norm
        atk_to_concentration[attack] = conc
        table[attack] = {
            "byzantine_energy_concentration": float(conc),
            "sparsity_ratio": float(spr),
        }

    plot_exp2_row_energy(
        attack_to_energy=atk_to_energy,
        byzantine_ids=byzantine_ids,
        attack_to_concentration=atk_to_concentration,
        out_file=str(out / "obs2_row_energy.png"),
    )

    result = {
        "note": "FedGraphGuard is ReG-Trust; this file implements Observation-2 independently.",
        "kappa": kappa,
        "byzantine_ids": list(byzantine_ids),
        "attack_stats": table,
    }
    (out / "obs2_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    return {"attacks": float(len(ATTACKS)), "clients": float(k_clients)}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Observation-2 (row-sparse perturbation).")
    parser.add_argument("--benign-logits-npy", required=True, help="Path to benign logits tensor (K,N_pub,C)")
    parser.add_argument("--out-dir", default="observations/outputs/obs2", help="Output directory")
    parser.add_argument("--kappa", type=int, default=5, help="Top-k classes for Jaccard")
    parser.add_argument("--byzantine-ids", nargs="+", type=int, default=[0, 1, 2, 3], help="Byzantine client ids")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    run_observation2(
        benign_logits_npy=args.benign_logits_npy,
        byzantine_ids=args.byzantine_ids,
        out_dir=args.out_dir,
        kappa=args.kappa,
        seed=args.seed,
    )
