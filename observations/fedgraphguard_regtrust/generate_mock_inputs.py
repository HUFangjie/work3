"""Generate mock inputs for ReG-Trust observation scripts.

This is intended for smoke testing the observation pipeline when real logits are
not yet exported from a training run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _make_benign_logits(k: int, n_pub: int, c: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Shared global template + client-specific perturbation to induce low-rank-ish structure.
    template = rng.normal(0.0, 1.0, size=(n_pub, c))
    client_factors = rng.normal(0.0, 0.8, size=(k, 3))
    basis = rng.normal(0.0, 1.0, size=(3, n_pub, c))

    logits = np.zeros((k, n_pub, c), dtype=np.float64)
    for i in range(k):
        structured = np.tensordot(client_factors[i], basis, axes=(0, 0))
        noise = rng.normal(0.0, 0.2, size=(n_pub, c))
        logits[i] = template + structured + noise
    return logits


def generate_mock_inputs(
    out_dir: str,
    k: int = 20,
    n_pub: int = 1000,
    c: int = 10,
    seeds_per_alpha: int = 5,
    alphas: tuple[float, ...] = (0.1, 0.3, 0.5, 1.0),
    seed: int = 2026,
) -> None:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    # For obs2/obs3
    benign = _make_benign_logits(k=k, n_pub=n_pub, c=c, seed=seed)
    np.save(root / "benign_logits.npy", benign)

    # For obs1
    obs1_payload = {}
    for alpha in alphas:
        for s in range(seeds_per_alpha):
            local_seed = int(rng.integers(1, 10_000_000))
            logits = _make_benign_logits(k=k, n_pub=n_pub, c=c, seed=local_seed)
            # Vary heterogeneity by alpha: lower alpha => larger client variance
            scale = 1.0 + (1.0 / max(alpha, 1e-6)) * 0.15
            centered = logits - np.mean(logits, axis=0, keepdims=True)
            logits = np.mean(logits, axis=0, keepdims=True) + scale * centered
            key = f"alpha_{alpha}_seed_{s}"
            obs1_payload[key] = logits

    np.savez(root / "obs1_logits.npz", **obs1_payload)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate mock inputs for observation scripts.")
    parser.add_argument("--out-dir", default="observations/mock_inputs", help="Output directory")
    parser.add_argument("--clients", type=int, default=20, help="Number of clients K")
    parser.add_argument("--public-samples", type=int, default=1000, help="Public set size N_pub")
    parser.add_argument("--num-classes", type=int, default=10, help="Number of classes C")
    parser.add_argument("--seeds-per-alpha", type=int, default=5, help="How many runs per alpha for obs1")
    parser.add_argument("--seed", type=int, default=2026, help="RNG seed")
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    generate_mock_inputs(
        out_dir=args.out_dir,
        k=args.clients,
        n_pub=args.public_samples,
        c=args.num_classes,
        seeds_per_alpha=args.seeds_per_alpha,
        seed=args.seed,
    )
