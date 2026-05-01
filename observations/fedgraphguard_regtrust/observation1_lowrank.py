from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from .obs_utils import ensure_dir, make_run_id, parse_obs1_key, save_json
from .plotting import plot_obs1_cumulative_energy_grid


def jaccard_from_logits(arr: np.ndarray, kappa: int) -> np.ndarray:
    k, n, c = arr.shape
    kk = min(kappa, c)
    topk = np.argpartition(arr, kth=c - kk, axis=2)[:, :, -kk:]
    s = np.eye(k, dtype=float)
    for i in range(k):
        for j in range(i + 1, k):
            vals = []
            for a, b in zip(topk[i], topk[j]):
                sa, sb = set(a.tolist()), set(b.tolist())
                vals.append(len(sa & sb) / len(sa | sb))
            s[i, j] = s[j, i] = float(np.mean(vals))
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logits-npz", required=True)
    p.add_argument("--out-dir", default="observations/outputs/obs1")
    p.add_argument("--dataset", default="cifar10", choices=["mnist", "cifar10", "tinyimagenet"])
    p.add_argument("--kappa", type=int, default=5)
    args = p.parse_args()

    out_dir = ensure_dir(args.out_dir)
    run_id = make_run_id(args.dataset, obs_name="obs1")
    csv_path = out_dir / f"{run_id}_metrics.csv"
    json_path = out_dir / f"{run_id}_metrics.json"
    png_path = out_dir / f"{run_id}_cumulative_energy.png"
    pdf_path = out_dir / f"{run_id}_cumulative_energy.pdf"

    payload = np.load(args.logits_npz)
    rows = []
    grouped = defaultdict(lambda: defaultdict(list))

    for key in payload.files:
        meta = parse_obs1_key(key)
        arr = np.asarray(payload[key], dtype=float)
        if arr.ndim != 3:
            raise ValueError(f"{key}: expected (K, N_pub, C), got {arr.shape}")
        s = jaccard_from_logits(arr, kappa=args.kappa)
        sig = np.linalg.svd(s, compute_uv=False)
        energy = np.cumsum(sig**2) / np.sum(sig**2)
        p_sig = sig / np.sum(sig)
        eff = float(np.exp(-np.sum(p_sig * np.log(p_sig + 1e-12))))
        r90 = int(np.searchsorted(energy, 0.90) + 1)
        r95 = int(np.searchsorted(energy, 0.95) + 1)

        beta = float(meta["beta"])
        mode = meta["model_mode"]
        grouped[beta][mode].append(energy)

        rows.append(
            {
                "dataset": meta["dataset"], "beta": beta, "model_mode": mode, "seed": int(meta["seed"]),
                "num_clients": arr.shape[0], "num_public": arr.shape[1], "kappa": args.kappa,
                "diag_mode": "included", "r90": r90, "r95": r95, "effective_rank": eff,
                "energy_rank1": float(energy[min(0, len(energy)-1)]),
                "energy_rank2": float(energy[min(1, len(energy)-1)]),
                "energy_rank3": float(energy[min(2, len(energy)-1)]),
                "energy_rank5": float(energy[min(4, len(energy)-1)]),
                "energy_rank10": float(energy[min(9, len(energy)-1)]),
            }
        )

    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    betas = sorted(grouped.keys())
    modes = ["homogeneous", "heterogeneous"]
    curves = {}
    for b in betas:
        curves[b] = {}
        for m in modes:
            mat = np.stack(grouped[b][m], axis=0)
            curves[b][m] = {"mean": mat.mean(axis=0), "std": mat.std(axis=0)}

    plot_obs1_cumulative_energy_grid(curves, betas, modes, args.dataset, str(png_path), str(pdf_path))

    save_json(json_path, {
        "run_id": run_id,
        "dataset": args.dataset,
        "kappa": args.kappa,
        "diag_mode": "included",
        "curves": {
            str(b): {m: {"mean": curves[b][m]["mean"].tolist(), "std": curves[b][m]["std"].tolist()} for m in modes}
            for b in betas
        },
    })

    print(csv_path)
    print(json_path)
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
