#!/usr/bin/env python
"""Plot AURC curve and annotate last-20%-round average AURC.

Usage:
  python analysis/plot_tail_aurc.py --metrics_csv logs/<exp>_metrics_rounds.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_round_aurc(path: str) -> List[Tuple[int, float]]:
    rows: List[Tuple[int, float]] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                rd = int(float(r["round"]))
                aurc = float(r["test_aurc"])
            except Exception:
                continue
            if math.isfinite(aurc):
                rows.append((rd, aurc))
    rows.sort(key=lambda x: x[0])
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--metrics_csv", type=str, required=True)
    p.add_argument("--out_png", type=str, default=None)
    args = p.parse_args()

    pts = load_round_aurc(args.metrics_csv)
    if not pts:
        raise RuntimeError("No valid (round, test_aurc) points found in metrics CSV.")

    rounds = np.array([x[0] for x in pts], dtype=np.int64)
    aurc = np.array([x[1] for x in pts], dtype=np.float64)
    tail_n = max(1, int(np.ceil(0.2 * len(aurc))))
    tail_avg = float(np.mean(aurc[-tail_n:]))

    plt.figure(figsize=(8, 4.5))
    plt.plot(rounds, aurc, label="Test AURC", linewidth=1.8)
    plt.axvspan(rounds[-tail_n], rounds[-1], alpha=0.15, color="orange", label="Last 20% rounds")
    plt.axhline(tail_avg, linestyle="--", color="red", label=f"Tail20 Avg AURC={tail_avg:.4f}")
    plt.xlabel("Round")
    plt.ylabel("AURC")
    plt.title("AURC vs Round (with last-20% average)")
    plt.legend()
    plt.tight_layout()

    out_png = args.out_png
    if out_png is None:
        out_png = args.metrics_csv.replace(".csv", "_aurc_tail20.png")
    plt.savefig(out_png, dpi=160)
    print(f"Saved plot: {out_png}")
    print(f"Tail20 Avg AURC: {tail_avg:.10f} (tail_n={tail_n}, total={len(aurc)})")


if __name__ == "__main__":
    main()

