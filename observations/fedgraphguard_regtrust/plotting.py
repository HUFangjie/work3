"""Standalone plotting utilities for FedGraphGuard/ReG-Trust observations."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


_PREFERRED_STYLES = ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot")
_AVAILABLE_STYLES = set(plt.style.available)
_STYLE_TO_USE = next((name for name in _PREFERRED_STYLES if name in _AVAILABLE_STYLES), "default")
plt.style.use(_STYLE_TO_USE)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_exp1_energy_curves(
    alpha_to_curve: Mapping[float, np.ndarray],
    alpha_to_std: Mapping[float, np.ndarray] | None,
    out_file: str,
) -> None:
    path = Path(out_file)
    _ensure_parent(path)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for alpha in sorted(alpha_to_curve.keys()):
        curve = np.asarray(alpha_to_curve[alpha])
        x = np.arange(1, len(curve) + 1)
        ax.plot(x, curve, marker="o", linewidth=2, label=f"α={alpha}")

        if alpha_to_std is not None and alpha in alpha_to_std:
            std = np.asarray(alpha_to_std[alpha])
            ax.fill_between(x, np.clip(curve - std, 0, 1), np.clip(curve + std, 0, 1), alpha=0.18)

    ax.axhline(0.9, linestyle="--", color="gray", linewidth=1.5, label="90% energy")
    ax.set_xlabel("Singular Value Index")
    ax.set_ylabel("Cumulative Energy Ratio")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Observation-1: Low-rank Structure of Similarity Matrix")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_exp2_row_energy(
    attack_to_energy: Mapping[str, np.ndarray],
    byzantine_ids: Iterable[int],
    attack_to_concentration: Mapping[str, float],
    out_file: str,
) -> None:
    path = Path(out_file)
    _ensure_parent(path)
    byz = set(int(i) for i in byzantine_ids)

    attacks = list(attack_to_energy.keys())
    n = len(attacks)
    rows = int(np.ceil(n / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(11, 4.2 * rows), squeeze=False)

    for idx, atk in enumerate(attacks):
        r, c = divmod(idx, 2)
        ax = axes[r][c]
        energy = np.asarray(attack_to_energy[atk])
        x = np.arange(len(energy))
        colors = ["tab:red" if i in byz else "tab:blue" for i in x]

        ax.bar(x, energy, color=colors, alpha=0.85)
        ax.set_title(f"{atk} attack")
        ax.set_xlabel("Client Index")
        ax.set_ylabel("Normalized Row Energy")
        ax.text(
            0.99,
            0.95,
            f"Concentration: {attack_to_concentration[atk] * 100:.1f}%",
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.7},
        )

    total_axes = rows * 2
    for idx in range(n, total_axes):
        r, c = divmod(idx, 2)
        axes[r][c].axis("off")

    fig.suptitle("Observation-2: Row-sparse Byzantine Perturbation", y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_exp3_trust_scores(
    trust_scores: np.ndarray,
    byzantine_ids: Sequence[int],
    out_file: str,
) -> None:
    path = Path(out_file)
    _ensure_parent(path)

    trust = np.asarray(trust_scores, dtype=np.float64)
    order = np.argsort(-trust)
    sorted_scores = trust[order]

    byz = set(int(i) for i in byzantine_ids)
    colors = ["tab:red" if int(i) in byz else "tab:blue" for i in order]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(sorted_scores))
    ax.bar(x, sorted_scores, color=colors, alpha=0.9)
    ax.axhline(1.0 / len(sorted_scores), linestyle="--", color="gray", linewidth=1.3)
    ax.set_xlabel("Client (sorted by trust)")
    ax.set_ylabel("PPR Trust Score")
    ax.set_title("Observation-3: PPR Trust Separation After Purification")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(i)) for i in order], rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
