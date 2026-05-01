from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def plot_obs1_cumulative_energy_grid(grouped_curves, betas, model_modes, dataset, out_png, out_pdf=None, y_min=0.75, y_max=1.0):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    axes = axes.flatten()
    colors = {"homogeneous": "tab:blue", "heterogeneous": "tab:orange"}

    for i, beta in enumerate(betas):
        ax = axes[i]
        for mode in model_modes:
            mean = grouped_curves[beta][mode]["mean"]
            std = grouped_curves[beta][mode]["std"]
            x = np.arange(1, len(mean) + 1)
            ax.plot(x, mean, label=mode.capitalize(), color=colors[mode], lw=2)
            ax.fill_between(x, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1), color=colors[mode], alpha=0.2)
        ax.axhline(0.90, ls="--", color="gray", lw=1.2)
        ax.set_title(f"( {chr(ord('a')+i)} ) Dirichlet beta = {beta}")
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(1, len(mean))
        ax.set_xlabel("Rank r")
        ax.set_ylabel("Cumulative spectral energy")

    fig.suptitle(f"Observation 1: Low-rank benign consensus on {dataset.upper()}")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels + ["90% threshold"] if "90% threshold" not in labels else labels, loc="lower center", ncol=3)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig(out_png, dpi=250)
    if out_pdf:
        fig.savefig(out_pdf)
    plt.close(fig)
