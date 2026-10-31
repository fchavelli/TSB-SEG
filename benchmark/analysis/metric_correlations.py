"""Pairwise correlation of the evaluation metrics (appendix).

Pearson correlations between the CPD metrics and between the SD metrics, over
every grid run (Non-Guided and Guided) of the methods on the 8 datasets:
``fig_metric_correlations``, two lower-triangular heatmaps with a shared colour
bar.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from analysis.data import GUIDED_FROM_NON_GUIDED, load_data
from analysis.helpers import MAIN_DATASETS, savefig
from analysis.runtime import INK, INK_2, PANEL_RC

CPD_METRICS = {
    "f1_score": "$F_1$",
    "covering_score": "Covering",
    "gaussian_f1_score": "Gaussian $F_1$",
    "bidirectional_covering_score": "BiC",
}
SD_METRICS = {
    "adjusted_rand_index_score": "ARI",
    "adjusted_mutual_info_score": "AMI",
    "weighted_adjusted_rand_index_score": "WARI",
    "state_matching_score": "SMS",
}

# Single-hue sequential ramp (dataviz reference blue, steps 100 -> 650). Every
# off-diagonal correlation lies in [0.5, 1], so the scale starts at 0.5.
_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#104281"]
CMAP = LinearSegmentedColormap.from_list("seq_blue", _RAMP)
VMIN, VMAX = 0.5, 1.0

# Drawn at print size: 0.7 of the NeurIPS text width (5.5 in).
FIG_W, FIG_H = 3.85, 1.72


def _runs(d) -> pd.DataFrame:
    """Every grid run of the paper's methods on the 8 main datasets, once.

    The Guided rows of ``GUIDED_FROM_NON_GUIDED`` are copies of
    their Non-Guided runs (``data.guided_from_non_guided``) and are dropped.
    """
    g = d.df_grid_g[~d.df_grid_g["algorithm"].isin(GUIDED_FROM_NON_GUIDED)]
    df = pd.concat([d.df_grid_ng, g], ignore_index=True)
    df = df[df["algorithm"] != "random"]
    return df[df["dataset"].isin(MAIN_DATASETS)]


def _heatmap(ax, df, metrics: dict, title: str):
    cols = list(metrics)
    sub = df[cols].dropna(how="any")
    corr = sub.corr().to_numpy()
    names = list(metrics.values())
    n = len(cols)
    # Lower triangle without the diagonal: rows 1..n-1, columns 0..n-2.
    for i in range(1, n):
        for j in range(i):
            v = corr[i, j]
            x, y = j, i - 1
            ax.add_patch(
                plt.Rectangle(
                    (x, y),
                    1,
                    1,
                    facecolor=CMAP((v - VMIN) / (VMAX - VMIN)),
                    edgecolor="white",
                    linewidth=1.2,
                )
            )
            ax.text(
                x + 0.5,
                y + 0.5,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=6.2,
                color="white" if v >= 0.8 else INK,
            )
    ax.set_xlim(0, n - 1)
    ax.set_ylim(n - 1, 0)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(n - 1) + 0.5)
    ax.set_xticklabels(names[:-1], rotation=30, ha="right", rotation_mode="anchor")
    ax.set_yticks(np.arange(n - 1) + 0.5)
    ax.set_yticklabels(names[1:])
    ax.tick_params(length=0, pad=2)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.grid(False)
    ax.set_title(title, fontsize=7, color=INK, pad=3)
    off = corr[np.tril_indices(n, k=-1)]
    return float(off.min()), float(off.max()), len(sub)


def make_figure(d=None):
    if d is None:
        d = load_data()
    df = _runs(d)

    with mpl.rc_context({**PANEL_RC, "xtick.labelsize": 6.2, "ytick.labelsize": 6.2}):
        fig = plt.figure(figsize=(FIG_W, FIG_H))
        # Two heatmaps of 3 x 3 cells and one shared colour bar, in inches.
        cell, bottom, left_pad = 0.36, 0.42, 0.62
        size = 3 * cell
        ax_c = fig.add_axes([left_pad / FIG_W, bottom / FIG_H, size / FIG_W, size / FIG_H])
        ax_s = fig.add_axes(
            [(left_pad + size + 0.62) / FIG_W, bottom / FIG_H, size / FIG_W, size / FIG_H]
        )
        cpd_min, cpd_max, n_cpd = _heatmap(ax_c, df, CPD_METRICS, "CPD metrics")
        sd_min, sd_max, n_sd = _heatmap(ax_s, df, SD_METRICS, "SD metrics")
        cax = fig.add_axes(
            [(left_pad + 2 * size + 0.80) / FIG_W, bottom / FIG_H, 0.07 / FIG_W, size / FIG_H]
        )
        cb = fig.colorbar(
            mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(VMIN, VMAX), cmap=CMAP), cax=cax
        )
        cb.set_ticks([0.5, 0.75, 1.0])
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=5.6, length=2, width=0.5, colors=INK_2)
        cb.set_label("Pearson $r$", fontsize=6, color=INK, labelpad=2)
        savefig(fig, "fig_metric_correlations", tight=False)

    print(f"CPD: {n_cpd} runs, off-diagonal r in [{cpd_min:.2f}, {cpd_max:.2f}]")
    print(f"SD : {n_sd} runs, off-diagonal r in [{sd_min:.2f}, {sd_max:.2f}]")
