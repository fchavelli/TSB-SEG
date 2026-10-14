"""Runtime per method, Non-Guided against Guided (appendix).

``fig_runtime_split_cpd`` and ``fig_runtime_split_sd``: per-series runtime of
the tuned configuration (selected per dataset on the score of the task), over
the 8 datasets. The light left half is the Non-Guided mode, the dark right half
the Guided mode; a method that runs in one mode only has a full-width box.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch

from analysis.data import load_data
from analysis.helpers import (
    ALGO_TASK,
    display_name,
    savefig,
    select_best_grid_per_dataset,
)
from analysis.runtime import FAMILY_COLOR, GRID, INK, PANEL_RC

METRIC_CPD = "bidirectional_covering_score"
METRIC_SD = "state_matching_score"
RUNTIME_COL = "execution_time_seconds"

# Extended scope only (timeouts or out of memory on whole datasets).
EXCLUDED_ALGOS = {"hidalgo", "tglad"}

# Style shared with score_runtime_split.py (PAMAP2): colour = method family as
# in the strip plots and Figure 6 (CPD method blue, SD method orange); the
# Non-Guided half is a light tint of it, the Guided half the full colour. Drawn
# at print size.
SPLIT_RC = {
    **PANEL_RC,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "axes.labelsize": 6.5,
    "legend.fontsize": 5.8,
}


def family_color(algo: str) -> str:
    return FAMILY_COLOR["SD" if ALGO_TASK.get(algo) == "SD" else "CPD"]


def _tint(color: str, amount: float = 0.62) -> tuple:
    """``color`` mixed with white (opaque, so the grid does not show through)."""
    r, g, b, _ = to_rgba(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount, 1.0)


def _best_grid_runtimes(
    df_grid: pd.DataFrame, metric: str, algos: set
) -> tuple[dict[str, np.ndarray], dict[str, float], int]:
    df = df_grid[df_grid["algorithm"].isin(algos)].copy()
    df = df[df["dataset"] != "pamap2"]
    df = df.dropna(subset=[metric, RUNTIME_COL, "trial_index", "parent_run_id"])
    if df.empty:
        return {}, {}, 0

    total = int(df.groupby("dataset")["trial_index"].nunique().sum())

    ranked = select_best_grid_per_dataset(df, metric, higher_is_better=True)
    if ranked.empty:
        return {}, {}, total

    best = ranked[ranked["config_rank"] == 1]
    runtimes: dict[str, np.ndarray] = {}
    coverage: dict[str, float] = {}
    for algo, sub in best.groupby("algorithm"):
        rt = sub[RUNTIME_COL].to_numpy(dtype=float)
        rt = rt[np.isfinite(rt) & (rt > 0)]
        if rt.size:
            runtimes[str(algo)] = rt
            n_covered = int(sub.groupby("dataset")["trial_index"].nunique().sum())
            coverage[str(algo)] = n_covered / total if total else 0.0
    return runtimes, coverage, total


def draw_split_box(ax, x, data, color, side, width=0.72):
    """Half-width box at ``x`` (side 'left' = Non-Guided, 'right' = Guided) or a
    full-width one ('full', method run in one mode only, in its mode's shade).

    ``side`` 'full-ng' draws a full-width box in the Non-Guided tint."""
    half = width / 2.0
    if side == "left":
        pos, w, face = x - half / 2.0, half * 0.92, _tint(color)
    elif side == "right":
        pos, w, face = x + half / 2.0, half * 0.92, color
    elif side == "full-ng":
        pos, w, face = x, width * 0.92, _tint(color)
    else:
        pos, w, face = x, width * 0.92, color
    dark = face == color
    bp = ax.boxplot(
        [data],
        positions=[pos],
        widths=[w],
        patch_artist=True,
        showfliers=False,
        whis=1.5,
        medianprops=dict(color="white" if dark else INK, linewidth=0.9),
        whiskerprops=dict(color=color, linewidth=0.6),
        capprops=dict(color=color, linewidth=0.6),
        boxprops=dict(linewidth=0.6),
    )
    for box in bp["boxes"]:
        box.set_facecolor(face)
        box.set_edgecolor(color)


def draw_pair(ax, x, algo, ng: dict, g: dict):
    color = family_color(algo)
    if algo in ng and algo in g:
        draw_split_box(ax, x, ng[algo], color, side="left")
        draw_split_box(ax, x, g[algo], color, side="right")
    elif algo in g:
        draw_split_box(ax, x, g[algo], color, side="full")
    elif algo in ng:
        draw_split_box(ax, x, ng[algo], color, side="full-ng")


def style_axis(ax, n: int, ylabel: str, log: bool):
    if log:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", which="major", color=GRID, linewidth=0.5)
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)
    ax.set_xlim(-0.6, n - 0.4)
    ax.tick_params(axis="x", length=0, pad=2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def legend_handles(families: set[str]) -> list:
    """NG/G shading in neutral grey, then the family colours if both are drawn."""
    grey = "#7d7c78"
    handles = [
        Patch(facecolor=_tint(grey), edgecolor=grey, linewidth=0.6, label="Non-Guided (left half)"),
        Patch(facecolor=grey, edgecolor=grey, linewidth=0.6, label="Guided (right half)"),
    ]
    for fam in ("CPD", "SD"):
        if fam in families and len(families) > 1:
            handles.append(
                Patch(facecolor=FAMILY_COLOR[fam], edgecolor="none", label=f"{fam} method")
            )
    return handles


def _save_one(
    rt_ng: dict,
    rt_g: dict,
    name: str,
    size: tuple[float, float],
    margins: tuple[float, float, float, float],
    legend_ncol: int,
):
    algos = sorted(
        set(rt_ng) | set(rt_g), key=lambda a: float(np.median(rt_g[a] if a in rt_g else rt_ng[a]))
    )
    with mpl.rc_context(SPLIT_RC):
        fig = plt.figure(figsize=size)
        left, bottom, right, top = margins
        ax = fig.add_axes(
            [
                left / size[0],
                bottom / size[1],
                1 - (left + right) / size[0],
                1 - (bottom + top) / size[1],
            ]
        )
        for x, algo in enumerate(algos):
            draw_pair(ax, x, algo, rt_ng, rt_g)
        style_axis(ax, len(algos), "Runtime per series (s)", log=True)
        ax.set_xticks(range(len(algos)))
        ax.set_xticklabels(
            [display_name(a) for a in algos], rotation=45, ha="right", rotation_mode="anchor"
        )
        fams = {"SD" if ALGO_TASK.get(a) == "SD" else "CPD" for a in algos}
        ax.legend(
            handles=legend_handles(fams),
            loc="lower left",
            bbox_to_anchor=(0.0, 1.01),
            ncol=legend_ncol,
            frameon=False,
            handlelength=1.2,
            handleheight=0.8,
            columnspacing=1.0,
            borderaxespad=0,
            labelcolor=INK,
        )
        savefig(fig, name, tight=False)


def make_figure(d=None):
    if d is None:
        d = load_data()

    cpd_algos = {a for a in d.cpd_algos if a != "random" and a not in EXCLUDED_ALGOS}
    sd_algos = {a for a in d.sd_algos if a != "random" and a not in EXCLUDED_ALGOS}

    rt_cpd_ng, _, _ = _best_grid_runtimes(d.df_grid_ng, METRIC_CPD, cpd_algos)
    rt_cpd_g, _, _ = _best_grid_runtimes(d.df_grid_g, METRIC_CPD, cpd_algos)
    rt_sd_ng, _, _ = _best_grid_runtimes(d.df_grid_ng, METRIC_SD, sd_algos)
    rt_sd_g, _, _ = _best_grid_runtimes(d.df_grid_g, METRIC_SD, sd_algos)

    # CPD: full text width (5.5 in); SD: panel (c) of a three-panel row.
    _save_one(
        rt_cpd_ng,
        rt_cpd_g,
        "fig_runtime_split_cpd",
        size=(5.5, 2.15),
        margins=(0.42, 0.52, 0.05, 0.22),
        legend_ncol=4,
    )
    _save_one(
        rt_sd_ng,
        rt_sd_g,
        "fig_runtime_split_sd",
        size=(2.0, 1.85),
        margins=(0.40, 0.50, 0.04, 0.30),
        legend_ncol=1,
    )
