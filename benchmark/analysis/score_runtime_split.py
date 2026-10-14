"""PAMAP2 case study: tuned score and runtime of every scalable method on PAMAP2.

``fig_pamap2`` at text width: CPD (BiC) on the left, SD (SMS)
on the right; top row the tuned score of each series, bottom row the runtime
of the tuned configuration, both as Non-Guided / Guided split boxes in the
style of ``runtime_split.py`` (family colours, print size). The tuned
configuration is selected on PAMAP2 itself, like the other datasets.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.data import load_data
from analysis.helpers import ALGO_TASK, display_name, savefig, select_best_grid_per_dataset
from analysis.runtime_split import (
    INK,
    SPLIT_RC,
    draw_pair,
    legend_handles,
    style_axis,
)

METRIC_CPD = "bidirectional_covering_score"
METRIC_SD = "state_matching_score"
RUNTIME_COL = "execution_time_seconds"

# The 17 methods with at least one configuration that ran every PAMAP2 series
# (DynP: Guided only).
SCALES_ALGOS = {
    "binseg",
    "bottomup",
    "changefinder",
    "prophet",
    "window",
    "tire",
    "ggs",
    "fluss",
    "amoc",
    "pelt",
    "ticc",
    "time2state",
    "e2usd",
    "autoplait",
    "igts",
    "hdp-hsmm",
    "dynp",
}


def _best_grid_stats(
    df_grid: pd.DataFrame, metric: str, algos: set
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Per-series score and runtime of the tuned configuration on PAMAP2."""
    df = df_grid[df_grid["algorithm"].isin(algos)].copy()
    df = df[df["dataset"] == "pamap2"]
    df = df.dropna(subset=[metric, RUNTIME_COL, "trial_index", "parent_run_id"])
    if df.empty:
        return {}, {}

    ranked = select_best_grid_per_dataset(df, metric, higher_is_better=True)
    if ranked.empty:
        return {}, {}

    best = ranked[ranked["config_rank"] == 1]
    scores: dict[str, np.ndarray] = {}
    runtimes: dict[str, np.ndarray] = {}
    for algo, sub in best.groupby("algorithm"):
        sc = sub[metric].to_numpy(dtype=float)
        rt = sub[RUNTIME_COL].to_numpy(dtype=float)
        valid = np.isfinite(sc) & np.isfinite(rt) & (rt > 0)
        if valid.any():
            scores[str(algo)] = sc[valid]
            runtimes[str(algo)] = rt[valid]
    return scores, runtimes


def _order(scores_ng: dict, scores_g: dict) -> list[str]:
    """By median tuned score, Non-Guided when the method runs in that mode."""

    def key(a):
        return float(np.median(scores_ng[a] if a in scores_ng else scores_g[a]))

    return sorted(set(scores_ng) | set(scores_g), key=key)


def _task_columns(fig, gs_col, data: dict, ylabel: str, show_ylabels: bool):
    s_ng, s_g, r_ng, r_g = data["s_ng"], data["s_g"], data["r_ng"], data["r_g"]
    algos = _order(s_ng, s_g)
    ax_s = fig.add_subplot(gs_col[0])
    ax_r = fig.add_subplot(gs_col[1], sharex=ax_s)
    for x, algo in enumerate(algos):
        draw_pair(ax_s, x, algo, s_ng, s_g)
        draw_pair(ax_r, x, algo, r_ng, r_g)
    style_axis(ax_s, len(algos), ylabel if show_ylabels else ylabel, log=False)
    style_axis(ax_r, len(algos), "Runtime per series (s)" if show_ylabels else "", log=True)
    ax_s.set_ylim(0, 1)
    ax_s.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
    ax_s.tick_params(axis="x", labelbottom=False)
    ax_r.set_xticks(range(len(algos)))
    ax_r.set_xticklabels(
        [display_name(a) for a in algos], rotation=45, ha="right", rotation_mode="anchor"
    )
    return ax_s, ax_r, algos


def make_figure(d=None):
    if d is None:
        d = load_data()

    cpd_algos = {a for a in d.cpd_algos if a in SCALES_ALGOS}
    sd_algos = {a for a in d.sd_algos if a in SCALES_ALGOS}
    tasks = {}
    for task, metric, algos in (("CPD", METRIC_CPD, cpd_algos), ("SD", METRIC_SD, sd_algos)):
        s_ng, r_ng = _best_grid_stats(d.df_grid_ng, metric, algos)
        s_g, r_g = _best_grid_stats(d.df_grid_g, metric, algos)
        tasks[task] = dict(s_ng=s_ng, s_g=s_g, r_ng=r_ng, r_g=r_g)
        for name, sc in (("NG", s_ng), ("G", s_g)):
            means = {a: round(float(np.mean(v)), 3) for a, v in sc.items()}
            print(
                f"{task} {name} mean tuned score:",
                dict(sorted(means.items(), key=lambda kv: kv[1])),
            )

    n_cpd = len(set(tasks["CPD"]["s_ng"]) | set(tasks["CPD"]["s_g"]))
    n_sd = len(set(tasks["SD"]["s_ng"]) | set(tasks["SD"]["s_g"]))
    with mpl.rc_context(SPLIT_RC):
        fig = plt.figure(figsize=(5.5, 3.1))
        outer = fig.add_gridspec(
            2,
            2,
            width_ratios=[n_cpd, n_sd + 1.2],
            height_ratios=[1, 1],
            left=0.075,
            right=0.995,
            bottom=0.17,
            top=0.88,
            wspace=0.16,
            hspace=0.12,
        )
        ax_cs, ax_cr, _ = _task_columns(
            fig, (outer[0, 0], outer[1, 0]), tasks["CPD"], "Tuned BiC", True
        )
        ax_ss, ax_sr, _ = _task_columns(
            fig, (outer[0, 1], outer[1, 1]), tasks["SD"], "Tuned SMS", True
        )
        ax_sr.set_ylabel("")
        ax_cs.set_title("(a) CPD", loc="left", fontsize=6.6, color=INK, pad=3)
        ax_ss.set_title("(b) SD", loc="left", fontsize=6.6, color=INK, pad=3)
        fams = {
            "SD" if ALGO_TASK.get(a) == "SD" else "CPD"
            for t in tasks.values()
            for a in set(t["s_ng"]) | set(t["s_g"])
        }
        fig.legend(
            handles=legend_handles(fams),
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=4,
            frameon=False,
            handlelength=1.2,
            handleheight=0.8,
            columnspacing=1.2,
            labelcolor=INK,
        )
        savefig(fig, "fig_pamap2", tight=False)
