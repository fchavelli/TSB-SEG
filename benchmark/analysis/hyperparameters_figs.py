"""Influence of the hyperparameters on the target score (appendix).

``hyperparams/fig_hyperparams_{cpd,sd}_{ng,g}``: the score (BiC for CPD, SMS
for SD) of every grid configuration, one panel per method, Non-Guided and
Guided.

Design notes:
* Methods without a grid to show are excluded: AMOC and AutoPlait (one
  configuration), CLaP from the CPD figures (scored through ClaSP), and tGLAD
  and Hidalgo (extended scope only).
* The most influential hyper-parameter (largest std of marginal score
  means) is used as the primary x-axis. Up to ``MAX_HPS`` parameters are
  shown per panel.
* Param names and categorical values are abbreviated through ``SHORT_NAME``
  / ``SHORT_VALUE`` so legends fit in-axes.
* Colours come from the dataviz reference palette (colour-blind safe).
"""

from __future__ import annotations

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from analysis.data import load_data
from analysis.helpers import METRIC_CPD, METRIC_SD, OUTPUT_DIR, analyzer, display_name

# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────

EXCLUDED_ALGOS = {"amoc", "autoplait", "tglad", "hidalgo"}
EXCLUDED_PER_TASK = {"CPD": {"clap"}, "SD": set()}

COMPLETION_THRESHOLD = 0.65

MAX_HPS = 3

OUT_DIR = OUTPUT_DIR / "hyperparams"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SHORT_NAME = {
    "window_size": "win",
    "k_neighbours": "k_n",
    "excl_radius": "excl",
    "exclusion_factor": "excl_f",
    "n_estimators": "n_est",
    "max_epochs": "epochs",
    "multivariate_strategy": "mv",
    "smooth_window": "sm_win",
    "discount": "disc",
    "penalty": "pen",
    "model": "cost",  # the segment cost of the penalised detectors
    "classifier": "clf",
    "distance": "dist",
    "domain": "dom",
    "lambda_parameter": "lambda",
    "n_max_states": "n_states",
    "dur_alpha": "dur_a",
    "chain_len": "chain",
    "min_size": "min",
    "kernel": "kern",
    "threshold": "thr",
    "threshold_factor": "thr_f",
    "delta_threshold": "delta_thr",
    "changepoint_prior_scale": "prior",
    "prominence_threshold": "prom_thr",
}

SHORT_VALUE = {
    "znormed_euclidean_distance": "znorm",
    "cinvariant_euclidean_distance": "cinv",
    "euclidean_distance": "eucl",
    "ensembling": "ens",
    "mrhydra": "mrhy",
    "rocket": "rkt",
}


def _short_name(key: str) -> str:
    return SHORT_NAME.get(key, key)


def _short_val(val) -> str:
    s = str(val)
    if s in SHORT_VALUE:
        return SHORT_VALUE[s]
    try:
        v = float(s)
    except ValueError:
        return s
    if v.is_integer():
        return str(int(v))
    if 0 < v < 1 and (1 / v).is_integer() and int(1 / v) & (int(1 / v) - 1) == 0:
        return f"1/{int(1 / v)}"  # powers of two (Window penalties)
    return f"{v:g}"


# ─────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────


def get_grid_params(run_ids):
    df_p = analyzer.manager.fetch_run_params(list(run_ids))
    if df_p.empty:
        return pd.DataFrame()
    return df_p.pivot(index="run_id", columns="key", values="value").reset_index()


def _per_algo_max_runs(df_grid: pd.DataFrame) -> dict[str, int]:
    r"""Per-algorithm theoretical run budget.

    For each (algorithm, dataset) pair we take the maximum number of distinct
    series ever completed by a single hyper-parameter combination, then sum
    across the datasets the algorithm actually exercises. This rewards
    algorithms that only run on a subset of datasets (e.g.\ multivariate-only
    methods such as ``ticc``) by not penalising them for the datasets they
    cannot process.
    """
    df_valid = df_grid[df_grid["dataset"] != "pamap2"]
    counts = (
        df_valid.groupby(["algorithm", "parent_run_id", "dataset"])["run_id"]
        .nunique()
        .reset_index()
    )
    per_algo_dataset = counts.groupby(["algorithm", "dataset"])["run_id"].max().reset_index()
    return per_algo_dataset.groupby("algorithm")["run_id"].sum().to_dict()


# ─────────────────────────────────────────────────────────────────────
# Main per-task processing
# ─────────────────────────────────────────────────────────────────────


def process_task(task_name, target_metric, algos, df_grid, filename, also_excluded=()):
    print(f"\nProcessing {task_name} for {filename}...")
    excluded = EXCLUDED_ALGOS | EXCLUDED_PER_TASK.get(task_name, set()) | set(also_excluded)
    fig_data: list[dict] = []
    algo_max_runs = _per_algo_max_runs(df_grid)

    for algo in tqdm(algos):
        if algo in excluded:
            continue
        max_runs = algo_max_runs.get(algo, 0)
        if max_runs <= 0:
            continue

        df_algo = df_grid[(df_grid["algorithm"] == algo) & (df_grid["dataset"] != "pamap2")].copy()
        if df_algo.empty:
            continue

        parent_ids = set(df_algo["parent_run_id"].dropna().unique())
        if not parent_ids:
            continue

        df_params = get_grid_params(parent_ids)
        if df_params.empty:
            continue

        # Detect varying grid_* params, drop the bookkeeping ones.
        hyperparams: list[str] = []
        for c in list(df_params.columns):
            if not c.startswith("grid_") or c == "grid_execution_role":
                continue
            clean = c[5:]
            df_params[clean] = df_params[c].astype(str)
            if df_params[clean].nunique() > 1:
                hyperparams.append(clean)

        if not hyperparams:
            continue

        merged_all = df_algo.merge(df_params, left_on="parent_run_id", right_on="run_id")

        # Aggregate per (hyperparam combination):
        #   sum of scores over the series that ran;
        #   count of distinct (parent, dataset, series) runs.
        # Padded score = sum / global_max_runs (treats missing as 0).
        grouped = merged_all.groupby(hyperparams)
        sums = grouped[target_metric].sum().rename("sum_score")
        counts = grouped["run_id_x"].nunique().rename("n_runs")
        scores = pd.concat([sums, counts], axis=1).reset_index()
        scores["completion"] = scores["n_runs"] / max_runs
        scores[target_metric] = scores["sum_score"] / max_runs

        scores = scores[scores["completion"] >= COMPLETION_THRESHOLD].copy()
        if scores.empty:
            continue

        # Drop params that collapsed to a single value after the
        # completion filter — they would clutter the legend with a
        # constant label that carries no information.
        hyperparams = [hp for hp in hyperparams if scores[hp].nunique() > 1]
        if not hyperparams:
            continue

        # Influence ranking: std of marginal means per parameter.
        # The most influential param is used as primary x-axis.
        influence = []
        for hp in hyperparams:
            infl = scores.groupby(hp)[target_metric].mean().std()
            influence.append((0.0 if pd.isna(infl) else infl, hp))
        influence.sort(reverse=True)
        sorted_hps = [hp for _, hp in influence]

        # ChangeFinder: split by multivariate_strategy as separate panels.
        if algo == "changefinder" and "multivariate_strategy" in sorted_hps:
            other_hps = [h for h in sorted_hps if h != "multivariate_strategy"][:MAX_HPS]
            for strat in ["l2", "ensembling"]:
                sub = scores[scores["multivariate_strategy"] == strat].copy()
                if sub.empty:
                    continue
                fig_data.append(
                    {
                        "algo": f"changefinder ({strat})",
                        "data": sub,
                        "hps": other_hps,
                        "target": target_metric,
                    }
                )
        else:
            fig_data.append(
                {
                    "algo": algo,
                    "data": scores,
                    "hps": sorted_hps[:MAX_HPS],
                    "target": target_metric,
                }
            )

    if not fig_data:
        print("No valid algorithms found for plotting.")
        return

    _render_figure(fig_data, filename)


# ─────────────────────────────────────────────────────────────────────
# Plot rendering
# ─────────────────────────────────────────────────────────────────────


# Style: every panel has the same size in the CPD and SD figures
# (four columns at the text width, saved at print size, so that the largest
# figure, CPD Guided, still fits on one page with its caption), colours from the
# dataviz reference palette with a white ring instead of a black edge, no box
# around the axes and no vertical separators.
_HP_PALETTE = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
]
_HP_SHAPES = ["o", "s", "^", "D", "v", "P", "X", "*"]
_INK, _INK_2, _GRID = "#0b0b0b", "#52514e", "#e1e0d9"
_COLS = 4
_CELL_W, _CELL_H = 5.5 / 4, 1.62  # inches, at print size (NeurIPS text width 5.5 in)


def _sorted_vals(series: pd.Series) -> list:
    try:
        return sorted(series.unique(), key=lambda v: float(v))
    except (TypeError, ValueError):
        return sorted(series.unique())


def _render_figure(fig_data: list[dict], filename: str) -> None:
    n_plots = len(fig_data)
    rows = (n_plots + _COLS - 1) // _COLS
    # Fixed geometry (inches) instead of tight_layout, so that every panel has
    # exactly the same size in all four figures: per cell, a title band on top,
    # then the axes, then the tick labels, the x label and the legend.
    top_band, title_h, foot_h, left_w, right_w = 0.15, 0.22, 0.62, 0.32, 0.06
    fig_w, fig_h = _COLS * _CELL_W, rows * _CELL_H + top_band
    fig = plt.figure(figsize=(fig_w, fig_h))
    axes = []
    for i in range(rows * _COLS):
        r, c = divmod(i, _COLS)
        x0 = c * _CELL_W + left_w
        y0 = fig_h - top_band - (r + 1) * _CELL_H + foot_h
        axes.append(
            fig.add_axes(
                [
                    x0 / fig_w,
                    y0 / fig_h,
                    (_CELL_W - left_w - right_w) / fig_w,
                    (_CELL_H - foot_h - title_h) / fig_h,
                ]
            )
        )

    for i, item in enumerate(fig_data):
        ax = axes[i]
        merged, hps, target, algo_name = item["data"], item["hps"], item["target"], item["algo"]
        if not hps:
            continue

        main_hp = hps[0]
        x_vals = _sorted_vals(merged[main_hp])
        x_map = {v: xi for xi, v in enumerate(x_vals)}
        second_hp = hps[1] if len(hps) >= 2 else None
        third_hp = hps[2] if len(hps) >= 3 else None
        s_vals = _sorted_vals(merged[second_hp]) if second_hp else []
        t_vals = _sorted_vals(merged[third_hp]) if third_hp else []

        for _, row in merged.iterrows():
            x = x_map[row[main_hp]]
            color, marker = _HP_PALETTE[0], "o"
            if second_hp:
                s_idx = s_vals.index(row[second_hp])
                color = _HP_PALETTE[s_idx % len(_HP_PALETTE)]
                if len(s_vals) > 1:
                    x += np.linspace(-0.22, 0.22, len(s_vals))[s_idx]
            if third_hp:
                marker = _HP_SHAPES[t_vals.index(row[third_hp]) % len(_HP_SHAPES)]
            ax.scatter(
                x,
                row[target],
                marker=marker,
                color=color,
                s=14,
                edgecolors="white",
                linewidths=0.5,
                zorder=3,
            )

        handles: list = []
        if second_hp:
            for s_idx, s_val in enumerate(s_vals):
                handles.append(
                    mlines.Line2D(
                        [],
                        [],
                        color=_HP_PALETTE[s_idx % len(_HP_PALETTE)],
                        marker="o",
                        linestyle="None",
                        markersize=3.5,
                        markeredgecolor="white",
                        markeredgewidth=0.4,
                        label=f"{_short_name(second_hp)}={_short_val(s_val)}",
                    )
                )
        if third_hp:
            for t_idx, t_val in enumerate(t_vals):
                handles.append(
                    mlines.Line2D(
                        [],
                        [],
                        color=_INK_2,
                        marker=_HP_SHAPES[t_idx % len(_HP_SHAPES)],
                        linestyle="None",
                        markersize=3.5,
                        markeredgecolor="white",
                        markeredgewidth=0.4,
                        label=f"{_short_name(third_hp)}={_short_val(t_val)}",
                    )
                )
        xlabels = [_short_val(v) for v in x_vals]
        long = sum(len(t) for t in xlabels) > 14
        if handles:
            ax.legend(
                handles=handles,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.55 if long else -0.36),
                fontsize=5,
                frameon=False,
                ncol=min(len(handles), 2),
                handletextpad=0.1,
                columnspacing=0.5,
                borderpad=0.1,
                labelspacing=0.2,
                labelcolor=_INK,
            )

        ax.set_xticks(range(len(x_vals)))
        ax.set_xticklabels(
            xlabels,
            fontsize=5.5,
            rotation=30 if long else 0,
            ha="right" if long else "center",
            rotation_mode="anchor" if long else "default",
        )
        ax.set_xlim(-0.6, len(x_vals) - 0.4)
        ax.tick_params(
            axis="both", labelsize=5.5, colors=_INK_2, labelcolor=_INK, width=0.5, length=2, pad=1.5
        )
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(_INK_2)
            ax.spines[side].set_linewidth(0.5)
        ax.grid(True, axis="y", color=_GRID, linewidth=0.5)
        ax.set_axisbelow(True)

        title_extra = "" if "(" not in algo_name else " " + algo_name[algo_name.find("(") :]
        ax.set_title(
            f"{display_name(algo_name.split(' ')[0])}{title_extra}", fontsize=7, color=_INK, pad=3
        )
        ax.set_xlabel(_short_name(main_hp), fontsize=6, color=_INK, labelpad=1 if not long else 3)
        ax.set_ylabel("")

    for j in range(len(fig_data), len(axes)):
        fig.delaxes(axes[j])

    metric_label = "BiC" if any(item["target"] == METRIC_CPD for item in fig_data) else "SMS"
    fig.text(
        0.01,
        1 - 0.04 / fig_h,
        f"Score: {metric_label}",
        ha="left",
        va="top",
        fontsize=7,
        color=_INK,
    )
    out_pdf = OUT_DIR / f"{filename}.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{filename}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved {out_pdf}")


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────


def main(d=None) -> None:
    if d is None:
        d = load_data()

    # Panels in alphabetical order of the displayed name (d.*_algos are sets,
    # whose order changes from one run to the next).
    cpd_algos = sorted(
        (a for a in d.cpd_algos if a != "random"), key=lambda a: display_name(a).lower()
    )
    sd_algos = sorted(
        (a for a in d.sd_algos if a != "random"), key=lambda a: display_name(a).lower()
    )

    process_task("CPD", METRIC_CPD, cpd_algos, d.df_grid_ng, "fig_hyperparams_cpd_ng")
    process_task("SD", METRIC_SD, sd_algos, d.df_grid_ng, "fig_hyperparams_sd_ng")
    # PELT cannot take n_CP: its Guided grid is a copy of the Non-Guided one
    # (data.GUIDED_FROM_NON_GUIDED), so it is shown once, in the Non-Guided figure.
    process_task(
        "CPD", METRIC_CPD, cpd_algos, d.df_grid_g, "fig_hyperparams_cpd_g", also_excluded={"pelt"}
    )
    process_task("SD", METRIC_SD, sd_algos, d.df_grid_g, "fig_hyperparams_sd_g")
