"""Performance figures of RQ1: strip, critical-difference and spider panels.

For each task (CPD, SD) and mode (Non-Guided ``main``, Guided ``guided``):
``fig_<task>_<mode>_strip`` (Default, tuned, median and worst-case score per
method), ``fig_<task>_<mode>_cd`` (critical-difference diagram of the tuned
scores) and ``fig_<task>_<mode>_spider{,_dataset}`` (tuned score of the top
methods per domain and per dataset).
"""

from __future__ import annotations

from itertools import combinations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from scipy.stats import wilcoxon

from analysis.data import load_data
from analysis.helpers import (
    ALGO_TASK,
    DOMAIN_ORDER,
    TEXT_W,
    display_name,
    savefig,
)

# Font sizes of the stand-alone panels, legible after LaTeX scaling; no global
# tight bounding box, so that a saved file has the requested size.
_PANEL_RC = {
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "savefig.bbox": "standard",
    "savefig.pad_inches": 0.05,
}


# ── Strip glyph ──────────────────────────────────────────────────────
# Each method is one "drop": a thick body from the median to the tuned
# configuration (the head, a dot) and a thin, lighter stem from the median
# down to the worst one, so that the eye goes to the tuned score and the
# median rather than to the worst; the Default configuration is an open
# diamond. Colour carries one thing only, the method family (CPD method or SD
# method, validated for colour-vision deficiency); the method is named on the
# axis.
_FAMILY_COLOR = {"CPD": "#2a78d6", "SD": "#eb6834"}
_FAMILY_LABEL = {"CPD": "CPD method", "SD": "SD method"}
_INK = "#0b0b0b"
_INK_2 = "#52514e"
_GRID = "#e1e0d9"
_AMOC_REF = "#a8a69e"  # dotted reference line at the tuned score of AMOC


def _tint(color: str, amount: float = 0.55) -> tuple:
    """``color`` mixed with white (opaque, unlike an alpha that shows the grid)."""
    r, g, b, _ = to_rgba(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount, 1.0)


def _family(algo: str) -> str:
    return "SD" if ALGO_TASK.get(algo, "") == "SD" else "CPD"


def _draw_drop(ax, x, stats: dict, color: str, scale: float = 1.0):
    """One method: stem worst→median, body median→best, head at best, ◇ default."""
    best, median = stats.get("best"), stats.get("median")
    worst, default = stats.get("worst"), stats.get("default")
    if worst is not None and median is not None:
        ax.plot(
            [x, x],
            [worst, median],
            color=_tint(color),
            lw=1.1 * scale,
            solid_capstyle="butt",
            zorder=2,
        )
        ax.scatter([x], [worst], s=7 * scale, color=_tint(color), linewidths=0, zorder=2)
    if median is not None and best is not None:
        ax.plot(
            [x, x],
            [median, best],
            color=color,
            lw=4.2 * scale,
            solid_capstyle="butt",
            zorder=3,
        )
    if best is not None:
        ax.scatter(
            [x],
            [best],
            s=34 * scale,
            color=color,
            edgecolors="white",
            linewidths=0.9,
            zorder=4,
        )
    if default is not None:
        ax.scatter(
            [x],
            [default],
            marker="D",
            s=15 * scale,
            facecolor="white",
            edgecolors=_INK,
            linewidths=0.8,
            zorder=5,
        )


def _style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_INK_2)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=_INK_2, width=0.6, length=2.5, labelcolor=_INK)
    ax.grid(True, axis="y", color=_GRID, linewidth=0.6, linestyle="-")
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)


def _drop_legend_handles(families=()) -> list:
    grey = "#7d7c78"
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            color=grey,
            markersize=5,
            linestyle="None",
            markeredgecolor="white",
            markeredgewidth=0.8,
            label="Tuned",
        ),
        Line2D([], [], color=grey, linewidth=4.2, label="Median to tuned"),
        Line2D([], [], color=_tint(grey), linewidth=1.1, label="Worst to median"),
        Line2D(
            [],
            [],
            marker="D",
            color="white",
            markersize=3.8,
            linestyle="None",
            markeredgecolor=_INK,
            markeredgewidth=0.8,
            label="Default",
        ),
    ]
    for fam in families:
        handles.append(
            mpl.patches.Patch(
                facecolor=_FAMILY_COLOR[fam], edgecolor="none", label=_FAMILY_LABEL[fam]
            )
        )
    return handles


def plot_strip(
    ax: plt.Axes,
    df: pd.DataFrame,
    score_col: str = "score_best",
    ylabel: str = "Score",
    show_hatch: bool = True,
    show_legend: bool = True,
    legend_ncol: int = 2,
    ylim: tuple | None = None,
    legend_loc: str = "upper left",
):
    """Strip plot: one drop per method (means across series), worst on the left.

    ``show_hatch``: True on the CPD panels, which mix CPD and SD methods
    (coloured by family, with the family in the legend); False on the SD
    panels.
    """
    df = df[df["algorithm"] != "random"]
    algo_order = df.groupby("algorithm")[score_col].mean().sort_values().index.tolist()
    n = len(algo_order)
    _style_axes(ax)

    families = []
    for x, algo in enumerate(algo_order):
        sub = df[df["algorithm"] == algo]
        stats = {}
        for key, col in (
            ("best", "score_best"),
            ("median", "score_median"),
            ("worst", "score_worst"),
            ("default", "score_default"),
        ):
            if col in sub.columns and sub[col].notna().any():
                stats[key] = float(sub[col].mean())
        fam = _family(algo)
        if fam not in families:
            families.append(fam)
        _draw_drop(ax, x, stats, _FAMILY_COLOR[fam])

    if "amoc" in algo_order:
        # Reference: the single change point of AMOC (its tuned score), so that
        # the configurations falling below it can be read off.
        x0 = algo_order.index("amoc")
        y0 = float(df.loc[df["algorithm"] == "amoc", score_col].mean())
        ax.hlines(
            y0, x0, n - 0.4, colors=_AMOC_REF, linestyles=(0, (1.5, 1.5)), linewidth=0.7, zorder=0.5
        )

    ax.set_xticks(range(n))
    ax.set_xticklabels(
        [display_name(a) for a in algo_order], rotation=45, ha="right", rotation_mode="anchor"
    )
    if n > 17:
        ax.tick_params(axis="x", labelsize=7.5)
    ax.set_ylabel(ylabel, color=_INK)
    ax.set_xlim(-0.6, n - 0.4)
    if ylim is not None:
        ax.set_ylim(*ylim)

    if show_legend:
        fams = (
            [f for f in ("CPD", "SD") if f in families] if show_hatch and len(families) > 1 else []
        )
        handles = _drop_legend_handles(fams)
        if legend_loc == "above":
            # Above the axes, so that it never hides a method.
            ax.legend(
                handles=handles,
                loc="lower left",
                bbox_to_anchor=(0.0, 1.02, 1.0, 0.1),
                mode="expand",
                ncol=3,
                fontsize=6,
                frameon=False,
                handlelength=1.6,
                borderaxespad=0.0,
                labelcolor=_INK,
            )
        else:
            leg = ax.legend(
                handles=handles,
                loc=legend_loc,
                ncol=legend_ncol,
                fontsize=6,
                frameon=True,
                framealpha=1.0,
                edgecolor=_GRID,
                handlelength=1.6,
                borderpad=0.5,
                labelcolor=_INK,
            )
            leg.get_frame().set_linewidth(0.6)

    return algo_order


# ── Spider ───────────────────────────────────────────────────────────
# Colours and markers fixed per method, so that a method keeps its look across
# the Non-Guided and Guided panels of a task. With the top-6 methods of each
# mode, the CPD panels draw 8 methods in all. The assignment was searched
# (OKLab Delta E x100) among the reference palette plus a brown and a plum, so
# that every pair drawn in the same panel clears the normal-vision floor (15)
# and the colour-vision-deficiency target (8): worst pair 9.2 (CPD) and 10.2
# (SD). The marker shapes are a second encoding; the legend names every line.
# Two methods share a style only if they are never drawn in the same task.
SPIDER_TOP_K = 6
_SPIDER_STYLE = {
    # CPD panels
    "clasp": ("#2a78d6", "o"),
    "pelt": ("#eb6834", "s"),
    "binseg": ("#4a3aa7", "^"),
    "bottomup": ("#9a5b22", "v"),
    "window": ("#a6407e", "D"),
    "icid": ("#1baf7a", "P"),
    "dynp": ("#eda100", "X"),
    "ggs": ("#e87ba4", "*"),
    "changefinder": ("#1baf7a", "P"),
    # SD panels
    "clap": ("#2a78d6", "o"),
    "time2state": ("#eda100", "s"),
    "e2usd": ("#e87ba4", "^"),
    "hdp-hsmm": ("#008300", "v"),
    "ticc": ("#4a3aa7", "D"),
    "autoplait": ("#a6407e", "P"),
}
_DATASET_DISPLAY = {
    "actrectut": "ActRecTut",
    "has": "HAS",
    "mocap": "MoCap",
    "pump": "PUMP",
    "skab": "SKAB",
    "tssb": "TSSB",
    "usc-had": "USC-HAD",
    "utsa": "UTSA",
}


def _top_algos(df: pd.DataFrame, score_col: str = "score_best", top_k: int = SPIDER_TOP_K) -> list:
    means = df.groupby("algorithm")[score_col].mean().sort_values(ascending=False)
    return [a for a in means.index if a != "random"][:top_k]


def _check_spider_styles(*dfs: pd.DataFrame) -> None:
    """Every top-k method of ``dfs`` (one task) has its own spider style."""
    drawn = sorted({a for df in dfs for a in _top_algos(df)})
    missing = sorted(set(drawn) - set(_SPIDER_STYLE))
    if missing:
        raise ValueError(
            f"no spider colour for {missing}: extend _SPIDER_STYLE "
            "(and re-check the pairs drawn together)"
        )
    colours = [_SPIDER_STYLE[a][0] for a in drawn]
    if len(set(colours)) < len(colours):
        raise ValueError(f"spider methods {drawn} share a colour: re-assign _SPIDER_STYLE")


def plot_spider(
    ax: plt.Axes,
    df: pd.DataFrame,
    score_col: str = "score_best",
    top_k: int = SPIDER_TOP_K,
    group_by: str = "domain",
):
    """Radar plot of the tuned score of the top-k methods, per domain or dataset."""
    top_algos = _top_algos(df, score_col, top_k)

    group_means = (
        df[df["algorithm"].isin(top_algos)]
        .groupby(["algorithm", group_by])[score_col]
        .mean()
        .unstack(fill_value=0)
    )

    if group_by == "domain":
        active_groups = [d for d in DOMAIN_ORDER if d in group_means.columns]
        labels = active_groups
    else:
        active_groups = sorted([d for d in group_means.columns if d != "pamap2"])
        labels = [_DATASET_DISPLAY.get(d, d) for d in active_groups]

    group_means = group_means[active_groups]

    N = len(active_groups)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    # First category at the top, clockwise.
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    for algo in top_algos:
        values = group_means.loc[algo].tolist()
        values += values[:1]
        c, m = _SPIDER_STYLE[algo]
        ax.plot(
            angles,
            values,
            linewidth=1.0,
            color=c,
            label=display_name(algo),
            marker=m,
            markersize=3.6 if m != "*" else 4.8,
            markeredgecolor="white",
            markeredgewidth=0.4,
            solid_joinstyle="round",
            zorder=3,
        )

    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["", "0.5", "", "1.0"])
    ax.set_rlabel_position(180.0 / N)  # between the first two spokes
    ax.tick_params(axis="y", labelsize=5.5, labelcolor=_INK_2, pad=0)
    ax.grid(color=_GRID, linewidth=0.6)
    ax.spines["polar"].set_color(_GRID)
    ax.spines["polar"].set_linewidth(0.6)

    # Category names outside the circle, aligned away from it.
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])
    for ang, lab in zip(angles[:-1], labels):
        phi = np.pi / 2 - ang  # screen angle
        x, y = np.cos(phi), np.sin(phi)
        ha = "left" if x > 0.15 else ("right" if x < -0.15 else "center")
        va = "bottom" if y > 0.15 else ("top" if y < -0.15 else "center")
        ax.text(ang, 1.08, lab, ha=ha, va=va, fontsize=7, color=_INK)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=3,
        fontsize=6.5,
        frameon=False,
        handlelength=1.4,
        columnspacing=1.0,
        borderaxespad=0,
    )


# ── Critical-difference diagram ──────────────────────────────────────


def _holm_wilcoxon_matrix(scores, labels=None, lower_better=False):
    """Two-sided Wilcoxon signed-rank test on every pair of methods, Holm
    step-down over all pairs; the upper-triangular matrix of adjusted p-values.

    Replaces ``aeon``'s ``wilcoxon_test``, whose ``correction="holm"`` compares
    one-sided p-values with a fixed alpha/(k-1). With ``correction=None``, the
    bars join exactly the pairs this test does not separate at alpha = 0.1."""
    x = np.asarray(scores, dtype=float)
    k = x.shape[1]
    pairs = list(combinations(range(k), 2))
    raw = np.array(
        [
            1.0
            if np.all(x[:, i] == x[:, j])
            else float(wilcoxon(x[:, i], x[:, j], zero_method="wilcox").pvalue)
            for i, j in pairs
        ]
    )
    order = np.argsort(raw)
    adj = np.empty_like(raw)
    adj[order] = np.minimum(
        np.maximum.accumulate(raw[order] * (len(raw) - np.arange(len(raw)))), 1.0
    )
    out = np.eye(k)
    for (i, j), v in zip(pairs, adj):
        out[i, j] = v
    return out


def _strict_cliques(not_separated):
    """Bars of the diagram: maximal runs of rank-consecutive methods in which no
    pair is separated by the test (aeon's ``_build_cliques`` only checks the
    pairs that involve the first method of a run)."""
    ns = np.asarray(not_separated, dtype=bool)
    k = ns.shape[0]
    runs = []
    for i in range(k):
        j = i
        while j + 1 < k and all(ns[a, j + 1] for a in range(i, j + 1)):
            j += 1
        if j > i and not any(r[0] <= i and j <= r[1] for r in runs):
            runs.append((i, j))
    out = np.zeros((len(runs), k), dtype=bool)
    for r, (i, j) in enumerate(runs):
        out[r, i : j + 1] = True
    return out


def _critical_difference(scores, labels, **kw):
    """``aeon.plot_critical_difference`` with the two-sided Holm test and the
    strict bars above."""
    from aeon.visualisation.results import _critical_difference as cdmod

    saved = cdmod.wilcoxon_test, cdmod._build_cliques
    cdmod.wilcoxon_test, cdmod._build_cliques = _holm_wilcoxon_matrix, _strict_cliques
    try:
        return cdmod.plot_critical_difference(
            scores, labels, test="wilcoxon", correction=None, alpha=0.1, **kw
        )
    finally:
        cdmod.wilcoxon_test, cdmod._build_cliques = saved


def _save_cd_directly(
    df: pd.DataFrame, score_col: str, maximize: bool, name: str, width: float = 5.0
):
    """Save the critical-difference diagram of ``score_col`` (missing scores 0).

    The best method (lowest mean rank) is on the right, as the strip plots
    order methods worst to best from left to right."""
    df_clean = df.copy()
    df_clean["series_id"] = df_clean["dataset"] + "_" + df_clean["trial_index"].astype(str)
    pivot = df_clean.pivot_table(
        index="series_id", columns="algorithm", values=score_col, aggfunc="first"
    )
    if "random" in pivot.columns:
        pivot = pivot.drop(columns=["random"])
    if pivot.empty or pivot.shape[1] < 3:
        return
    pivot = pivot.fillna(0.0)
    pivot.columns = [display_name(c) for c in pivot.columns]

    _cd_font_rc = {
        "font.size": 28,
        "axes.titlesize": 28,
        "axes.labelsize": 28,
        "xtick.labelsize": 28,
        "ytick.labelsize": 28,
        "legend.fontsize": 26,
    }
    with mpl.rc_context(_cd_font_rc):
        cd_fig, _cd_ax = _critical_difference(
            pivot.values,
            list(pivot.columns),
            lower_better=not maximize,
            reverse=True,
            width=width,
        )

    # Every other rank tick, so that the rank labels do not overlap.
    for t in _cd_ax.texts:
        text_str = t.get_text()
        if text_str.isdigit() and t.get_position()[1] < 0.2 and int(text_str) % 2 == 0:
            t.set_visible(False)

    savefig(cd_fig, name, tight=True)
    plt.close(cd_fig)


# ── Panels ───────────────────────────────────────────────────────────


def _export_panels(
    df: pd.DataFrame,
    name_base: str,
    ylabel: str,
    show_hatch: bool,
    show_legend: bool = True,
    legend_loc: str = "upper left",
):
    """Strip, CD and spider panels of one (task, mode).

    The SD panels (``show_hatch=False``, few methods) have a narrow strip whose
    bar width matches the CPD strip once both are embedded in LaTeX.
    """
    square = not show_hatch
    with mpl.rc_context(_PANEL_RC):
        n_algos = max(1, df["algorithm"].nunique() - (1 if "random" in set(df["algorithm"]) else 0))
        if square:
            sd_w = max(TEXT_W * 0.22, n_algos * TEXT_W * 0.045)
            fig, ax = plt.subplots(figsize=(sd_w, TEXT_W * 0.25))
        else:
            fig, ax = plt.subplots(figsize=(TEXT_W * 0.55, TEXT_W * 0.25))
        plot_strip(
            ax,
            df,
            score_col="score_best",
            ylabel=ylabel,
            show_hatch=show_hatch,
            show_legend=show_legend,
            legend_ncol=1 if square else 2,
            ylim=(0.0, 0.8) if square else None,
            legend_loc=legend_loc if square else "above",
        )
        savefig(fig, f"{name_base}_strip", tight=True)

        _save_cd_directly(
            df, score_col="score_best", maximize=True, name=f"{name_base}_cd", width=5.0
        )

        # Spiders, saved at their printed size (half the text width).
        for group_by, suffix in (("domain", ""), ("dataset", "_dataset")):
            fig = plt.figure(figsize=(2.7, 3.0))
            ax = fig.add_subplot(111, polar=True)
            plot_spider(ax, df, score_col="score_best", group_by=group_by)
            savefig(fig, f"{name_base}_spider{suffix}", tight=True)
            plt.close(fig)


def make_all_figures(d=None):
    """The strip, CD and spider panels of both tasks and both modes.

    The legend of the CPD strips is shown on the Non-Guided panel, that of the
    SD strips on the Guided panel; the other panel shares it in the paper."""
    if d is None:
        d = load_data()
    _check_spider_styles(d.df_cpd, d.df_cpd_g)
    _export_panels(d.df_cpd, "fig_cpd_main", "Mean BiC", show_hatch=True)
    _check_spider_styles(d.df_sd, d.df_sd_g)
    _export_panels(d.df_sd, "fig_sd_main", "Mean SMS", show_hatch=False, show_legend=False)
    _export_panels(d.df_cpd_g, "fig_cpd_guided", "Mean BiC", show_hatch=True, show_legend=False)
    _export_panels(
        d.df_sd_g, "fig_sd_guided", "Mean SMS", show_hatch=False, legend_loc="lower right"
    )
