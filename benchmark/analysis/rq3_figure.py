"""FIGURE_RQ3 (paper Figure 6): runtime-accuracy for CPD, SMS errors for SD.

(a, b) Mean tuned BiC against the median runtime of the tuned
configuration, Non-Guided and Guided, on a shared score axis
(``runtime.runtime_panel``). (c) Average share of each SMS error type for
every SD algorithm and supervision mode, from ``sms_error_breakdown.parquet``,
with the error colours of Figure 5c.

Drawn at print size (5.5 in, the NeurIPS text width).
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory

from analysis.data import load_data
from analysis.error_mix import (
    PENALTY_COLORS,
    PENALTY_COLS,
    PENALTY_LABELS,
    read_sms_breakdown,
)
from analysis.helpers import display_name, savefig
from analysis.runtime import (
    EXCLUDED_ALGOS,
    INK,
    INK_2,
    METRIC_CPD,
    PANEL_RC,
    _runtime_table,
    _xlim,
    _ylim_from_data,
    runtime_panel,
)

FIG_W, FIG_H = 5.5, 1.97
BOTTOM, AX_H = 0.30, 1.42
TITLE_Y = BOTTOM + AX_H + 0.15

# Text one step smaller than the appendix panels (PANEL_RC): the two CPD
# panels name twenty methods each.
RQ3_RC = {
    **PANEL_RC,
    "font.size": 6,
    "axes.labelsize": 6.2,
    "xtick.labelsize": 5.6,
    "ytick.labelsize": 5.6,
    "legend.fontsize": 5.4,
}
FS_TITLE = 6.6
FS_NAME = 4.9  # method names in (a, b)
FS_SD_NAME = 5.5  # method names in (c)
FS_TAG = 4.8  # NG / G tags in (c)

# SD algorithms grouped as in the taxonomy: segment-based, representation-
# based, model-based (Hidalgo, extended scope, last).
SD_ORDER = ["clap", "time2state", "e2usd", "hdp-hsmm", "ticc", "autoplait", "hidalgo"]
SUPS = (("unsupervised", "NG"), ("semi_supervised", "G"))


def _sms_shares(d) -> pd.DataFrame:
    df = read_sms_breakdown(d)
    df = df[(df["dataset"] != "pamap2") & ~df["algorithm"].isin(["random", "ggs"])]
    totals = df[PENALTY_COLS].sum(axis=1).replace(0, np.nan)
    shares = df[PENALTY_COLS].div(totals, axis=0)
    shares = pd.concat([df[["algorithm", "supervision"]], shares], axis=1).dropna()
    return shares.groupby(["algorithm", "supervision"])[PENALTY_COLS].mean()


def _axes(fig, left: float, width: float):
    return fig.add_axes([left / FIG_W, BOTTOM / FIG_H, width / FIG_W, AX_H / FIG_H])


def _title(fig, x_in: float, text: str):
    fig.text(
        x_in / FIG_W, TITLE_Y / FIG_H, text, ha="left", va="baseline", fontsize=FS_TITLE, color=INK
    )


def _sms_panel(ax, props: pd.DataFrame):
    bar_h, pair_gap, group_gap = 0.78, 0.12, 0.6
    y, ticks, tags, groups = 0.0, [], [], []
    for algo in SD_ORDER:
        present = [(sup, tag) for sup, tag in SUPS if (algo, sup) in props.index]
        if not present:
            continue
        ys = []
        for sup, tag in present:
            left = 0.0
            for col in PENALTY_COLS:
                v = float(props.loc[(algo, sup), col])
                ax.barh(
                    y,
                    v,
                    left=left,
                    height=bar_h,
                    color=PENALTY_COLORS[col],
                    edgecolor="white",
                    linewidth=0.5,
                    zorder=2,
                )
                left += v
            ticks.append(y)
            tags.append(tag)
            ys.append(y)
            y += 1.0 + pair_gap
        groups.append((display_name(algo), float(np.mean(ys))))
        y += group_gap
    ax.set_ylim(y - group_gap - 0.4, -3.4)  # top-down; room for the legend on top
    # Method names on the left, the NG/G tag of each bar on its right.
    ax.set_yticks([yc for _, yc in groups])
    ax.set_yticklabels([name for name, _ in groups], fontsize=FS_SD_NAME, color=INK)
    ax.tick_params(axis="y", length=0, pad=2)
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    for yt, tag in zip(ticks, tags):
        ax.text(
            1.03, yt, tag, transform=trans, ha="left", va="center", fontsize=FS_TAG, color=INK_2
        )
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1, decimals=0))
    ax.set_xlabel("Share of SMS penalty")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    handles = [
        Patch(facecolor=PENALTY_COLORS[c], edgecolor="none", label=lab)
        for c, lab in zip(PENALTY_COLS, PENALTY_LABELS)
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=2,
        frameon=False,
        handlelength=0.9,
        handleheight=0.8,
        handletextpad=0.35,
        columnspacing=0.8,
        borderaxespad=0,
        labelspacing=0.25,
        fontsize=5.4,
    )


def make_figure(d=None):
    if d is None:
        d = load_data()
    cpd_algos = {a for a in d.cpd_algos if a != "random" and a not in EXCLUDED_ALGOS}
    ng = _runtime_table(d.df_cpd, d.df_grid_ng, METRIC_CPD, cpd_algos)
    g = _runtime_table(d.df_cpd_g, d.df_grid_g, METRIC_CPD, cpd_algos)
    ylim = _ylim_from_data(ng, g, pad=0.02)
    xlim = _xlim(ng, g)
    props = _sms_shares(d)
    print(ng.round(3).sort_values("score"))
    print(g.round(3).sort_values("score"))

    with mpl.rc_context(RQ3_RC):
        fig = plt.figure(figsize=(FIG_W, FIG_H))
        ax_a = _axes(fig, 0.38, 1.54)
        ax_b = _axes(fig, 2.02, 1.54)
        ax_c = _axes(fig, 4.22, 1.12)
        # Legend lower right: AMOC is lower left.
        runtime_panel(
            ax_a,
            ng,
            ylim,
            xlim,
            ylabel="Mean tuned BiC",
            legend="lower right",
            label_fs=FS_NAME,
            marker_scale=0.8,
        )
        # BottomUp to the right of its marker, away from GGS.
        # The "better" arrow is drawn once, in panel (a).
        runtime_panel(
            ax_b,
            g,
            ylim,
            xlim,
            fixed={"BottomUp": 0},
            label_fs=FS_NAME,
            marker_scale=0.8,
            better=False,
        )
        ax_b.tick_params(axis="y", labelleft=False)
        for ax in (ax_a, ax_b):
            ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.1))
        _sms_panel(ax_c, props)
        _title(fig, 0.08, "(a) CPD, Non-Guided")
        _title(fig, 2.02, "(b) CPD, Guided")
        _title(fig, 3.62, "(c) SD, SMS error types")
        savefig(fig, "fig_rq3", tight=False)
