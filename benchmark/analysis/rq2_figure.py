"""FIGURE_RQ2 (paper Figure 5): CPD vs. SD on every SD run, three panels.

(a) SMS against BiC, (b) histogram of the residual r = SMS - BiC and
(c) the average SMS penalty mix in each residual zone. The three panels
share one population (``residual_dist.pool_runs``: SD algorithms, PAMAP2
excluded, both scores in [0, 1], one row per run) and one envelope (the
two-sided 95 % band of ``residual_dist.envelope``), so the figure matches
the numbers quoted in the text.

Drawn at print size (5.5 in, the NeurIPS text width) so the fonts are the
sizes read on paper. Colour has two jobs only: dark ink marks the runs
outside the envelope (the two tails), grey the runs inside it; the four
SMS error types use ``error_mix.PENALTY_COLORS`` (also used by Figure 6c).
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from analysis.data import load_data
from analysis.error_mix import (
    PENALTY_COLORS,
    PENALTY_COLS,
    PENALTY_LABELS,
    PENALTY_TEXT,
    penalty_mix_by_zone,
)
from analysis.helpers import METRIC_CPD, METRIC_SD, savefig
from analysis.residual_dist import assign_zone, envelope, pool_runs

INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e1e0d9"
C_INSIDE = "#a8a7a1"  # runs inside the envelope
C_TAIL = "#2b2a28"  # runs outside it

RC = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "text.usetex": False,
    "font.size": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.6,
    "axes.edgecolor": INK_2,
    "axes.labelcolor": INK,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "xtick.labelcolor": INK,
    "ytick.labelcolor": INK,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.grid": False,
    "pdf.fonttype": 42,
}

FIG_W, FIG_H = 5.5, 1.95
BOTTOM, AX_H = 0.34, 1.30  # inches
TITLE_Y = BOTTOM + AX_H + 0.17  # baseline of the panel titles


def _axes(fig, left: float, width: float):
    return fig.add_axes([left / FIG_W, BOTTOM / FIG_H, width / FIG_W, AX_H / FIG_H])


def _clean(ax, grid_axis: str | None = None):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)


def _title(fig, ax, text: str):
    x0 = ax.get_position().x0
    fig.text(
        x0 - 0.30 / FIG_W, TITLE_Y / FIG_H, text, ha="left", va="baseline", fontsize=7.5, color=INK
    )


def _gt(up: float) -> str:
    return f"$r > {{+}}{up:.2f}$"


def _lt(lo: float) -> str:
    return f"$r < {{-}}{lo:.2f}$"


def _scatter(ax, df, lo, up):
    x, y = df[METRIC_CPD].to_numpy(float), df[METRIC_SD].to_numpy(float)
    inside = df["zone"].eq("near").to_numpy()
    ax.scatter(
        x[inside], y[inside], s=1.0, c=C_INSIDE, alpha=0.10, linewidths=0, rasterized=True, zorder=2
    )
    ax.scatter(
        x[~inside], y[~inside], s=1.0, c=C_TAIL, alpha=0.45, linewidths=0, rasterized=True, zorder=3
    )
    xs = np.linspace(0, 1, 101)
    ax.plot(xs, xs, color=INK, lw=0.7, zorder=4)
    for off in (up, -lo):
        ax.plot(xs, xs + off, color=INK, lw=0.7, ls=(0, (3, 2)), zorder=4)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_xticklabels(["0", "0.5", "1"])
    ax.set_yticklabels(["0", "0.5", "1"])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("BiC")
    ax.set_ylabel("SMS", labelpad=1)
    box = dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85)
    ax.text(
        0.03,
        0.97,
        f"upper tail\n{_gt(up)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6,
        color=INK,
        linespacing=1.1,
        bbox=box,
        zorder=6,
    )
    ax.text(
        0.97,
        0.03,
        f"lower tail\n{_lt(lo)}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6,
        color=INK,
        linespacing=1.1,
        bbox=box,
        zorder=6,
    )
    _clean(ax)


def _histogram(ax, df, lo, up):
    r = df["residual"].to_numpy(float)
    bins = np.linspace(-0.6, 0.72, 67)
    w = np.full(r.shape, 1.0 / len(r))
    for zone, color in (("near", C_INSIDE), ("under", C_TAIL), ("over", C_TAIL)):
        m = df["zone"].eq(zone).to_numpy()
        ax.hist(
            np.clip(r[m], -0.6, 0.72),
            bins=bins,
            weights=w[m],
            color=color,
            edgecolor="white",
            linewidth=0.25,
            zorder=2,
        )
    top = ax.get_ylim()[1]
    ax.axvline(0, color=INK, lw=0.7, zorder=3)
    for v, txt in ((-lo, f"${{-}}{lo:.2f}$"), (up, f"${{+}}{up:.2f}$")):
        ax.axvline(v, color=INK, lw=0.7, ls=(0, (3, 2)), ymax=0.6, zorder=3)
        ax.text(v, top * 0.62, txt, ha="center", va="bottom", fontsize=6, color=INK)
    near0 = float(np.mean(np.abs(r) <= 0.05))
    ax.annotate(
        f"{near0:.0%} of runs\nwith $|r| \\leq 0.05$",
        xy=(0.035, top * 0.88),
        xytext=(0.72, top * 0.98),
        fontsize=6,
        color=INK,
        ha="right",
        va="top",
        linespacing=1.1,
        arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.5, shrinkA=1, shrinkB=1),
    )
    ax.set_xlim(-0.6, 0.72)
    ax.set_xticks([-0.5, 0, 0.5])
    ax.set_xticklabels(["$-0.5$", "0", "$+0.5$"])
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1, decimals=0))
    ax.set_yticks([0, 0.05, 0.10])
    ax.set_xlabel("$r$ = SMS $-$ BiC")
    ax.set_ylabel("Share of runs", labelpad=1)
    _clean(ax, "y")


def _penalty_mix(ax, mix: pd.DataFrame, lo, up):
    rows = [
        ("over", f"upper tail\n{_gt(up)}"),
        ("near", "inside"),
        ("under", f"lower tail\n{_lt(lo)}"),
    ]
    y = np.arange(len(rows))[::-1].astype(float)  # upper tail on top
    for yi, (zone, _) in zip(y, rows):
        left = 0.0
        for col in PENALTY_COLS:
            v = float(mix.loc[zone, col])
            ax.barh(
                yi,
                v,
                left=left,
                height=0.62,
                color=PENALTY_COLORS[col],
                edgecolor="white",
                linewidth=0.6,
                zorder=2,
            )
            if v >= 0.12:
                ax.text(
                    left + v / 2,
                    yi,
                    f"{v:.0%}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color=PENALTY_TEXT[col],
                    zorder=3,
                )
            left += v
    ax.set_yticks(y)
    ax.set_yticklabels([lab for _, lab in rows], linespacing=1.1)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_ylim(-0.5, len(rows) - 0.5 + 0.75)  # room for the legend on top
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1, decimals=0))
    ax.set_xlabel("Share of SMS penalty")
    ax.spines["left"].set_visible(False)
    _clean(ax)
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
        columnspacing=0.9,
        borderaxespad=0,
        labelspacing=0.25,
    )


def make_figure(d=None):
    if d is None:
        d = load_data()
    df = pool_runs(d)
    lo, up = envelope(df)
    df = assign_zone(df, lo, up)
    mix = penalty_mix_by_zone(df, d)
    print(
        f"[rq2] N = {len(df):,}  envelope = [-{lo:.3f}, +{up:.3f}]  "
        f"inside = {df['zone'].eq('near').mean():.1%}"
    )
    print((mix * 100).round(1))

    with mpl.rc_context(RC):
        fig = plt.figure(figsize=(FIG_W, FIG_H))
        ax_a = _axes(fig, 0.36, AX_H)
        ax_b = _axes(fig, 2.12, 1.38)
        ax_c = _axes(fig, 4.18, 1.28)
        _scatter(ax_a, df, lo, up)
        _histogram(ax_b, df, lo, up)
        _penalty_mix(ax_c, mix, lo, up)
        _title(fig, ax_a, "(a) SMS against BiC")
        _title(fig, ax_b, "(b) Residual distribution")
        fig.text(
            (ax_c.get_position().x0 - 0.62 / FIG_W),
            TITLE_Y / FIG_H,
            "(c) Penalty mix per zone",
            ha="left",
            va="baseline",
            fontsize=7.5,
            color=INK,
        )
        savefig(fig, "fig_rq2", tight=False)
