"""Runtime against accuracy: mean tuned score against the median runtime of
the tuned configuration, one point per method.

``runtime_panel`` draws one panel (Figure 6a-b uses it for CPD);
``make_figure`` writes the SD panels of the appendix,
``fig_runtime_accuracy_sd_{ng,g}``.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from analysis.data import load_data
from analysis.helpers import (
    display_name,
    savefig,
    select_best_grid_per_dataset,
)

METRIC_CPD = "bidirectional_covering_score"
METRIC_SD = "state_matching_score"
RUNTIME_COL = "execution_time_seconds"

# Hidalgo: extended scope only.
EXCLUDED_ALGOS = {"hidalgo"}


def _runtime_table(
    df_score: pd.DataFrame,
    df_grid: pd.DataFrame,
    score_metric: str,
    algo_filter: set,
) -> pd.DataFrame:
    """Per-algorithm (mean best-grid score, median best-grid runtime)."""
    s = df_score[df_score["algorithm"].isin(algo_filter)]
    s = s[s["dataset"] != "pamap2"]
    score = s.groupby("algorithm")["score_best"].mean()

    g = df_grid[df_grid["algorithm"].isin(algo_filter)].copy()
    g = g[g["dataset"] != "pamap2"]
    g = g.dropna(subset=[score_metric, RUNTIME_COL, "trial_index", "parent_run_id"])
    if g.empty:
        runtime = pd.Series(dtype=float)
    else:
        ranked = select_best_grid_per_dataset(g, score_metric, higher_is_better=True)
        best = ranked[ranked["config_rank"] == 1]
        runtime = best.groupby("algorithm")[RUNTIME_COL].median()

    out = pd.concat([score.rename("score"), runtime.rename("runtime")], axis=1).dropna()
    out = out[out["runtime"] > 0]
    return out


def _ylim_from_data(*tables, pad=0.05) -> tuple[float, float]:
    vals = np.concatenate([t["score"].to_numpy() for t in tables if not t.empty])
    if vals.size == 0:
        return 0.0, 1.0
    lo = max(0.0, float(vals.min()) - pad)
    hi = min(1.0, float(vals.max()) + pad)
    # round to nearest 0.05 for readable ticks
    lo = np.floor(lo * 20) / 20
    hi = np.ceil(hi * 20) / 20
    if hi - lo < 0.2:
        hi = min(1.0, lo + 0.2)
    return lo, hi


# Colour = method family as in the strip plots
# (CPD method blue, SD method orange, with a redundant marker shape), every
# point named in grey ink without overlap (place_labels); drawn at print size.
FAMILY_COLOR = {"CPD": "#2a78d6", "SD": "#eb6834"}
FAMILY_MARKER = {"CPD": "o", "SD": "s"}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
PANEL_RC = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "text.usetex": False,
    "font.size": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6,
    "axes.linewidth": 0.6,
    "axes.edgecolor": INK_2,
    "axes.labelcolor": INK,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "xtick.labelcolor": INK,
    "ytick.labelcolor": INK,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "xtick.major.size": 2.5,
    "xtick.minor.size": 1.5,
    "ytick.major.size": 2.5,
    "axes.grid": False,
    "pdf.fonttype": 42,
}


def _family(algo: str) -> str:
    from analysis.helpers import ALGO_TASK

    return "SD" if ALGO_TASK.get(algo) == "SD" else "CPD"


# Candidate label positions around a marker: (dx, dy) in points, alignment,
# preference penalty. The far ones get a leader line.
_NEAR = [
    (4, 0, "left", "center", 0.0),
    (-4, 0, "right", "center", 0.3),
    (0, 3.5, "center", "bottom", 0.5),
    (0, -3.5, "center", "top", 0.5),
    (3, 2.5, "left", "bottom", 0.7),
    (-3, 2.5, "right", "bottom", 0.8),
    (3, -2.5, "left", "top", 0.7),
    (-3, -2.5, "right", "top", 0.8),
]
_CANDIDATES = (
    [(*c, False) for c in _NEAR]
    + [(dx * 2.2, dy * 2.2, ha, va, pen + 1.5, True) for dx, dy, ha, va, pen in _NEAR]
    + [(dx * 3.5, dy * 3.5, ha, va, pen + 3.0, True) for dx, dy, ha, va, pen in _NEAR]
)


def _box_dist(b, p) -> float:
    """Distance from point ``p`` to the box ``b`` = (x0, y0, x1, y1)."""
    dx = max(b[0] - p[0], 0.0, p[0] - b[2])
    dy = max(b[1] - p[1], 0.0, p[1] - b[3])
    return float(np.hypot(dx, dy))


def place_labels(
    ax,
    xy: np.ndarray,
    names: list[str],
    colors: list[str],
    lines: list[np.ndarray] = (),
    fontsize: float = 5.8,
    fixed: dict[str, int] | None = None,
):
    """Name every marker without overlaps (labels, markers, lines, legend).

    Each label picks one of ``_CANDIDATES`` around its marker; the choice
    minimises, by coordinate descent, the overlap (in pt^2) with the other
    labels, all markers, the legend, the sampled ``lines`` (data-coordinate
    polylines) and the outside of the axes, plus a small preference penalty.
    ``fixed`` pins a label (by name) to one candidate index of ``_CANDIDATES``.
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pt = fig.dpi / 72.0
    disp = ax.transData.transform(xy)
    sizes = []
    for name in names:
        t = ax.text(0, 0, name, fontsize=fontsize)
        bb = t.get_window_extent(renderer)
        t.remove()
        sizes.append((bb.width, bb.height))
    frame = ax.get_window_extent(renderer)
    leg = ax.get_legend()
    leg_bb = leg.get_window_extent(renderer) if leg is not None else None
    samples = []
    for line in lines:
        d = ax.transData.transform(line)
        for (x0, y0), (x1, y1) in zip(d[:-1], d[1:]):
            n = max(2, int(np.hypot(x1 - x0, y1 - y0) / (0.5 * pt)))
            samples.append(np.column_stack([np.linspace(x0, x1, n), np.linspace(y0, y1, n)]))
    samples = np.vstack(samples) if samples else np.empty((0, 2))
    r_mark = 3.0 * pt
    pad = 0.6 * pt

    def box(i, c):
        dx, dy, ha, va = c[0] * pt, c[1] * pt, c[2], c[3]
        w, h = sizes[i]
        x, y = disp[i, 0] + dx, disp[i, 1] + dy
        x0 = {"left": x, "right": x - w, "center": x - w / 2}[ha]
        y0 = {"bottom": y, "top": y - h, "center": y - h / 2}[va]
        return np.array([x0 - pad, y0 - pad, x0 + w + pad, y0 + h + pad])

    def inter(a, b):
        return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
            0.0, min(a[3], b[3]) - max(a[1], b[1])
        )

    marks = [np.array([x - r_mark, y - r_mark, x + r_mark, y + r_mark]) for x, y in disp]
    n = len(names)
    boxes = [[box(i, c) for c in _CANDIDATES] for i in range(n)]

    def leader(i, k):
        """Points along the leader line of label i at candidate k (or None)."""
        if not _CANDIDATES[k][5]:
            return None
        b = boxes[i][k]
        end = np.array([min(max(disp[i, 0], b[0]), b[2]), min(max(disp[i, 1], b[1]), b[3])])
        t = np.linspace(0.15, 1.0, 16)[:, None]
        return disp[i] + t * (end - disp[i])

    leaders = [[leader(i, k) for k in range(len(_CANDIDATES))] for i in range(n)]

    def hits(pts, b):
        if pts is None:
            return 0
        return int(
            (
                (pts[:, 0] > b[0]) & (pts[:, 0] < b[2]) & (pts[:, 1] > b[1]) & (pts[:, 1] < b[3])
            ).sum()
        )

    def own_cost(i, k):
        b, c = boxes[i][k], _CANDIDATES[k]
        cost = c[4] * 10.0
        cost += sum(inter(b, m) for m in marks) / pt**2 * 8.0
        area = (b[2] - b[0]) * (b[3] - b[1])
        inside = inter(b, [frame.x0, frame.y0, frame.x1, frame.y1])
        cost += (area - inside) / pt**2 * 3.0
        if leg_bb is not None:
            cost += inter(b, [leg_bb.x0, leg_bb.y0, leg_bb.x1, leg_bb.y1]) / pt**2 * 3.0
        # Ambiguity: the label must sit nearer its own marker than any other.
        d = [_box_dist(b, disp[j]) for j in range(n)]
        d_other = min((d[j] for j in range(n) if j != i), default=np.inf)
        if d_other < d[i] + 3.0 * pt:
            cost += 15.0 + (d[i] + 3.0 * pt - d_other) / pt * 5.0
        if len(samples):
            hit = (
                (samples[:, 0] > b[0])
                & (samples[:, 0] < b[2])
                & (samples[:, 1] > b[1])
                & (samples[:, 1] < b[3])
            )
            cost += hit.sum() * 6.0
        lead = leaders[i][k]
        if lead is not None:
            cost += sum(hits(lead, marks[j]) for j in range(n) if j != i) * 20.0
        return cost

    m = len(_CANDIDATES)
    own = np.array([[own_cost(i, k) for k in range(m)] for i in range(n)])
    for name, k in (fixed or {}).items():
        if name in names:
            i = names.index(name)
            own[i, [j for j in range(m) if j != k]] = 1e9
    # pair[i, k, j, q]: overlap of label i at candidate k with label j at q.
    pair = np.zeros((n, m, n, m))
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(m):
                for q in range(m):
                    v = inter(boxes[i][k], boxes[j][q]) / pt**2 * 6.0
                    v += (
                        hits(leaders[i][k], boxes[j][q]) + hits(leaders[j][q], boxes[i][k])
                    ) * 20.0
                    pair[i, k, j, q] = pair[j, q, i, k] = v

    def total(ch):
        return sum(own[i, ch[i]] for i in range(n)) + sum(
            pair[i, ch[i], j, ch[j]] for i in range(n) for j in range(i + 1, n)
        )

    rng = np.random.default_rng(0)
    best, best_cost = None, np.inf
    for restart in range(40):
        ch = np.argmin(own, axis=1) if restart == 0 else rng.integers(0, m, size=n)
        for _ in range(60):
            changed = False
            for i in rng.permutation(n):
                costs = own[i] + sum(pair[i, :, j, ch[j]] for j in range(n) if j != i)
                k = int(np.argmin(costs))
                if k != ch[i]:
                    ch[i], changed = k, True
            if not changed:
                break
        c = total(ch)
        if c < best_cost:
            best, best_cost = ch.copy(), c
    choice = [int(k) for k in best]

    for i, k in enumerate(choice):
        dx, dy, ha, va, _, leader = _CANDIDATES[k]
        ax.annotate(
            names[i],
            tuple(xy[i]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=fontsize,
            color=colors[i],
            zorder=5,
            bbox=dict(boxstyle="square,pad=0.08", facecolor="white", edgecolor="none"),
            arrowprops=(
                dict(arrowstyle="-", color=INK_2, lw=0.4, shrinkA=0.5, shrinkB=2.5)
                if leader
                else None
            ),
        )


def runtime_panel(
    ax,
    tbl: pd.DataFrame,
    ylim: tuple[float, float],
    xlim: tuple[float, float],
    ylabel: str | None = None,
    legend: bool | str = False,
    fixed: dict[str, int] | None = None,
    label_fs: float = 5.8,
    marker_scale: float = 1.0,
    better: bool = True,
):
    """One runtime-accuracy panel (call inside ``mpl.rc_context(PANEL_RC)``).

    ``legend`` is False or a matplotlib legend location; ``label_fs`` is the
    size of the method names, ``marker_scale`` scales the marker areas;
    ``better`` draws an arrow towards the upper-left corner (fast and accurate).
    """
    for algo, row in tbl.iterrows():
        fam = _family(algo)
        ax.scatter(
            row["runtime"],
            row["score"],
            marker=FAMILY_MARKER[fam],
            s=(16 if fam == "CPD" else 13) * marker_scale,
            color=FAMILY_COLOR[fam],
            edgecolors="white",
            linewidths=0.5,
            zorder=4,
        )

    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Median runtime (s)")
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)

    if legend:
        families = {_family(a) for a in tbl.index}
        handles = [
            Line2D(
                [],
                [],
                marker="o",
                ls="",
                color=FAMILY_COLOR["CPD"],
                markeredgecolor="white",
                markersize=4,
                label="CPD method",
            ),
            Line2D(
                [],
                [],
                marker="s",
                ls="",
                color=FAMILY_COLOR["SD"],
                markeredgecolor="white",
                markersize=3.6,
                label="SD method",
            ),
        ]
        handles = [h for h, fam in zip(handles, ("CPD", "SD")) if fam in families]
        ax.legend(
            handles=handles,
            loc=legend if isinstance(legend, str) else "lower left",
            frameon=False,
            handlelength=1.2,
            handletextpad=0.4,
            borderaxespad=0.2,
            labelspacing=0.3,
        )

    if better:
        # Fast and accurate methods sit in the upper-left corner.
        ax.annotate(
            "better",
            xy=(0.02, 0.98),
            xycoords="axes fraction",
            xytext=(0.05, 0.86),
            textcoords="axes fraction",
            ha="left",
            va="top",
            fontsize=label_fs,
            color=INK_2,
            style="italic",
            arrowprops=dict(
                arrowstyle="-|>", color=INK_2, lw=0.6, mutation_scale=5, shrinkA=0, shrinkB=0
            ),
        )

    algos = list(tbl.index)
    place_labels(
        ax,
        tbl[["runtime", "score"]].to_numpy(float),
        [display_name(a) for a in algos],
        [INK_2] * len(algos),
        fixed=fixed,
        fontsize=label_fs,
    )


def _xlim(*tables) -> tuple[float, float]:
    xs = np.concatenate([t["runtime"].to_numpy() for t in tables if not t.empty])
    return float(xs.min()) / 2.5, float(xs.max()) * 1.6


def make_figure(d=None):
    if d is None:
        d = load_data()

    cpd_algos = {a for a in d.cpd_algos if a != "random" and a not in EXCLUDED_ALGOS}
    sd_algos = {a for a in d.sd_algos if a != "random" and a not in EXCLUDED_ALGOS}

    ng_cpd = _runtime_table(d.df_cpd, d.df_grid_ng, METRIC_CPD, cpd_algos)
    g_cpd = _runtime_table(d.df_cpd_g, d.df_grid_g, METRIC_CPD, cpd_algos)
    ng_sd = _runtime_table(d.df_sd, d.df_grid_ng, METRIC_SD, sd_algos)
    g_sd = _runtime_table(d.df_sd_g, d.df_grid_g, METRIC_SD, sd_algos)

    # The CPD panels are drawn by rq3_figure.py (Figure 6a-b); the SD panels
    # go to the appendix, at their print size.
    sd_ylim = _ylim_from_data(ng_sd, g_sd, pad=0.03)
    sd_xlim = _xlim(ng_sd, g_sd)
    for tbl, name in ((ng_sd, "fig_runtime_accuracy_sd_ng"), (g_sd, "fig_runtime_accuracy_sd_g")):
        with mpl.rc_context(PANEL_RC):
            fig = plt.figure(figsize=(1.75, 1.75))
            ax = fig.add_axes([0.40 / 1.75, 0.34 / 1.75, 1.30 / 1.75, 1.36 / 1.75])
            ng = name.endswith("_ng")
            runtime_panel(
                ax,
                tbl,
                sd_ylim,
                sd_xlim,
                ylabel="Mean tuned SMS",
                # Guided: Time2State labelled on its right, clear of the arrow.
                fixed=({"HDP-HSMM": 3} if ng else {"Time2State": 0}),
            )
            savefig(fig, name, tight=False)

    print(f"CPD: {len(ng_cpd)} NG, {len(g_cpd)} G")
    print(f"SD : {len(ng_sd)} NG, {len(g_sd)} G")
