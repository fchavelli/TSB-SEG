"""Sensitivity of the tuned scores to the features of the series (appendix).

For every algorithm and every supervision mode (NG = non-guided,
G = guided), we compute the Spearman rank correlation between its
``score_best`` (best grid configuration on the target metric) and four
per-series features:

    * length            (number of samples)
    * dimensions        (number of channels)
    * n_change_points   (ground-truth segmentation count)
    * n_states          (ground-truth state count)

Two heatmaps are produced side by side:
  - CPD score, all algorithms (CLaP scored through ClaSP);
  - SD score, SD algorithms only.

One row per algorithm, the four features in each supervision mode as
columns; "-" marks a mode the algorithm does not run in. The ``random``
baseline is dropped.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats as _scstats

from analysis.data import load_data
from analysis.helpers import (
    ALGO_DISPLAY,
    ALGO_TASK,
    BENCH_DIR,
    savefig,
)

FEATURES_CSV = BENCH_DIR / "datasets" / "dataset_features.csv"

FEATURE_COLS = ["length", "dimensions", "n_change_points", "n_states"]
FEATURE_LABELS = {
    "length": r"$n$",
    "dimensions": r"$d$",
    "n_change_points": r"$n_{CP}$",
    "n_states": r"$n_S$",
}


def _load_features() -> pd.DataFrame:
    f = pd.read_csv(FEATURES_CSV).rename(columns={"series": "trial_index"})
    f["trial_index"] = f["trial_index"].astype("Int64")
    return f[["dataset", "trial_index", *FEATURE_COLS]]


def _merge_features(df: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["trial_index"] = pd.to_numeric(df["trial_index"], errors="coerce").astype("Int64")
    return df.merge(feats, on=["dataset", "trial_index"], how="left")


def _spearman_table(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    rows: dict[str, dict[str, float]] = {}
    for algo, sub in df.groupby("algorithm"):
        sub = sub.dropna(subset=[score_col, *FEATURE_COLS])
        if len(sub) < 5:
            continue
        rho_row: dict[str, float] = {}
        for feat in FEATURE_COLS:
            x = sub[feat].to_numpy(dtype=float)
            y = sub[score_col].to_numpy(dtype=float)
            if np.unique(x).size < 2 or np.unique(y).size < 2:
                rho_row[feat] = np.nan
            else:
                rho, _ = _scstats.spearmanr(x, y)
                rho_row[feat] = float(rho)
        rows[algo] = rho_row
    if not rows:
        return pd.DataFrame(columns=FEATURE_COLS)
    return pd.DataFrame.from_dict(rows, orient="index")[FEATURE_COLS]


def _build_task_table(
    df_ng: pd.DataFrame,
    df_g: pd.DataFrame,
    feats: pd.DataFrame,
    algo_filter: set | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    df_ng = _merge_features(df_ng, feats)
    df_g = _merge_features(df_g, feats)
    if algo_filter is not None:
        df_ng = df_ng[df_ng["algorithm"].isin(algo_filter)]
        df_g = df_g[df_g["algorithm"].isin(algo_filter)]
    df_ng = df_ng[df_ng["algorithm"] != "random"]
    df_g = df_g[df_g["algorithm"] != "random"]

    corr_ng = _spearman_table(df_ng, "score_best")
    corr_g = _spearman_table(df_g, "score_best")

    algos = sorted(set(corr_ng.index) | set(corr_g.index))

    score_means: dict[str, float] = {}
    for a in algos:
        vals = []
        for src in (df_ng, df_g):
            v = src.loc[src["algorithm"] == a, "score_best"].dropna().to_numpy()
            if v.size:
                vals.append(float(v.mean()))
        score_means[a] = float(np.mean(vals)) if vals else -np.inf
    algos = sorted(algos, key=lambda a: score_means[a], reverse=True)

    rows: list[list[float]] = []
    support: list[list[bool]] = []
    index: list[tuple[str, str]] = []
    for a in algos:
        for mode, table in (("NG", corr_ng), ("G", corr_g)):
            if a not in table.index:
                continue
            index.append((a, mode))
            rows.append(table.loc[a].to_numpy(dtype=float).tolist())
            support.append([True] * len(FEATURE_COLS))
    corr = pd.DataFrame(
        rows,
        index=pd.MultiIndex.from_tuples(index, names=("algorithm", "mode")),
        columns=FEATURE_COLS,
    )
    return corr, np.asarray(support, dtype=bool)


# Diverging scale: red when the score falls as the feature grows,
# blue when it rises, a neutral grey at zero (dataviz reference diverging pair).
CMAP = LinearSegmentedColormap.from_list(
    "rho", ["#a3302f", "#e8908a", "#f0efec", "#86b6ef", "#184f95"]
)
_NA_COLOR = "#ffffff"
_INK, _INK_2, _GRID = "#0b0b0b", "#52514e", "#e1e0d9"
_MODE_LABEL = {"NG": "Non-Guided", "G": "Guided"}


def _wide(corr: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """One row per algorithm, columns (mode, feature): NG block then G block."""
    algos = list(dict.fromkeys(a for a, _m in corr.index))
    cols = [(m, f) for m in ("NG", "G") for f in FEATURE_COLS]
    out = pd.DataFrame(np.nan, index=algos, columns=pd.MultiIndex.from_tuples(cols))
    for (a, m), row in corr.iterrows():
        for f in FEATURE_COLS:
            out.loc[a, (m, f)] = row[f]
    return out, algos


def _draw_heatmap(ax, corr, support, title):
    wide, algos = _wide(corr)
    data = wide.to_numpy(dtype=float)
    n_rows, n_cols = data.shape
    im = ax.imshow(
        np.ma.masked_invalid(data),
        cmap=CMAP,
        vmin=-1.0,
        vmax=1.0,
        aspect="auto",
        interpolation="none",
    )
    ax.set_facecolor(_NA_COLOR)
    for i in range(n_rows):
        for j in range(n_cols):
            v = data[i, j]
            if np.isnan(v):
                ax.text(j, i, "-", ha="center", va="center", fontsize=6, color=_INK_2)
                continue
            s = f"{abs(v):.2f}".lstrip("0")
            ax.text(
                j,
                i,
                ("−" + s) if v < 0 else s,
                ha="center",
                va="center",
                fontsize=5.5,
                color="white" if abs(v) > 0.6 else _INK,
            )
    # white gaps between cells, a wider one between the two modes
    ax.set_xticks(np.arange(n_cols + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_rows + 1) - 0.5, minor=True)
    ax.grid(False, which="major")
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.axvline(len(FEATURE_COLS) - 0.5, color="white", linewidth=4)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels([FEATURE_LABELS[f] for _m, f in wide.columns], fontsize=6.5, color=_INK)
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels([ALGO_DISPLAY.get(a, a) for a in algos], fontsize=6.5, color=_INK)
    for k, m in enumerate(("NG", "G")):
        ax.text(
            (k + 0.5) * len(FEATURE_COLS) - 0.5,
            -1.75,
            _MODE_LABEL[m],
            ha="center",
            va="bottom",
            fontsize=6.5,
            color=_INK_2,
        )
    ax.set_title(title, fontsize=7.5, color=_INK, pad=24, loc="left")
    return im


def make_figure(d=None):
    if d is None:
        d = load_data()

    feats = _load_features()
    corr_cpd, sup_cpd = _build_task_table(d.df_cpd, d.df_cpd_g, feats)
    sd_algo_set = {a for a, t in ALGO_TASK.items() if t == "SD"}
    corr_sd, sup_sd = _build_task_table(
        d.df_sd,
        d.df_sd_g,
        feats,
        algo_filter=sd_algo_set,
    )
    n_cpd = len(dict.fromkeys(a for a, _m in corr_cpd.index))
    n_sd = len(dict.fromkeys(a for a, _m in corr_sd.index))

    # CPD on top, SD below, same columns; rows = methods. Saved at print size
    # (text width 5.5 in) without usetex, like the other figures.
    row_h = 0.15
    with plt.rc_context({"text.usetex": False}):
        fig = plt.figure(figsize=(5.5, (n_cpd + n_sd) * row_h + 1.5))
        gs = fig.add_gridspec(3, 1, height_ratios=[n_cpd, n_sd, 1.2], hspace=0.6)
        ax_cpd = fig.add_subplot(gs[0, 0])
        ax_sd = fig.add_subplot(gs[1, 0])
        ax_cb = fig.add_subplot(gs[2, 0])
        _draw_heatmap(ax_cpd, corr_cpd, sup_cpd, "CPD score (BiC)")
        im = _draw_heatmap(ax_sd, corr_sd, sup_sd, "SD score (SMS)")
        pos = ax_cb.get_position()
        ax_cb.set_position(
            [
                pos.x0 + pos.width * 0.3,
                pos.y0 + pos.height * 0.55,
                pos.width * 0.4,
                pos.height * 0.22,
            ]
        )
        cbar = plt.colorbar(im, cax=ax_cb, orientation="horizontal")
        cbar.set_label(r"Spearman $\rho$", fontsize=6.5, color=_INK)
        cbar.ax.tick_params(labelsize=6, colors=_INK_2, length=2)
        cbar.outline.set_visible(False)
        savefig(fig, "fig_feature_sensitivity")
    plt.close(fig)
