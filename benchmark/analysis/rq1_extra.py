"""Additional views of the RQ1 results (appendix).

Three outputs, all computed from the per-series tuned and default scores of
``load_data``:

* ``table_oracle_selection.tex``: single best method (highest mean tuned score)
  against a per-series oracle that picks the best method of each series, in each
  (task, mode); tabular body only, the caption lives in the paper.
* ``fig_{cpd,sd}_{main,guided}_cd_default.pdf``: critical-difference diagrams of
  the Default configurations (same test and layout as the tuned ones).
* ``fig_dataset_boxplots_{cpd,sd}.pdf``: per-dataset distribution of the
  per-series tuned score, Non-Guided (light left half) and Guided (dark right
  half), as in the runtime split figures.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.data import GUIDED_FROM_NON_GUIDED, load_data
from analysis.helpers import ALGO_TASK, MAIN_DATASETS, OUTPUT_DIR, display_name, savefig
from analysis.performance import SPIDER_TOP_K, _save_cd_directly, _top_algos
from analysis.runtime import FAMILY_COLOR, GRID, INK
from analysis.runtime_split import SPLIT_RC, _tint, legend_handles

DATASETS = MAIN_DATASETS
DATASET_DISPLAY = {
    "actrectut": "ActRecTut",
    "has": "HAS",
    "mocap": "MoCap",
    "pump": "PUMP",
    "skab": "SKAB",
    "tssb": "TSSB",
    "usc-had": "USC-HAD",
    "utsa": "UTSA",
}
DROP = {"random"}


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df["algorithm"].isin(DROP) & df["dataset"].isin(DATASETS)]


def _pivot(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return _clean(df).pivot_table(
        index=["dataset", "trial_index"], columns="algorithm", values=col, aggfunc="first"
    )


# ─────────────────────────────────────────────────────────────────────
# 1. Single best method vs per-series oracle
# ─────────────────────────────────────────────────────────────────────


def oracle_rows(d) -> list[dict]:
    rows = []
    for task, mode, df in [
        ("CPD", "Non-Guided", d.df_cpd),
        ("CPD", "Guided", d.df_cpd_g),
        ("SD", "Non-Guided", d.df_sd),
        ("SD", "Guided", d.df_sd_g),
    ]:
        p = _pivot(df, "score_best")
        assert not p.isna().any().any(), f"{task} {mode}: missing tuned scores"
        means = p.mean().sort_values(ascending=False)
        sbs = means.index[0]
        others = p.drop(columns=sbs).max(axis=1)
        rows.append(
            {
                "task": task,
                "mode": mode,
                "n_methods": p.shape[1],
                "n_series": len(p),
                "sbs": display_name(sbs),
                "sbs_score": means.iloc[0],
                "sbs_strict_best": float((p[sbs] > others).mean()),
                "oracle_top3": p[means.index[:3]].max(axis=1).mean(),
                "oracle_all": p.max(axis=1).mean(),
            }
        )
    return rows


def oracle_table(rows: list[dict]) -> str:
    lines = [
        r"\begin{tabular}{l l c l c c c c}",
        r"\toprule",
        r"\textbf{Task} & \textbf{Mode} & \textbf{\#Methods} & \textbf{Single best} & "
        r"\textbf{Score} & \textbf{Strict best on} & \textbf{Oracle, top 3} & "
        r"\textbf{Oracle, all} \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r['task']} & {r['mode']} & {r['n_methods']} & {r['sbs']} & "
            f"{r['sbs_score']:.3f} & {100 * r['sbs_strict_best']:.0f}\\% & "
            f"{r['oracle_top3']:.3f} & {r['oracle_all']:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


# ─────────────────────────────────────────────────────────────────────
# 2. Critical-difference diagrams of the Default configurations
# ─────────────────────────────────────────────────────────────────────


def default_cds(d) -> dict[str, dict[str, int]]:
    """Draw the four CD diagrams; return the missing default runs (scored 0)."""
    missing = {}
    for name, df in [
        ("fig_cpd_main_cd_default", d.df_cpd),
        ("fig_cpd_guided_cd_default", d.df_cpd_g),
        ("fig_sd_main_cd_default", d.df_sd),
        ("fig_sd_guided_cd_default", d.df_sd_g),
    ]:
        df = _clean(df)
        n_nan = df[df["score_default"].isna()].groupby("algorithm").size()
        missing[name] = {a: int(v) for a, v in n_nan.items()}
        # _save_cd_directly fills a missing score with 0, as the padding rule does.
        _save_cd_directly(df, score_col="score_default", maximize=True, name=name, width=5.0)
    return missing


# ─────────────────────────────────────────────────────────────────────
# 3. Per-dataset boxplots of the per-series tuned score
# ─────────────────────────────────────────────────────────────────────


def _half_box(ax, pos, data, color, face, width):
    bp = ax.boxplot(
        [data],
        positions=[pos],
        widths=[width],
        patch_artist=True,
        whis=1.5,
        showfliers=True,
        flierprops=dict(
            marker="o", markersize=1.2, markerfacecolor=color, markeredgecolor="none", alpha=0.55
        ),
        medianprops=dict(color="white" if face == color else INK, linewidth=0.8),
        whiskerprops=dict(color=color, linewidth=0.5),
        capprops=dict(color=color, linewidth=0.5),
        boxprops=dict(linewidth=0.5),
    )
    for box in bp["boxes"]:
        box.set_facecolor(face)
        box.set_edgecolor(color)


def _draw_pair(ax, x, algo, ng: pd.Series | None, g: pd.Series | None, width=0.72):
    color = FAMILY_COLOR["SD" if ALGO_TASK.get(algo) == "SD" else "CPD"]
    half = width / 2.0
    if ng is not None and g is not None:
        _half_box(ax, x - half / 2.0, ng, color, _tint(color), half * 0.92)
        _half_box(ax, x + half / 2.0, g, color, color, half * 0.92)
    elif g is not None:
        _half_box(ax, x, g, color, color, width * 0.92)
    elif ng is not None:
        _half_box(ax, x, ng, color, _tint(color), width * 0.92)


def dataset_boxplots(
    df_ng: pd.DataFrame,
    df_g: pd.DataFrame,
    algos: list[str],
    name: str,
    ylabel: str,
    ylim: tuple[float, float],
):
    df_ng, df_g = _clean(df_ng), _clean(df_g)
    with mpl.rc_context(SPLIT_RC):
        fig, axes = plt.subplots(2, 4, figsize=(5.5, 3.4), sharey=True)
        fig.subplots_adjust(left=0.08, right=0.995, bottom=0.19, top=0.89, wspace=0.08, hspace=0.30)
        for ax, ds in zip(axes.flat, DATASETS):
            n_series = (
                df_ng.loc[df_ng["dataset"] == ds, "trial_index"].nunique()
                or df_g.loc[df_g["dataset"] == ds, "trial_index"].nunique()
            )
            for x, algo in enumerate(algos):
                ng = df_ng[(df_ng["algorithm"] == algo) & (df_ng["dataset"] == ds)]["score_best"]
                g = df_g[(df_g["algorithm"] == algo) & (df_g["dataset"] == ds)]["score_best"]
                # Methods that cannot take K: their Guided rows copy the Non-Guided ones.
                if algo in GUIDED_FROM_NON_GUIDED:
                    g = pd.Series(dtype=float)
                _draw_pair(
                    ax,
                    x,
                    algo,
                    ng.to_numpy() if len(ng) else None,
                    g.to_numpy() if len(g) else None,
                )
            ax.set_title(f"{DATASET_DISPLAY[ds]} ({n_series})", fontsize=6.5, pad=2, color=INK)
            ax.set_ylim(*ylim)
            ax.set_yticks(np.round(np.arange(np.ceil(ylim[0] * 5) / 5, 1.0001, 0.2), 1))
            ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.1f"))
            ax.set_xlim(-0.6, len(algos) - 0.4)
            ax.grid(True, axis="y", color=GRID, linewidth=0.5)
            ax.set_axisbelow(True)
            ax.tick_params(axis="x", length=0, pad=2)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.set_xticks(range(len(algos)))
            if ax in axes[1]:
                ax.set_xticklabels(
                    [display_name(a) for a in algos],
                    rotation=55,
                    ha="right",
                    rotation_mode="anchor",
                    fontsize=5.5,
                )
            else:
                ax.set_xticklabels([])
        for ax in axes[:, 0]:
            ax.set_ylabel(ylabel)
        fams = {"SD" if ALGO_TASK.get(a) == "SD" else "CPD" for a in algos}
        fig.legend(
            handles=legend_handles(fams),
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=4,
            frameon=False,
            handlelength=1.2,
            handleheight=0.8,
            columnspacing=1.0,
            labelcolor=INK,
        )
        savefig(fig, name, tight=False)


def _ordered(df_ng: pd.DataFrame, df_g: pd.DataFrame, algos: set[str]) -> list[str]:
    """Methods sorted by their best mean tuned score over the two modes."""
    best = {}
    for df in (_clean(df_ng), _clean(df_g)):
        for a, v in (
            df[df["algorithm"].isin(algos)].groupby("algorithm")["score_best"].mean().items()
        ):
            best[a] = max(best.get(a, -np.inf), v)
    return sorted(algos, key=lambda a: -best[a])


def make_all(d=None):
    if d is None:
        d = load_data()

    rows = oracle_rows(d)
    (OUTPUT_DIR / "table_oracle_selection.tex").write_text(
        "% Auto-generated by analysis/rq1_extra.py\n" + oracle_table(rows)
    )
    for r in rows:
        print(
            f"  {r['task']} {r['mode']}: single best {r['sbs']} {r['sbs_score']:.3f} "
            f"(strict best on {100 * r['sbs_strict_best']:.1f}%), oracle top-3 "
            f"{r['oracle_top3']:.3f}, oracle {r['oracle_all']:.3f}"
        )

    for name, miss in default_cds(d).items():
        if miss:
            print(f"  {name}: default runs missing (scored 0): {miss}")

    # CPD: the top methods of each mode, as in the spider plots; SD: every method.
    cpd = set(_top_algos(_clean(d.df_cpd), top_k=SPIDER_TOP_K)) | set(
        _top_algos(_clean(d.df_cpd_g), top_k=SPIDER_TOP_K)
    )
    sd = set(_clean(d.df_sd)["algorithm"]) | set(_clean(d.df_sd_g)["algorithm"])
    dataset_boxplots(
        d.df_cpd,
        d.df_cpd_g,
        _ordered(d.df_cpd, d.df_cpd_g, cpd),
        "fig_dataset_boxplots_cpd",
        "Tuned BiC per series",
        (0.0, 1.02),
    )
    lo = min(_clean(d.df_sd)["score_best"].min(), _clean(d.df_sd_g)["score_best"].min())
    dataset_boxplots(
        d.df_sd,
        d.df_sd_g,
        _ordered(d.df_sd, d.df_sd_g, sd),
        "fig_dataset_boxplots_sd",
        "Tuned SMS per series",
        (np.floor(lo * 10) / 10, 1.02),
    )
