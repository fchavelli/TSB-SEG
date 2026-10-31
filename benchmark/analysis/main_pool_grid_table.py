"""Per-(algorithm, dataset) score tables of the main-pool methods (appendix).

One table per task, one row per (method, dataset), with the Default,
worst-case, median and tuned scores of both modes (Non-Guided, Guided) and
the median runtime of the tuned configuration. The CPD table also lists the
SD methods, scored on the change points their labels induce; CLaP is scored
through ClaSP there. Configurations are selected as in ``data.py``
(``select_best_grid_per_dataset``, ``aggregate_grid_summaries``); a cell
with no configuration shows ``-``.

Outputs: ``table_main_pool_{cpd,sd}.tex`` (BiC, SMS),
``table_main_pool_cpd_gf1.tex`` (Gaussian F1),
``table_main_pool_sd_{ari,wari}.tex`` (ARI, WARI) and ``table_main_pool.csv``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.data import load_data
from analysis.helpers import (
    ALGO_DISPLAY,
    LIMITED_COVERAGE_ALGOS,
    MAIN_DATASETS,
    METRIC_CPD,
    METRIC_SD,
    OUTPUT_DIR,
    TYPOLOGY,
    aggregate_grid_summaries,
    select_best_grid_per_dataset,
)

# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────

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

METRIC_BY_TASK = {"CPD": METRIC_CPD, "SD": METRIC_SD}
METRIC_DISPLAY = {"CPD": "BiC", "SD": "SMS"}

# Secondary metrics, one table each: (task, metric key, file suffix, name).
SECONDARY_METRICS: list[tuple[str, str, str, str]] = [
    ("CPD", "gaussian_f1_score", "gf1", r"Gaussian $F_1$"),
    ("SD", "adjusted_rand_index_score", "ari", r"ARI"),
    ("SD", "weighted_adjusted_rand_index_score", "wari", r"WARI"),
]


def _algo_task(algo: str) -> str:
    return TYPOLOGY["algorithms"][algo]["task"]


# Main pool: every method of the typology but the extended-scope ones, in the
# typology order.
MAIN_POOL_ALGOS = [a for a in TYPOLOGY["algorithms"] if a not in LIMITED_COVERAGE_ALGOS]
CPD_MAIN = [a for a in MAIN_POOL_ALGOS if _algo_task(a) == "CPD"]
SD_MAIN = [a for a in MAIN_POOL_ALGOS if _algo_task(a) == "SD"]
# SD methods in the CPD tables: CLaP takes its change points from ClaSP and is
# not reported separately (data._build drops it the same way).
SD_ON_CPD = [a for a in SD_MAIN if a != "clap"]
CPD_GROUPS = [
    (None, CPD_MAIN),
    ("SD methods, scored on the change points induced by their labels", SD_ON_CPD),
]
SD_GROUPS = [(None, SD_MAIN)]


# ─────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────


def _per_cell_default(df_default: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Mean default-mode score per (algorithm, dataset)."""
    if df_default is None or df_default.empty or metric not in df_default.columns:
        return pd.DataFrame(columns=["algorithm", "dataset", "score_default"])
    return (
        df_default.dropna(subset=[metric])
        .groupby(["algorithm", "dataset"], as_index=False)[metric]
        .mean()
        .rename(columns={metric: "score_default"})
    )


def _per_cell_grid(df_grid: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Mean best/worst/median grid scores + median best-grid runtime."""
    cols = ["algorithm", "dataset", "score_best", "score_worst", "score_median", "time_best_median"]
    if df_grid is None or df_grid.empty or metric not in df_grid.columns:
        return pd.DataFrame(columns=cols)

    ranked = select_best_grid_per_dataset(df_grid, metric, higher_is_better=True)
    if ranked.empty:
        return pd.DataFrame(columns=cols)

    summary = aggregate_grid_summaries(ranked, metric)
    if summary.empty:
        return pd.DataFrame(columns=cols)

    cell = summary.groupby(["algorithm", "dataset"], as_index=False)[
        ["score_best", "score_worst", "score_median"]
    ].mean()

    if "execution_time_seconds" in df_grid.columns:
        best_rows = ranked[ranked["config_rank"] == 1]
        times = (
            best_rows.groupby(["algorithm", "dataset"], as_index=False)["execution_time_seconds"]
            .median()
            .rename(columns={"execution_time_seconds": "time_best_median"})
        )
        cell = cell.merge(times, on=["algorithm", "dataset"], how="left")
    else:
        cell["time_best_median"] = np.nan

    return cell


def _build_task_frame(d, task: str, metric: str | None = None) -> pd.DataFrame:
    """Long-form per-(algo, dataset, mode) frame for one task and metric."""
    if metric is None:
        metric = METRIC_BY_TASK[task]
    pool = CPD_MAIN + SD_ON_CPD if task == "CPD" else SD_MAIN

    rows = []
    for mode_key, df_default, df_grid in [
        ("NG", d.df_default_ng, d.df_grid_ng),
        ("G", d.df_default_g, d.df_grid_g),
    ]:
        defaults = _per_cell_default(df_default, metric)
        grids = _per_cell_grid(df_grid, metric)
        cell = grids.merge(defaults, on=["algorithm", "dataset"], how="outer")
        cell = cell[cell["algorithm"].isin(pool)].copy()
        cell = cell[cell["dataset"].isin(MAIN_DATASETS)].copy()
        cell["mode"] = mode_key
        rows.append(cell)

    df = pd.concat(rows, ignore_index=True)
    if df.empty:
        return df
    # Pivot to wide: one row per (algo, dataset), columns per mode/config.
    long = df.melt(
        id_vars=["algorithm", "dataset", "mode"],
        value_vars=[
            "score_default",
            "score_worst",
            "score_median",
            "score_best",
            "time_best_median",
        ],
        var_name="kind",
        value_name="value",
    )
    # Strip BOTH possible prefixes so the time column ends up named
    # "<MODE>_time" rather than "<MODE>_time_best_median" (and matches the
    # lookup keys NG_default/NG_worst/NG_median/NG_best/NG_time below).
    long["short_kind"] = (
        long["kind"]
        .str.replace("score_", "", regex=False)
        .str.replace("time_best_median", "time", regex=False)
    )
    long["col"] = long["mode"] + "_" + long["short_kind"]
    wide = long.pivot_table(
        index=["algorithm", "dataset"],
        columns="col",
        values="value",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None

    # Order rows by algo display name then dataset canonical order.
    wide["_algo_display"] = wide["algorithm"].map(lambda a: ALGO_DISPLAY.get(a, a))
    wide["_ds_order"] = wide["dataset"].map(lambda d: MAIN_DATASETS.index(d))
    wide = wide.sort_values(["_algo_display", "_ds_order"]).reset_index(drop=True)
    wide = wide.drop(columns=["_algo_display", "_ds_order"])
    return wide


# ─────────────────────────────────────────────────────────────────────
# LaTeX rendering
# ─────────────────────────────────────────────────────────────────────


def _fmt_score(v) -> str:
    if v is None or pd.isna(v):
        return "-"
    return f"{v:.2f}"


def _fmt_time(v) -> str:
    if v is None or pd.isna(v):
        return "-"
    if v >= 1000:
        return f"{v / 1000:.1f}k"
    if v >= 100:
        return f"{v:.0f}"
    return f"{v:.1f}"


def _render_table(
    df_task: pd.DataFrame,
    task: str,
    metric_display: str,
    label: str,
    caption: str,
) -> str:
    """longtable: per (algo, dataset) row, 4 grid configs * 2 modes per row.

    The algorithms come in the blocks of ``CPD_GROUPS`` / ``SD_GROUPS``; a
    titled block opens with a full-width italic row.

    Emits a self-contained ``longtable`` so it can break across pages.
    The caption and ``\\label`` are baked into the table; the consumer
    just needs to ``\\input`` the file inside the document body.
    """
    if df_task.empty:
        return "% (empty)"

    header_groups = (
        r"\multicolumn{2}{c}{} & "
        r"\multicolumn{5}{c}{\textbf{Non-Guided}} & "
        r"\multicolumn{5}{c}{\textbf{Guided}} \\"
        r" \cmidrule(lr){3-7} \cmidrule(lr){8-12}"
    )
    header_cols = (
        r" \textbf{Algo} & \textbf{Dataset} "
        r" & Def. & Worst & Med. & Tuned & $\tilde{t}$ [s] "
        r" & Def. & Worst & Med. & Tuned & $\tilde{t}$ [s] \\"
    )

    lines = [
        "% Auto-generated by analysis/main_pool_grid_table.py",
        f"% Main-pool algorithms - metric = {metric_display}.",
        "% Per (algorithm, dataset) cell: mean default / worst-case / median /",
        "% tuned score in both supervision modes, plus median wall-clock of the",
        "% tuned configuration ($\\tilde{t}$, seconds). '-' = not evaluated.",
        r"\begingroup",
        r"\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{0.95}\small",
        r"\begin{longtable}{l l c c c c c c c c c c}",
        rf"\caption{{{caption}}} \label{{{label}}} \\",
        r"\toprule",
        header_groups,
        header_cols,
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{12}{l}{\emph{(continued from previous page)}} \\",
        r"\toprule",
        header_groups,
        header_cols,
        r"\midrule",
        r"\endhead",
        r"\midrule \multicolumn{12}{r}{\emph{(continued on next page)}} \\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]

    present = set(df_task["algorithm"])
    first = True
    for title, members in CPD_GROUPS if task == "CPD" else SD_GROUPS:
        # Rows are already sorted by display name (_build_task_frame).
        block = [a for a in dict.fromkeys(df_task["algorithm"]) if a in members]
        if not block:
            continue
        if not first:
            lines.append(r"\midrule")
        if title:
            lines.append(rf"\multicolumn{{12}}{{l}}{{\emph{{{title}}}}} \\*")
            lines.append(r"\midrule")
        first = False
        for algo in block:
            present.discard(algo)
            _render_algo(lines, df_task, algo, last=algo == block[-1])
    if present:
        raise ValueError(f"algorithms outside every table block: {sorted(present)}")

    lines.extend([r"\end{longtable}", r"\endgroup"])
    return "\n".join(lines)


def _render_algo(lines: list[str], df_task: pd.DataFrame, algo: str, last: bool) -> None:
    """The rows of one algorithm (one per dataset), then a rule unless ``last``."""
    sub = df_task[df_task["algorithm"] == algo]
    n_rows = len(sub)
    for i, (_, r) in enumerate(sub.iterrows()):
        algo_cell = (
            rf"\multirow{{{n_rows}}}{{*}}{{\textbf{{{ALGO_DISPLAY.get(algo, algo)}}}}}"
            if i == 0
            else ""
        )
        ds_cell = DATASET_DISPLAY.get(r["dataset"], r["dataset"])
        ng_def = _fmt_score(r.get("NG_default"))
        ng_w = _fmt_score(r.get("NG_worst"))
        ng_m = _fmt_score(r.get("NG_median"))
        ng_b = _fmt_score(r.get("NG_best"))
        ng_t = _fmt_time(r.get("NG_time"))
        g_def = _fmt_score(r.get("G_default"))
        g_w = _fmt_score(r.get("G_worst"))
        g_m = _fmt_score(r.get("G_median"))
        g_b = _fmt_score(r.get("G_best"))
        g_t = _fmt_time(r.get("G_time"))
        # ``\\*`` forbids a page break inside an algorithm's block, so
        # that its multirow name never lands below the page footer.
        end = r"\\" if i == n_rows - 1 else r"\\*"
        lines.append(
            f"{algo_cell} & {ds_cell} & "
            f"{ng_def} & {ng_w} & {ng_m} & {ng_b} & {ng_t} & "
            f"{g_def} & {g_w} & {g_m} & {g_b} & {g_t} {end}"
        )
    if not last:
        lines.append(r"\midrule")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────


def main(d=None) -> None:
    out_dir = OUTPUT_DIR
    if d is None:
        d = load_data()

    print("Building CPD frame…")
    df_cpd = _build_task_frame(d, "CPD")
    print("Building SD frame…")
    df_sd = _build_task_frame(d, "SD")

    long_rows = []
    for task, df_task in [("CPD", df_cpd), ("SD", df_sd)]:
        df_long = df_task.copy()
        df_long.insert(0, "task", task)
        long_rows.append(df_long)
    csv_path = out_dir / "table_main_pool.csv"
    pd.concat(long_rows, ignore_index=True).to_csv(csv_path, index=False)
    print(f"  CSV written: {csv_path}")

    captions = {
        "CPD": (
            r"\textbf{Main-pool algorithms on CPD: per-(algorithm, dataset) "
            r"default / worst-case / median / tuned BiC, "
            r"in both Non-Guided (NG) and Guided (G) modes.} "
            r"The SD methods (last block) are scored on the change points "
            r"induced by their labels, except CLaP, which labels the "
            r"segments of ClaSP and is not scored separately. "
            r"$\tilde{t}$ is the median wall-clock of the tuned "
            r"configuration in seconds. ``-'' = (mode, dataset) "
            r"pair not evaluated for that algorithm (e.g.\ intrinsically "
            r"Guided methods have no NG entries)."
        ),
        "SD": (
            r"\textbf{Main-pool SD algorithms: per-(algorithm, dataset) "
            r"default / worst-case / median / tuned SMS, "
            r"in both Non-Guided (NG) and Guided (G) modes.} "
            r"Conventions identical to Table~\ref{tab:main_pool_cpd}."
        ),
    }
    labels = {"CPD": "tab:main_pool_cpd", "SD": "tab:main_pool_sd"}

    for task, df_task in [("CPD", df_cpd), ("SD", df_sd)]:
        tex = _render_table(
            df_task,
            task,
            METRIC_DISPLAY[task],
            label=labels[task],
            caption=captions[task],
        )
        tex_path = out_dir / f"table_main_pool_{task.lower()}.tex"
        tex_path.write_text(tex)
        print(f"  Wrote {tex_path.name}")

    # ---- Secondary metrics (Gaussian F1 for CPD, ARI / WARI for SD) ----
    for task, metric_key, suffix, display in SECONDARY_METRICS:
        print(f"Building {task} frame for secondary metric {display}…")
        df_task = _build_task_frame(d, task, metric=metric_key)
        label = f"tab:main_pool_{task.lower()}_{suffix}"
        primary_label = labels[task]
        primary_display = METRIC_DISPLAY[task]
        caption = (
            rf"\textbf{{Main-pool {'algorithms on CPD' if task == 'CPD' else 'SD algorithms'}: per-(algorithm, dataset) "
            rf"default / worst-case / median / tuned {display}, "
            rf"in both Non-Guided (NG) and Guided (G) modes.}} "
            rf"Same conventions as Table~\ref{{{primary_label}}}, with "
            rf"{primary_display} replaced by {display}. The tuned "
            rf"configuration in this table is selected on {display} itself "
            rf"(not on {primary_display}), so each table is self-contained."
        )
        tex = _render_table(
            df_task,
            task,
            display,
            label=label,
            caption=caption,
        )
        tex_path = out_dir / f"table_main_pool_{task.lower()}_{suffix}.tex"
        tex_path.write_text(tex)
        print(f"  Wrote {tex_path.name}")
