"""Extended-scope pool: the methods of ``LIMITED_COVERAGE_ALGOS`` (appendix).

tGLAD, TS-CP2, EAgglo and BOCD (CPD) and Hidalgo (SD) cannot be compared with
the main pool because they produce no result on at least one whole dataset
(timeouts, out of memory, or, for Hidalgo, multivariate series only). This
module reports, per (method, dataset, mode), the series completed, the
failure causes (out of memory or timeout), the Default and tuned scores and
the median runtime, from every series run of the four experiments.

For grid modes the tuned configuration has the highest mean score on the
dataset, whatever its coverage. Cells never evaluated (e.g. Hidalgo on
univariate datasets) show ``-``.

Outputs: ``table_excluded_algos_{cpd,sd}.tex`` (BiC, SMS),
``table_excluded_algos_cpd_gf1.tex`` (Gaussian F1),
``table_excluded_algos_sd_{ari,wari}.tex`` (ARI, WARI) and the long-form
``table_excluded_algos*.csv``.
"""

from __future__ import annotations

import re
import sqlite3

import numpy as np
import pandas as pd

from analysis.helpers import (
    ALGO_DISPLAY,
    LIMITED_COVERAGE_ALGOS,
    MAIN_DATASETS,
    METRIC_CPD,
    METRIC_SD,
    OUTPUT_DIR,
    TYPOLOGY,
    analyzer,
    manager,
)

# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────

EXPECTED_PER_DS = {
    "actrectut": 2,
    "has": 250,
    "mocap": 9,
    "pump": 120,
    "skab": 34,
    "tssb": 75,
    "usc-had": 70,
    "utsa": 32,
}

# (logical experiment key, mode label, is-grid?)
MODES: list[tuple[str, str, bool]] = [
    ("non_guided", "default_ng", False),
    ("guided", "default_g", False),
    ("grid_non_guided", "grid_ng", True),
    ("grid_guided", "grid_g", True),
]

# Algorithms studied here. Hidalgo logs both BiC (CPD) and SMS (SD)
# metrics, so it appears in both tables (with the appropriate metric)
# even though it is fundamentally an SD method.
CPD_SUPPLEMENT = ["tglad", "tscp2", "eagglo", "bocd", "hidalgo"]
SD_SUPPLEMENT = ["hidalgo"]
SUPPLEMENT_ALGOS = list(dict.fromkeys(CPD_SUPPLEMENT + SD_SUPPLEMENT))
assert set(SUPPLEMENT_ALGOS) <= LIMITED_COVERAGE_ALGOS

METRIC_BY_TASK = {"CPD": METRIC_CPD, "SD": METRIC_SD}
METRIC_DISPLAY = {"CPD": "BiC", "SD": "SMS"}

# Failure-cause regex patterns. Order matters (first match wins, OOM
# before generic).
_OOM_PATTERN = re.compile(
    r"unable to allocate|out of memory|memoryerror|\boom\b|killed",
    re.IGNORECASE,
)
_TIMEOUT_PATTERN = re.compile(
    r"timed[ -]?out|timeout|time limit|signal 9|sigkill",
    re.IGNORECASE,
)


def _algo_task(algo: str) -> str:
    return TYPOLOGY["algorithms"][algo]["task"]


# ─────────────────────────────────────────────────────────────────────
# Direct MLflow SQLite query
# ─────────────────────────────────────────────────────────────────────


def _fetch_all_children(
    experiment_keys: list[str],
    metric_key: str,
    parent_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Pull every child run (any status) for the given experiment keys.

    If ``parent_ids`` is provided, restrict children to those parents
    (used for default modes where ``analyzer.fetch_parent_stats`` has
    already deduplicated re-runs to one parent per (algo, dataset)).

    Returns columns:
        algorithm, dataset, parent_run_id, child_id, trial_index,
        status, error_message, duration_s, score
    """
    exp_ids = manager.get_experiment_ids(experiment_keys)
    if not exp_ids:
        return pd.DataFrame()

    ids = manager._in_clause(exp_ids)
    db_path = manager.get_db_path()

    parent_filter = ""
    if parent_ids is not None:
        if not parent_ids:
            return pd.DataFrame()
        parent_filter = f"AND tp.value IN ({manager._in_clause(parent_ids)})"

    q = f"""
        SELECT
            r.run_uuid                                    AS child_id,
            tp.value                                      AS parent_run_id,
            COALESCE(pa.value, ta.value)                  AS algorithm,
            COALESCE(pd.value, td.value)                  AS dataset,
            ti.value                                      AS trial_index,
            r.status                                      AS status,
            te.value                                      AS error_message,
            (r.end_time - r.start_time) / 1000.0          AS duration_s,
            m.value                                       AS score
        FROM runs r
        JOIN tags  tp ON r.run_uuid = tp.run_uuid AND tp.key = 'mlflow.parentRunId'
        LEFT JOIN params pa ON r.run_uuid = pa.run_uuid AND pa.key = 'algorithm_name'
        LEFT JOIN tags   ta ON r.run_uuid = ta.run_uuid AND ta.key = 'algorithm_name'
        LEFT JOIN params pd ON r.run_uuid = pd.run_uuid AND pd.key = 'dataset_name'
        LEFT JOIN tags   td ON r.run_uuid = td.run_uuid AND td.key = 'dataset_name'
        LEFT JOIN tags   ti ON r.run_uuid = ti.run_uuid AND ti.key = 'dataset_trial_index'
        LEFT JOIN tags   te ON r.run_uuid = te.run_uuid AND te.key = 'error_message'
        LEFT JOIN (
            SELECT run_uuid, value
            FROM latest_metrics
            WHERE key = ?
        ) m ON r.run_uuid = m.run_uuid
        WHERE r.experiment_id IN ({ids})
          {parent_filter}
    """
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        df = pd.read_sql_query(q, conn, params=(metric_key,))

    # For grid modes (no parent filter), still drop any child whose
    # trial_index appears multiple times for the same parent — keep the
    # most recent (highest end_time) — but here we keep them all and
    # rely on _summarise_grid to use child_id-based counting (nunique).
    return df


def _classify_failure(err) -> str:
    """Return one of {'oom', 'timeout', 'other'} for a failed child."""
    if not isinstance(err, str) or not err:
        return "other"
    if _OOM_PATTERN.search(err):
        return "oom"
    if _TIMEOUT_PATTERN.search(err):
        return "timeout"
    return "other"


def _summarise_default(sub: pd.DataFrame) -> dict:
    """Aggregate default-mode children for one (algo, dataset)."""
    out = {
        "n_total_runs": len(sub),
        "n_finished": 0,
        "n_oom": 0,
        "n_timeout": 0,
        "n_other": 0,
        "score": np.nan,
        "median_time_s": np.nan,
    }
    if sub.empty:
        return out

    finished = sub[sub["status"] == "FINISHED"]
    failed = sub[sub["status"] != "FINISHED"]
    # Count distinct series, not raw child rows (some children may share
    # a trial_index across re-runs even after dedup if trial_index is
    # missing in MLflow tags).
    out["n_finished"] = int(finished["trial_index"].dropna().nunique()) if not finished.empty else 0
    if not failed.empty:
        causes = failed["error_message"].map(_classify_failure)
        out["n_oom"] = int((causes == "oom").sum())
        out["n_timeout"] = int((causes == "timeout").sum())
        out["n_other"] = int((causes == "other").sum())
    if not finished.empty:
        scores = finished["score"].dropna()
        if len(scores):
            out["score"] = float(scores.mean())
        times = finished["duration_s"].dropna()
        if len(times):
            out["median_time_s"] = float(times.median())
    return out


def _summarise_grid(sub: pd.DataFrame) -> dict:
    """Aggregate grid-mode children for one (algo, dataset).

    Best config = parent_run_id with the highest mean score over its
    finished children, *regardless* of coverage. Failure counts are
    aggregated over ALL grid attempts (so the reader sees the full
    cost of running this algo on the grid).
    """
    out = {
        "n_total_runs": len(sub),
        "n_finished": 0,
        "n_oom": 0,
        "n_timeout": 0,
        "n_other": 0,
        "score": np.nan,
        "median_time_s": np.nan,
        "best_parent": None,
    }
    if sub.empty:
        return out

    finished_all = sub[sub["status"] == "FINISHED"]
    failed_all = sub[sub["status"] != "FINISHED"]
    if not failed_all.empty:
        causes = failed_all["error_message"].map(_classify_failure)
        out["n_oom"] = int((causes == "oom").sum())
        out["n_timeout"] = int((causes == "timeout").sum())
        out["n_other"] = int((causes == "other").sum())

    if finished_all.empty:
        return out

    per_parent = (
        finished_all.dropna(subset=["score"])
        .groupby("parent_run_id")
        .agg(
            mean_score=("score", "mean"),
            n_fin=("trial_index", lambda s: s.dropna().nunique()),
            med_time=("duration_s", "median"),
        )
    )
    if per_parent.empty:
        out["n_finished"] = int(finished_all["trial_index"].dropna().nunique())
        times = finished_all["duration_s"].dropna()
        if len(times):
            out["median_time_s"] = float(times.median())
        return out

    best = per_parent["mean_score"].idxmax()
    out["best_parent"] = best
    out["score"] = float(per_parent.loc[best, "mean_score"])
    out["n_finished"] = int(per_parent.loc[best, "n_fin"])
    out["median_time_s"] = float(per_parent.loc[best, "med_time"])
    return out


# ─────────────────────────────────────────────────────────────────────
# Build long-form DataFrame
# ─────────────────────────────────────────────────────────────────────


def _get_default_parent_ids(experiment_key: str) -> list[str]:
    """Return deduplicated parent run-ids for a default mode.

    ``analyzer.fetch_parent_stats`` already keeps only the latest parent
    per (experiment_id, dataset, algorithm), so we can safely use its
    ``run_id`` list to suppress re-runs.
    """
    df_parents = analyzer.fetch_parent_stats([experiment_key])
    if df_parents.empty:
        return []
    return df_parents["run_id"].dropna().unique().tolist()


def _dedup_trial_reruns(df: pd.DataFrame) -> pd.DataFrame:
    """For (parent_run_id, trial_index) duplicates keep the latest child.

    MLflow can keep stale child rows when a trial is rerun within the
    same parent. We approximate "latest" by max ``child_id`` lexical
    order as a stable tiebreak (we do not have start_time here); for
    correctness we fall back to keeping the FINISHED row over a FAILED
    one when both exist for the same trial.
    """
    if df.empty or "trial_index" not in df.columns:
        return df
    # Prefer FINISHED rows on conflict.
    df = df.copy()
    df["_finished_first"] = (df["status"] == "FINISHED").astype(int)
    df = df.sort_values(["_finished_first", "child_id"], ascending=[False, False])
    df = df.drop_duplicates(subset=["parent_run_id", "trial_index"], keep="first").drop(
        columns=["_finished_first"]
    )
    return df


def _build_long_df(
    metric_overrides: dict[str, tuple[str, str]] | None = None,
) -> pd.DataFrame:
    r"""Return one row per (algorithm, dataset, mode) for SUPPLEMENT_ALGOS.

    ``metric_overrides`` maps task name ("CPD" / "SD") to ``(metric_key,
    metric_display)``. Defaults to ``METRIC_BY_TASK`` /
    ``METRIC_DISPLAY`` (i.e.\ BiC for CPD and SMS for SD).
    """
    rows: list[dict] = []
    cache: dict[tuple[str, str], pd.DataFrame] = {}
    parent_cache: dict[str, list[str]] = {}

    metric_overrides = metric_overrides or {}

    # Process each (algo, task) pair so that algorithms which produce
    # both CPD and SD outputs (Hidalgo) are evaluated under both metrics
    # and surface in both supplement tables.
    algo_tasks: list[tuple[str, str]] = []
    for algo in CPD_SUPPLEMENT:
        algo_tasks.append((algo, "CPD"))
    for algo in SD_SUPPLEMENT:
        algo_tasks.append((algo, "SD"))

    for algo, task in algo_tasks:
        metric, metric_disp = metric_overrides.get(
            task,
            (METRIC_BY_TASK[task], METRIC_DISPLAY[task]),
        )
        print(f"Processing {algo} ({task}, metric={metric})...")

        for exp_key, mode_label, is_grid in MODES:
            cache_key = (exp_key, metric)
            if cache_key not in cache:
                if is_grid:
                    df_fetched = _fetch_all_children([exp_key], metric)
                else:
                    if exp_key not in parent_cache:
                        parent_cache[exp_key] = _get_default_parent_ids(exp_key)
                    df_fetched = _fetch_all_children(
                        [exp_key], metric, parent_ids=parent_cache[exp_key]
                    )
                df_fetched = _dedup_trial_reruns(df_fetched)
                cache[cache_key] = df_fetched
            df_all = cache[cache_key]
            df_algo = df_all[df_all["algorithm"] == algo]

            for ds in MAIN_DATASETS:
                expected = EXPECTED_PER_DS[ds]
                sub = df_algo[df_algo["dataset"] == ds]
                if is_grid:
                    stats = _summarise_grid(sub)
                else:
                    stats = _summarise_default(sub)
                rows.append(
                    {
                        "algorithm": algo,
                        "task": task,
                        "metric": metric_disp,
                        "dataset": ds,
                        "expected": expected,
                        "mode": mode_label,
                        **{
                            k: stats[k]
                            for k in (
                                "n_total_runs",
                                "n_finished",
                                "n_oom",
                                "n_timeout",
                                "n_other",
                                "score",
                                "median_time_s",
                            )
                        },
                    }
                )
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────
# LaTeX rendering
# ─────────────────────────────────────────────────────────────────────


def _fmt_score(x) -> str:
    return "-" if pd.isna(x) else f"{x:.2f}"


def _fmt_time(x) -> str:
    if pd.isna(x):
        return "-"
    if x < 1:
        return f"{x:.2f}"
    if x < 100:
        return f"{x:.1f}"
    return f"{int(round(x))}"


def _fmt_compl(row: pd.Series) -> str:
    """Render the completion cell for a (algo, dataset, mode) row.

    Returns "-" if the cell was never evaluated, else "n_fin/expected".
    Failure causes (timeout vs OOM) are not reliably stored in MLflow
    for our runs (no ``error_message`` tag), so we omit them here and
    discuss the underlying causes in the prose analysis.
    """
    n_total = int(row["n_total_runs"])
    if n_total == 0:
        return "-"
    n_fin = int(row["n_finished"])
    expected = int(row["expected"])
    return f"{n_fin}/{expected}"


def _render_table(df_task: pd.DataFrame, task: str, metric_display: str) -> str:
    """Render the long-form rows for one task as a LaTeX tabular.

    Wide layout: row = (algorithm, dataset). For each of the 4 modes we
    show 3 sub-columns: Compl., Score, t_med [s].
    """
    mode_order = ["default_ng", "default_g", "grid_ng", "grid_g"]
    mode_titles = {
        "default_ng": "Default, NG",
        "default_g": "Default, G",
        "grid_ng": "Tuned, NG",
        "grid_g": "Tuned, G",
    }

    pivot = df_task.set_index(["algorithm", "dataset", "mode"]).sort_index()

    # 3 cols per mode → ccc. No vertical lines (booktabs only).
    col_spec = "l l c " + " ".join(["c c c"] * len(mode_order))

    lines = [
        "% Auto-generated by analysis/excluded_algos_analysis.py",
        f"% Per-dataset coverage / score / median runtime for the {task} algorithms",
        "% excluded from the main pipeline because of structurally limited coverage.",
        f"% Score = {metric_display}. Compl. = #finished / #expected.",
        "% Tuned = config with highest mean score per dataset (any coverage).",
        "% t_med = median wall-clock per finished trial (s). NG = non-guided, G = guided.",
        "% '-' = configuration never evaluated.",
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
    ]

    # Top header: mode group spans
    top = ["", "", ""]
    for m in mode_order:
        top.append(rf"\multicolumn{{3}}{{c}}{{\textbf{{{mode_titles[m]}}}}}")
    lines.append(" & ".join(top) + r" \\")

    # cmidrules under each mode group (cols start at 4)
    rules = []
    start = 4
    for _ in mode_order:
        rules.append(rf"\cmidrule(lr){{{start}-{start + 2}}}")
        start += 3
    lines.append(" ".join(rules))

    # Sub-header
    sub = [r"\textbf{Algorithm}", r"\textbf{Dataset}", r"\textbf{Series}"]
    for _ in mode_order:
        sub.extend(
            [
                r"Compl.",
                rf"{metric_display}",
                r"$\tilde{t}$ [s]",
            ]
        )
    lines.append(" & ".join(sub) + r" \\")
    lines.append(r"\midrule")

    # Body
    algos_in_df = list(df_task["algorithm"].unique())
    for algo in algos_in_df:
        per_ds = df_task[df_task["algorithm"] == algo]
        ds_with_data = per_ds.groupby("dataset")["n_total_runs"].sum()
        ordered_ds = [
            ds for ds in MAIN_DATASETS if ds in ds_with_data.index and ds_with_data.loc[ds] > 0
        ]
        if not ordered_ds:
            continue

        n_rows = len(ordered_ds)
        for i, ds in enumerate(ordered_ds):
            algo_cell = (
                rf"\multirow{{{n_rows}}}{{*}}{{{ALGO_DISPLAY.get(algo, algo)}}}" if i == 0 else ""
            )
            cells = [algo_cell, ds, str(EXPECTED_PER_DS[ds])]
            for m in mode_order:
                key = (algo, ds, m)
                if key in pivot.index:
                    r = pivot.loc[key]
                    cells.append(_fmt_compl(r))
                    cells.append(_fmt_score(r["score"]))
                    cells.append(_fmt_time(r["median_time_s"]))
                else:
                    cells.extend(["-", "-", "-"])
            lines.append(" & ".join(cells) + r" \\")
        if algo != algos_in_df[-1]:
            lines.append(r"\midrule")

    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────


def _emit_tables(
    df_long: pd.DataFrame,
    suffix: str,
    metric_display_map: dict[str, str],
    only_tasks: set[str] | None = None,
) -> None:
    """Render the CPD and SD tables for one metric flavour.

    ``suffix`` is appended to the file basename (empty string = primary).
    ``metric_display_map`` is the per-task display name used in headers.
    ``only_tasks`` optionally restricts the set of task tables emitted.
    """
    sep = f"_{suffix}" if suffix else ""
    for task, algos, tag in [
        ("CPD", CPD_SUPPLEMENT, "cpd"),
        ("SD", SD_SUPPLEMENT, "sd"),
    ]:
        if only_tasks is not None and task not in only_tasks:
            continue
        df_task = df_long[(df_long["algorithm"].isin(algos)) & (df_long["task"] == task)].copy()
        if df_task.empty:
            print(f"  no data for {task}{sep} — skipping")
            continue
        df_task["algorithm"] = pd.Categorical(df_task["algorithm"], categories=algos, ordered=True)
        df_task = df_task.sort_values(["algorithm", "dataset", "mode"])

        body = _render_table(df_task, task, metric_display_map[task])
        tex_path = OUTPUT_DIR / f"table_excluded_algos_{tag}{sep}.tex"
        tex_path.write_text(body)
        print(f"  wrote {tex_path}")


def main() -> None:
    print("Building the long-form table...")
    df_long = _build_long_df()

    csv_path = OUTPUT_DIR / "table_excluded_algos.csv"
    df_long.to_csv(csv_path, index=False)
    print(f"  wrote {csv_path}")

    _emit_tables(df_long, suffix="", metric_display_map=METRIC_DISPLAY)

    # Secondary metrics: Gaussian F1 for CPD; ARI and WARI for SD.
    secondaries = [
        ("gf1", {"CPD": ("gaussian_f1_score", "gF1")}),
        ("ari", {"SD": ("adjusted_rand_index_score", "ARI")}),
        ("wari", {"SD": ("weighted_adjusted_rand_index_score", "WARI")}),
    ]
    for suffix, override in secondaries:
        print(f"\nBuilding long-form table for secondary metric '{suffix}'...")
        df_sec = _build_long_df(metric_overrides=override)
        df_sec.to_csv(OUTPUT_DIR / f"table_excluded_algos_{suffix}.csv", index=False)
        display_map = {**METRIC_DISPLAY}
        for task, (_, disp) in override.items():
            display_map[task] = disp
        # Only emit tables for the task(s) actually overridden, to avoid
        # re-emitting the primary-metric table under a misleading name.
        only_tasks = set(override.keys())
        _emit_tables(
            df_sec,
            suffix=suffix,
            metric_display_map=display_map,
            only_tasks=only_tasks,
        )
