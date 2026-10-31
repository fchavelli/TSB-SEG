"""Shared settings of the paper analysis: paths, results database, algorithm
names and colours, series domains, figure style and grid summaries.

The results database and the output directory can be changed with
environment variables:

    TSSEG_ANALYSIS_DB   results database (default: results/results.db, rebuilt
                        by scripts/build_results_db.py)
    TSSEG_ANALYSIS_OUT  directory of the figures and tables (default:
                        figures/paper)
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

BENCH_DIR = Path(__file__).resolve().parent.parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

from benchmark_analysis import BenchmarkAnalyzer  # noqa: E402
from mlflow_manager import MLflowBenchmarkManager  # noqa: E402

# ── Paths and results database ───────────────────────────────────────

CONFIG_PATH = BENCH_DIR / "configs" / "benchmark_config_v3.yaml"
DB_PATH = Path(os.environ.get("TSSEG_ANALYSIS_DB", BENCH_DIR.parent / "results" / "results.db"))
OUTPUT_DIR = Path(os.environ.get("TSSEG_ANALYSIS_OUT", BENCH_DIR.parent / "figures" / "paper"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# The 8 datasets of the evaluation; PAMAP2 is the scalability case study.
MAIN_DATASETS = ["actrectut", "has", "mocap", "pump", "skab", "tssb", "usc-had", "utsa"]

# DataFrames of data.load_data(), cached per database directory.
CACHE_DIR = BENCH_DIR / "analysis" / "cache" / DB_PATH.parent.name
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# SMS penalty of every Default SD run, split by error type (Figures 5c and
# 6c), computed from the predicted labels by scripts/sms_error_breakdown.py.
SMS_ERRORS = DB_PATH.parent / "sms_error_breakdown.parquet"

manager = MLflowBenchmarkManager(CONFIG_PATH, f"sqlite:///{DB_PATH}")
analyzer = BenchmarkAnalyzer(manager)

# ── Algorithms: task, category, name and colour ──────────────────────

with open(BENCH_DIR / "typology.json") as f:
    TYPOLOGY = json.load(f)

ALGO_CATEGORY: dict[str, str] = {}
ALGO_TASK: dict[str, str] = {}  # "CPD", "SD" or "baseline"
for cat_name, cat_info in TYPOLOGY["categories"].items():
    for algo in cat_info["algorithms"]:
        ALGO_CATEGORY[algo] = cat_name
        ALGO_TASK[algo] = cat_info["task"]
ALGO_TASK["random"] = "baseline"
for algo, info in TYPOLOGY["algorithms"].items():
    ALGO_CATEGORY.setdefault(algo, f"{info['task']} / {info['subtype']}")

ALGO_DISPLAY: dict[str, str] = {
    algo: info["display"] for algo, info in TYPOLOGY["algorithms"].items()
}

# One fixed colour per method, from Paul Tol's colour-blind safe bright and
# muted schemes, shared by the strip, spider and CD figures.
ALGO_COLORS: dict[str, str] = {
    "amoc": "#DDCC77",
    "autoplait": "#CC3311",
    "binseg": "#AA3377",
    "bocd": "#228833",
    "bottomup": "#44AA99",
    "changefinder": "#66CCEE",
    "clap": "#0077BB",
    "clasp": "#4477AA",
    "dynp": "#882255",
    "e2usd": "#882255",
    "eagglo": "#CCBB44",
    "espresso": "#332288",
    "fluss": "#332288",
    "ggs": "#EE6677",
    "hdp-hsmm": "#DDCC77",
    "hidalgo": "#0077BB",
    "icid": "#228833",
    "igts": "#4477AA",
    "pelt": "#EE7733",
    "prophet": "#999933",
    "random": "#888888",
    "tglad": "#AA3377",
    "ticc": "#AA4499",
    "time2state": "#999933",
    "tire": "#CC3311",
    "tscp2": "#EE7733",
    "window": "#EE6677",
}

# ── Series domains ───────────────────────────────────────────────────

# Datasets whose series all share one domain.
DATASET_DOMAIN_DIRECT = {
    "actrectut": "HAR",
    "has": "HAR",
    "mocap": "HAR",
    "usc-had": "HAR",
    "pump": "Industry",
    "skab": "Industry",
    "pamap2": "HAR",
}

_DOMAIN_REMAP = {
    "biology": "Biology",
    "medical": "Medicine",
    "simulated": "Simulation",
    "industrial": "Industry",
    "HAR": "HAR",
    "IoT": "Other",
    "traffic": "Other",
    "robotic": "Robotics",
    "Calligraphy": "Other",
    "Environment": "Other",
    "archeology": "Other",
    "unknown": "Other",
}

# (dataset, series) -> domain for TSSB and UTSA, whose series are in the row
# order of these files: series 1 is row 0.
SERIES_DOMAIN: dict[tuple[str, str], str] = {}
for _ds in ("tssb", "utsa"):
    _domains = pd.read_csv(BENCH_DIR / "datasets" / f"{_ds}.csv")
    for _idx, _row in _domains.iterrows():
        _domain = _DOMAIN_REMAP.get(str(_row["Domain"]).strip(), "Other")
        SERIES_DOMAIN[(_ds, str(_idx + 1))] = _domain

DOMAIN_ORDER = ["Biology", "Medicine", "Industry", "Robotics", "Simulation", "HAR", "Other"]

# ── Figure style (NeurIPS) ───────────────────────────────────────────

COL_W = 3.25  # column width, in
TEXT_W = 6.75  # text width, in
DPI = 300

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "axes.linewidth": 0.6,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.4,
        "figure.dpi": 120,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "text.usetex": False,
        "mathtext.fontset": "cm",
        # TrueType fonts in PDF, so that hatching renders.
        "pdf.fonttype": 42,
        "hatch.linewidth": 1.5,
    }
)


def savefig(fig, name: str, tight=True):
    """Save a figure as PDF and PNG in OUTPUT_DIR."""
    for ext in ("pdf", "png"):
        path = OUTPUT_DIR / f"{name}.{ext}"
        fig.savefig(path, dpi=DPI, bbox_inches="tight" if tight else None)
    print(f"  → {OUTPUT_DIR / name}.{{pdf,png}}")
    plt.close(fig)


def get_domain(dataset: str, trial_index) -> str:
    """Domain of a (dataset, series) pair."""
    if dataset in DATASET_DOMAIN_DIRECT:
        return DATASET_DOMAIN_DIRECT[dataset]
    ti = str(trial_index) if trial_index is not None else ""
    return SERIES_DOMAIN.get((dataset, ti), "Other")


def display_name(algo: str) -> str:
    """Name of an algorithm in the paper."""
    return ALGO_DISPLAY.get(algo, algo)


def algo_color(algo: str) -> str:
    """Colour of an algorithm."""
    return ALGO_COLORS.get(algo, "#888888")


# ── Grid summaries ───────────────────────────────────────────────────


def select_best_grid_per_dataset(
    df_grid: pd.DataFrame,
    metric: str,
    higher_is_better: bool = True,
    min_coverage: float = 0.9,
) -> pd.DataFrame:
    """Rank the grid configurations of each (algorithm, dataset).

    Configurations are ranked by their mean score over the series of the
    dataset. Those that ran at least ``min_coverage`` of the series run by the
    most complete configuration come first; the others follow.
    """
    required = [metric, "algorithm", "dataset", "trial_index", "parent_run_id"]
    if not all(c in df_grid.columns for c in required):
        missing = [c for c in required if c not in df_grid.columns]
        print(f"Missing columns: {missing}")
        return pd.DataFrame()

    df = df_grid.dropna(subset=[metric, "trial_index"]).copy()

    config_stats = (
        df.groupby(["algorithm", "dataset", "parent_run_id"])
        .agg(config_mean=(metric, "mean"), n_series=("trial_index", "nunique"))
        .reset_index()
    )
    max_series = (
        config_stats.groupby(["algorithm", "dataset"])["n_series"]
        .max()
        .reset_index()
        .rename(columns={"n_series": "max_series"})
    )
    config_stats = config_stats.merge(max_series, on=["algorithm", "dataset"])
    config_stats["is_complete"] = config_stats["n_series"] >= (
        min_coverage * config_stats["max_series"]
    )

    config_stats["_sort"] = config_stats["is_complete"].map({True: 0, False: 1})
    config_stats = config_stats.sort_values(
        ["algorithm", "dataset", "_sort", "config_mean"],
        ascending=[True, True, True, not higher_is_better],
    )
    config_stats["config_rank"] = config_stats.groupby(["algorithm", "dataset"]).cumcount() + 1
    config_stats = config_stats.drop(columns=["_sort", "n_series", "max_series"])
    config_stats = config_stats.rename(columns={"is_complete": "config_complete"})

    return df.merge(
        config_stats[
            [
                "algorithm",
                "dataset",
                "parent_run_id",
                "config_rank",
                "config_mean",
                "config_complete",
            ]
        ],
        on=["algorithm", "dataset", "parent_run_id"],
    )


def aggregate_grid_summaries(
    df_grid_ranked: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    """Tuned, worst-case and median score of every (algorithm, dataset, series).

    The tuned score is that of the best-ranked configuration; worst-case and
    median range over the complete configurations (all of them when none is
    complete), so that a configuration that ran few series does not set the
    worst case. Ties keep one configuration per series.
    """
    dedup_cols = ["algorithm", "dataset", "trial_index"]
    if "config_complete" in df_grid_ranked.columns:
        any_complete = df_grid_ranked.groupby(["algorithm", "dataset"])[
            "config_complete"
        ].transform("any")
        pool = df_grid_ranked[df_grid_ranked["config_complete"] | ~any_complete]
    else:
        pool = df_grid_ranked

    best = df_grid_ranked[df_grid_ranked["config_rank"] == 1].copy()
    best = best.drop_duplicates(subset=dedup_cols, keep="first")
    best = best.rename(columns={metric: "score_best"})

    max_rank = pool.groupby(["algorithm", "dataset"])["config_rank"].max().reset_index()
    max_rank.columns = ["algorithm", "dataset", "max_rank"]
    worst_mask = pool.merge(max_rank, on=["algorithm", "dataset"])
    worst_mask = worst_mask[worst_mask["config_rank"] == worst_mask["max_rank"]]
    worst_agg = (
        worst_mask.groupby(dedup_cols)[metric]
        .first()
        .reset_index()
        .rename(columns={metric: "score_worst"})
    )

    median_agg = (
        pool.groupby(dedup_cols)[metric]
        .median()
        .reset_index()
        .rename(columns={metric: "score_median"})
    )

    result = best[dedup_cols + ["score_best"]].copy()
    result = result.merge(worst_agg, on=dedup_cols, how="left")
    return result.merge(median_agg, on=dedup_cols, how="left")


# ── Metrics and coverage ─────────────────────────────────────────────

CPD_METRICS = [
    "bidirectional_covering_score",
    "f1_score",
    "f1_score_score",
    "covering_score",
    "gaussian_f1_score",
    "execution_time_seconds",
]
SD_METRICS = [
    "state_matching_score",
    "state_matching_score_score",  # SMS logged under this key by some runs
    "adjusted_rand_index_score",
    "weighted_adjusted_rand_index_score",
    "adjusted_mutual_info_score",
    "execution_time_seconds",
]
ALL_METRICS = sorted(set(CPD_METRICS + SD_METRICS))

METRIC_CPD = "bidirectional_covering_score"
METRIC_SD = "state_matching_score"

# Methods missing at most this fraction of the series stay in the main pool,
# their missing series scored 0; ESPRESSO, Guided, misses 123 of 592 (20.8 %).
MAX_GAP_FRACTION = 0.21
_SCORE_COLS = ["score_best", "score_worst", "score_median", "score_default"]

# Methods that time out or run out of memory on whole datasets, or take
# multivariate series only: out of the main pool, analysed on the datasets
# they cover by excluded_algos_analysis.py.
LIMITED_COVERAGE_ALGOS: set[str] = {
    "tglad",
    "tscp2",
    "eagglo",
    "hidalgo",
    "bocd",
}


def _coalesce_sms(df):
    """Merge ``state_matching_score_score`` into ``state_matching_score``."""
    canon = "state_matching_score"
    alias = "state_matching_score_score"
    if alias in df.columns:
        if canon in df.columns:
            df[canon] = df[canon].combine_first(df[alias])
        else:
            df[canon] = df[alias]
        df = df.drop(columns=[alias])
    return df


def apply_gap_tolerance(df, label="CPD"):
    """Score 0 the missing series of the methods within ``MAX_GAP_FRACTION``.

    Three sources of zeros: series without a tuned score, series without a
    row (added), and worst-case or median scores missing where a tuned score
    exists. Methods beyond the tolerance are dropped.

    Returns (df_out, report_list, excluded_set).
    """
    ref_trials = (
        df.groupby(["dataset", "trial_index"]).size().reset_index()[["dataset", "trial_index"]]
    )
    ref_set = set(zip(ref_trials["dataset"], ref_trials["trial_index"]))
    n_ref = len(ref_set)

    _padded_mask = df["score_best"].isna().copy()

    padded_report = []
    excluded_gap = set()

    for algo in sorted(df["algorithm"].unique()):
        sub = df[df["algorithm"] == algo]
        present = set(zip(sub["dataset"], sub["trial_index"]))
        n_nan = int(sub["score_best"].isna().sum())
        missing = ref_set - present
        n_missing = len(missing)
        total_gap = n_nan + n_missing

        if total_gap == 0:
            continue

        if total_gap / n_ref > MAX_GAP_FRACTION:
            excluded_gap.add(algo)
            continue

        ds_detail = {}
        if n_nan > 0:
            nan_ds = sub[sub["score_best"].isna()].groupby("dataset").size()
            for ds_name, cnt in nan_ds.items():
                ds_detail.setdefault(ds_name, {"nan": 0, "miss": 0})
                ds_detail[ds_name]["nan"] = int(cnt)
        for ds_name, _ti in missing:
            ds_detail.setdefault(ds_name, {"nan": 0, "miss": 0})
            ds_detail[ds_name]["miss"] += 1

        padded_report.append((algo, n_nan, n_missing, total_gap, ds_detail))

        if missing:
            new_rows = pd.DataFrame(
                [
                    {
                        "algorithm": algo,
                        "dataset": ds_name,
                        "trial_index": ti,
                        **{c: 0.0 for c in _SCORE_COLS},
                        "domain": get_domain(ds_name, ti),
                        "category": ALGO_CATEGORY.get(algo, ""),
                        "display": ALGO_DISPLAY.get(algo, algo),
                    }
                    for ds_name, ti in missing
                ]
            )
            df = pd.concat([df, new_rows], ignore_index=True)
            _padded_mask = pd.concat(
                [_padded_mask, pd.Series([False] * len(new_rows))],
                ignore_index=True,
            )

    for c in _SCORE_COLS:
        if c in df.columns:
            df.loc[_padded_mask, c] = 0.0

    _has_best = df["score_best"].notna()
    for c in ["score_worst", "score_median"]:
        if c in df.columns:
            _structural_nan = _has_best & df[c].isna()
            n_filled = int(_structural_nan.sum())
            if n_filled:
                df.loc[_structural_nan, c] = 0.0
                print(f"  Filled {n_filled} structural NaN in {c} with 0")

    if excluded_gap:
        df = df[~df["algorithm"].isin(excluded_gap)].copy()

    if padded_report:
        print(f"  Gap tolerance (padded trials→0, max {MAX_GAP_FRACTION:.0%}):")
        for algo, _n_nan, _n_miss, total_gap, ds_detail in padded_report:
            parts = []
            for ds_name in sorted(ds_detail):
                d = ds_detail[ds_name]
                desc = []
                if d["nan"]:
                    desc.append(f"{d['nan']} NaN→0")
                if d["miss"]:
                    desc.append(f"{d['miss']} added")
                parts.append(f"{ds_name}:{'+'.join(desc)}")
            means = {}
            for c in _SCORE_COLS:
                if c in df.columns:
                    vals = df.loc[df["algorithm"] == algo, c]
                    means[c] = vals.mean()
            m_str = ", ".join(f"{c.split('_')[1]}={v:.3f}" for c, v in means.items())
            print(
                f"    {ALGO_DISPLAY.get(algo, algo):15s}  gap={total_gap}/{n_ref}  "
                f"{', '.join(parts)}  ({m_str})"
            )

    if excluded_gap:
        print(f"  Excluded (gap > {MAX_GAP_FRACTION:.0%}): {sorted(excluded_gap)}")

    mean_check = df.groupby("algorithm")[["score_best", "score_worst"]].mean()
    algo_violated = mean_check[mean_check["score_best"] < mean_check["score_worst"]]
    if len(algo_violated):
        for a in algo_violated.index:
            r = mean_check.loc[a]
            print(f"  ⚠ {a}: mean(best)={r['score_best']:.4f} < mean(worst)={r['score_worst']:.4f}")
    else:
        print(f"  ✓ mean(best) ≥ mean(worst) for all {df['algorithm'].nunique()} algos")

    mean_all = df.groupby("algorithm")[_SCORE_COLS].mean()
    for a in mean_all.index:
        r = mean_all.loc[a]
        if r["score_default"] > r["score_best"] + 1e-6:
            print(f"  ⚠ {a}: default={r['score_default']:.4f} > best={r['score_best']:.4f}")

    return df, padded_report, excluded_gap
