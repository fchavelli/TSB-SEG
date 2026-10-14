"""Results of the four experiments, as the figures and tables read them.

``load_data()`` fetches the series runs of the Default and grid experiments,
Non-Guided and Guided, from the results database, applies the selection rules
of the paper, and builds one table per task and mode with the Default, tuned,
worst-case and median score of every (algorithm, dataset, series). The fetched
and built tables are cached as pickles in ``analysis/cache/``.

    d = load_data()                   # cached tables when present
    d = load_data(refresh=True)       # fetch again from the database
    d.df_cpd, d.df_sd, d.df_cpd_g, d.df_sd_g, d.cpd_algos, d.sd_algos
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml

from analysis.helpers import (
    ALGO_CATEGORY,
    ALGO_DISPLAY,
    ALL_METRICS,
    BENCH_DIR,
    CACHE_DIR,
    LIMITED_COVERAGE_ALGOS,
    METRIC_CPD,
    METRIC_SD,
    TYPOLOGY,
    _coalesce_sms,
    aggregate_grid_summaries,
    analyzer,
    apply_gap_tolerance,
    get_domain,
    select_best_grid_per_dataset,
)

_RAW_CACHE_NAMES = ("df_default_ng", "df_default_g", "df_grid_ng", "df_grid_g")
_BUILT_CACHE_NAMES = ("df_cpd", "df_sd", "df_cpd_g", "df_sd_g")
ALL_CACHE_NAMES = _RAW_CACHE_NAMES + _BUILT_CACHE_NAMES


def _cache_path(name: str):
    return CACHE_DIR / f"{name}.pkl"


def _load_pickle(name: str):
    p = _cache_path(name)
    return pd.read_pickle(p) if p.exists() else None


def _save_pickle(name: str, df: pd.DataFrame) -> None:
    df.to_pickle(_cache_path(name))


# ── Fetches ──────────────────────────────────────────────────────────


def _fetch_default(experiments_key: str, label: str) -> pd.DataFrame:
    print(f"Fetching {label} default parents...")
    df_parents = analyzer.fetch_parent_stats([experiments_key])
    df_valid = analyzer.validate_completeness(df_parents, strategy="strict")
    print(
        f"  Valid parents: {len(df_valid)} "
        f"({df_valid['algorithm'].nunique()} algos, {df_valid['dataset'].nunique()} datasets)"
    )
    print(f"Fetching {label} default metrics...")
    df = analyzer.fetch_metrics(df_valid, metric_keys=ALL_METRICS, deduplicate_series=True)
    df = _coalesce_sms(df)
    print(f"  Children: {len(df)}")
    return df


def _fetch_grid(mode: str, label: str) -> pd.DataFrame:
    print(f"Fetching grid {label} metrics...")
    df = analyzer.fetch_grid_metrics(mode, metric_keys=ALL_METRICS)
    df = _coalesce_sms(df)
    print(f"  Grid children: {len(df)} ({df['algorithm'].nunique()} algos)")
    return df


# ── Selection rules ──────────────────────────────────────────────────


def _drop_cells(df_grid: pd.DataFrame, label: str, algos, rule, why: str) -> pd.DataFrame:
    """Drop the grid cells of ``algos`` whose parameters satisfy ``rule``."""
    parent_ids = list(
        df_grid.loc[df_grid["algorithm"].isin(algos), "parent_run_id"].dropna().unique()
    )
    if not parent_ids:
        return df_grid
    df_p = analyzer.manager.fetch_run_params(parent_ids)
    if df_p.empty:
        return df_grid
    pivoted = df_p.pivot(index="run_id", columns="key", values="value").reset_index()
    bad_parents = rule(pivoted)
    if bad_parents:
        print(f"  [patch] {label}: removing {len(bad_parents)} parent configs ({why})")
        return df_grid[~df_grid["parent_run_id"].isin(bad_parents)].copy()
    return df_grid


def _drop_invalid_windows(df_grid: pd.DataFrame, label: str) -> pd.DataFrame:
    """E2USD and Time2State cells with a step longer than the window (64, 100)."""

    def rule(p):
        if "grid_window_size" not in p or "grid_step" not in p:
            return []
        return p[(p["grid_window_size"] == "64") & (p["grid_step"] == "100")]["run_id"].tolist()

    return _drop_cells(df_grid, label, ["e2usd", "time2state"], rule, "window 64, step 100")


def _drop_tglad_test_runs(df_grid: pd.DataFrame, label: str) -> pd.DataFrame:
    """tGLAD test cells with 10 training epochs."""

    def rule(p):
        if "grid_epochs" not in p:
            return []
        return p[p["grid_epochs"] == "10"]["run_id"].tolist()

    return _drop_cells(df_grid, label, ["tglad"], rule, "tGLAD, 10 epochs")


# Methods that take no number of change points or states: AMOC returns one
# change point, PELT stops on its penalty, the AutoPlait binary has no such
# parameter. Their Guided runs are their Non-Guided runs.
GUIDED_FROM_NON_GUIDED = ("amoc", "pelt", "autoplait")


def guided_from_non_guided(
    df_g: pd.DataFrame,
    df_ng: pd.DataFrame,
    label: str,
    algos: tuple[str, ...] = GUIDED_FROM_NON_GUIDED,
) -> pd.DataFrame:
    """Replace the guided rows of ``algos`` with their non-guided copies."""
    if df_ng is None or "algorithm" not in df_ng:
        return df_g
    for algo in algos:
        ng = df_ng[df_ng["algorithm"] == algo]
        if ng.empty:
            continue
        df_g = df_g[df_g["algorithm"] != algo].copy()
        cols = [c for c in df_g.columns if c in ng.columns]
        df_g = pd.concat([df_g, ng[cols].copy()], ignore_index=True)
        print(f"  [patch] {label}: replaced {algo} rows with {len(ng)} NG copies")
    return df_g


# Methods rerun with a corrected detector in the grid experiments only: their
# Default runs are those of the Default configuration of
# configs/algorithm/<algo>.yaml in the grid. TIRE, ChangeFinder and FLUSS were
# rerun in Non-Guided mode only (new selection rule without K), so their Guided
# Default runs are those of the Default experiment.
DEFAULT_FROM_GRID = (
    "igts",
    "hdp-hsmm",
    "pelt",
    "binseg",
    "bottomup",
    "window",
    "dynp",
    "prophet",
    "amoc",
    "autoplait",
)
DEFAULT_FROM_GRID_NG = ("tire", "changefinder", "fluss")


def _default_from_grid_algos(mode: str) -> tuple[str, ...]:
    return DEFAULT_FROM_GRID + (DEFAULT_FROM_GRID_NG if mode == "non-guided" else ())


_ALGO_CONFIG_DIR = BENCH_DIR.parent / "configs" / "algorithm"


def _default_config(algo: str) -> tuple[dict, list[str]]:
    """``instance`` of the algorithm's yaml and the names of its tuned parameters."""
    with open(_ALGO_CONFIG_DIR / f"{algo}.yaml") as f:
        cfg = yaml.safe_load(f)
    tuned = [k for axis in cfg.get("tunable_parameters") or [] for k in axis]
    return dict(cfg["instance"]), tuned


def _same_param(logged, value) -> bool:
    """A logged MLflow param (a string, or NaN when absent) against a yaml value."""
    if not isinstance(logged, str):
        return False
    if value is None or isinstance(value, bool):
        return logged == str(value)
    if isinstance(value, int | float):
        try:
            return float(logged) == float(value)
        except ValueError:
            return False
    return logged == str(value)


def _default_cell_parents(df_grid: pd.DataFrame, algo: str, label: str) -> set:
    """``parent_run_id`` of the default-configuration cell, one per dataset.

    The swept axes select the cell. In guided mode the count parameter
    (``k_max``, ``n_max_states``) is replaced per series by the ground truth
    and logged at its default, so it does not tell cells apart. Every
    ``algo_*`` parameter of the selected cells must then equal the yaml, so
    that a cell matching on the swept axes only is never taken.
    """
    instance, tuned = _default_config(algo)
    g = df_grid[df_grid["algorithm"] == algo]
    p = analyzer.manager.fetch_run_params(list(g["parent_run_id"].dropna().unique()))
    p = p.pivot(index="run_id", columns="key", values="value")
    swept = [k for k in tuned if f"grid_{k}" in p and p[f"grid_{k}"].nunique() > 1]
    keep = np.ones(len(p), dtype=bool)
    for k in swept:
        keep &= p[f"grid_{k}"].map(lambda v, k=k: _same_param(v, instance[k])).to_numpy()
    hit = p.index[keep]

    expected = {("algo_target" if k == "_target_" else f"algo_{k}"): v for k, v in instance.items()}
    differ = sorted(
        {
            k
            for pid in hit
            for k, v in expected.items()
            if not _same_param(p.at[pid, k] if k in p else None, v)
        }
    )
    if differ:
        raise ValueError(
            f"{label}: the {algo} cells with the default {swept} differ from "
            f"configs/algorithm/{algo}.yaml on {differ}"
        )

    per_ds = g[g["parent_run_id"].isin(hit)].groupby("dataset")["parent_run_id"].nunique()
    missing = sorted(set(g["dataset"]) - set(per_ds.index))
    duplicated = per_ds[per_ds > 1].to_dict()
    if missing or duplicated:
        raise ValueError(
            f"{label}: {algo} default cell missing for {missing}, duplicated for {duplicated}"
        )
    return set(hit)


def default_from_grid(
    df_default: pd.DataFrame, df_grid: pd.DataFrame, mode: str, label: str
) -> pd.DataFrame:
    """Replace the default rows of ``DEFAULT_FROM_GRID`` by their default grid cell.

    ``mode`` is the ``mode`` value of the default frame (``"non-guided"`` or
    ``"guided"``); ``DEFAULT_FROM_GRID_NG`` applies to ``"non-guided"`` only.
    """
    if df_default is None or "algorithm" not in df_default:
        return df_default
    algos = _default_from_grid_algos(mode)
    has_grid = df_grid is not None and "algorithm" in df_grid
    out = [df_default[~df_default["algorithm"].isin(algos)]]
    for algo in algos:
        if not has_grid or not (df_grid["algorithm"] == algo).any():
            if (df_default["algorithm"] == algo).any():
                raise ValueError(f"{label}: no {algo} grid to read its default from")
            continue
        rows = df_grid[df_grid["parent_run_id"].isin(_default_cell_parents(df_grid, algo, label))]
        rows = rows[[c for c in df_default.columns if c in rows.columns]].copy()
        rows["mode"] = mode
        if "is_partial" in df_default.columns:
            rows["is_partial"] = False
        print(
            f"  [patch] {label}: {algo} default read from its default grid cell "
            f"({len(rows)} series, {rows['dataset'].nunique()} datasets)"
        )
        out.append(rows)
    return pd.concat(out, ignore_index=True)


# ── Tables per task and mode ─────────────────────────────────────────


def _build(
    df_grid: pd.DataFrame,
    df_default: pd.DataFrame,
    algos: set,
    metric: str,
    label: str,
    cpd: bool,
) -> pd.DataFrame:
    """Default, tuned, worst-case and median score per (algorithm, dataset, series).

    On CPD, methods missing a whole dataset are dropped, and CLaP is scored
    through ClaSP, which segments its series.
    """
    print(f"Ranking grid configs for {label}...")
    ranked = select_best_grid_per_dataset(df_grid, metric, higher_is_better=True)
    if not ranked.empty:
        grid_summary = aggregate_grid_summaries(ranked, metric)
        print(
            f"  Grid {label} summaries: {len(grid_summary)} rows "
            f"({grid_summary['algorithm'].nunique()} algos)"
        )
    else:
        grid_summary = pd.DataFrame()
        print(f"  ⚠ No grid {label} data")

    default = df_default[["algorithm", "dataset", "trial_index", metric]].copy()
    default = default.rename(columns={metric: "score_default"})

    if not grid_summary.empty:
        df = grid_summary.merge(default, on=["algorithm", "dataset", "trial_index"], how="outer")
    else:
        df = default.copy()
        df[["score_best", "score_worst", "score_median"]] = np.nan

    df["domain"] = df.apply(lambda r: get_domain(r["dataset"], r["trial_index"]), axis=1)
    df["category"] = df["algorithm"].map(ALGO_CATEGORY)
    df["display"] = df["algorithm"].map(ALGO_DISPLAY)

    df = df[df["algorithm"].isin(algos)].copy()
    df = df[df["dataset"] != "pamap2"].copy()
    df = df[~df["algorithm"].isin(LIMITED_COVERAGE_ALGOS)].copy()

    if cpd:
        n_datasets = df["dataset"].nunique()
        algo_ds_count = df.groupby("algorithm")["dataset"].nunique()
        incomplete = set(algo_ds_count[algo_ds_count < n_datasets].index)
        if incomplete:
            print(f"  Dropping algos missing entire datasets: {sorted(incomplete)}")
        df = df[df["algorithm"].isin(set(algo_ds_count[algo_ds_count == n_datasets].index))]
        df = df[df["algorithm"] != "clap"].copy()

    df, _, _ = apply_gap_tolerance(df, label)
    print(
        f"\n{label}: {len(df)} rows, {df['algorithm'].nunique()} algos, "
        f"{df['dataset'].nunique()} datasets"
    )
    return df


def _build_algo_sets():
    """Algorithms scored on CPD (every method) and on SD (the SD methods)."""
    cpd_algos: set = set()
    sd_algos: set = set()
    for cat_info in TYPOLOGY["categories"].values():
        cpd_algos.update(cat_info["algorithms"])
        if cat_info["task"] == "SD":
            sd_algos.update(cat_info["algorithms"])
    cpd_algos.add("random")
    sd_algos.add("random")
    return cpd_algos, sd_algos


def load_data(refresh: bool = False, refresh_built: bool = False) -> SimpleNamespace:
    """Return a namespace with all DataFrames and algorithm sets.

    Parameters
    ----------
    refresh
        Fetch the series runs from the database again instead of the cache.
    refresh_built
        Rebuild df_cpd / df_sd / df_cpd_g / df_sd_g even if cached.
    """
    cpd_algos, sd_algos = _build_algo_sets()

    cache: dict = {n: None for n in ALL_CACHE_NAMES}
    if not refresh:
        for n in _RAW_CACHE_NAMES:
            cache[n] = _load_pickle(n)
            if cache[n] is not None:
                print(f"  [cache] loaded {n}.pkl ({len(cache[n])} rows)")

    if cache["df_default_ng"] is None:
        cache["df_default_ng"] = _fetch_default("non_guided", "non-guided")
        _save_pickle("df_default_ng", cache["df_default_ng"])
    if cache["df_default_g"] is None:
        cache["df_default_g"] = _fetch_default("guided", "guided")
        _save_pickle("df_default_g", cache["df_default_g"])
    if cache["df_grid_ng"] is None:
        cache["df_grid_ng"] = _fetch_grid("non-guided", "non-guided")
        _save_pickle("df_grid_ng", cache["df_grid_ng"])
    if cache["df_grid_g"] is None:
        cache["df_grid_g"] = _fetch_grid("guided", "guided")
        _save_pickle("df_grid_g", cache["df_grid_g"])

    # Selection rules, applied on every load: the cache mirrors the database.
    for key, label in (("df_grid_ng", "grid_ng"), ("df_grid_g", "grid_g")):
        cache[key] = _drop_invalid_windows(cache[key], label)
    for key, label in (("df_grid_ng", "grid_ng"), ("df_grid_g", "grid_g")):
        cache[key] = _drop_tglad_test_runs(cache[key], label)
    cache["df_default_g"] = guided_from_non_guided(
        cache["df_default_g"], cache["df_default_ng"], "default_g"
    )
    cache["df_grid_g"] = guided_from_non_guided(cache["df_grid_g"], cache["df_grid_ng"], "grid_g")
    cache["df_default_ng"] = default_from_grid(
        cache["df_default_ng"], cache["df_grid_ng"], "non-guided", "default_ng"
    )
    cache["df_default_g"] = default_from_grid(
        cache["df_default_g"], cache["df_grid_g"], "guided", "default_g"
    )

    if not (refresh or refresh_built):
        for n in _BUILT_CACHE_NAMES:
            cache[n] = _load_pickle(n)
            if cache[n] is not None:
                print(f"  [cache] loaded {n}.pkl ({len(cache[n])} rows)")

    builds = {
        "df_cpd": ("df_grid_ng", "df_default_ng", cpd_algos, METRIC_CPD, "CPD", True),
        "df_sd": ("df_grid_ng", "df_default_ng", sd_algos, METRIC_SD, "SD", False),
        "df_cpd_g": ("df_grid_g", "df_default_g", cpd_algos, METRIC_CPD, "CPD-g", True),
        "df_sd_g": ("df_grid_g", "df_default_g", sd_algos, METRIC_SD, "SD-g", False),
    }
    for name, (grid, default, algos, metric, label, cpd) in builds.items():
        if cache[name] is None:
            cache[name] = _build(cache[grid], cache[default], algos, metric, label, cpd)
            _save_pickle(name, cache[name])

    return SimpleNamespace(cpd_algos=cpd_algos, sd_algos=sd_algos, **cache)
