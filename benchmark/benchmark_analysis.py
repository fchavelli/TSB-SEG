"""
``BenchmarkAnalyzer``: selection of the runs read by the analysis.

On top of ``MLflowBenchmarkManager`` (data access), it adds:

- **Mode assignment** (non-guided / guided / grid_non-guided / grid_guided).
- **Completeness validation** (strict / relaxed).
- **Metric pivoting**, one row per series run.
- **Grid deduplication**: a configuration run more than once counts once.

The experiments, groups and expected series counts live in
``configs/benchmark_config_v3.yaml``. Methods listed as ``partial_algos``
(ESPRESSO, Prophet, TICC, which do not complete every dataset) are kept
even when incomplete, and flagged.
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal

import pandas as pd
from mlflow_manager import MLflowBenchmarkManager


class BenchmarkAnalyzer:
    """Analysis layer for benchmark experiment results.

    Parameters
    ----------
    manager : MLflowBenchmarkManager
        Configured data access manager.
    """

    def __init__(self, manager: MLflowBenchmarkManager):
        self.manager = manager
        self._cfg = manager.config

        # Load from config (with sensible defaults)
        self.expected_series: dict[str, int] = self._cfg.get("expected_series", {})
        self.partial_algos: set[str] = set(self._cfg.get("partial_algos", []))

    # ── Mode assignment ───────────────────────────────────────────────

    def assign_modes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add a ``mode`` column based on config groups.

        Maps each ``experiment_id`` to the group it belongs to
        (``default``, ``guided``, ``grid_default``, ``grid_guided``).
        Runs not in any group get ``mode='other'``.
        """
        if df.empty or "experiment_id" not in df.columns:
            return df

        df = df.copy()
        df["mode"] = "other"

        groups: dict = self._cfg.get("groups", {})
        for group_name, keys in groups.items():
            exp_ids = self.manager.get_experiment_ids(keys)
            if exp_ids:
                exp_ids_str = {str(x) for x in exp_ids}
                mask = df["experiment_id"].astype(str).isin(exp_ids_str)
                df.loc[mask, "mode"] = group_name

        return df

    # ── Parent run fetching (with mode) ───────────────────────────────

    def fetch_parent_stats(self, experiment_keys: list[str] | None = None) -> pd.DataFrame:
        """Fetch parent runs with child stats and mode assignment.

        If *experiment_keys* is ``None``, all keys across all groups are used.
        """
        if experiment_keys is None:
            experiment_keys = self.manager.get_all_group_keys()

        df = self.manager.fetch_parent_stats(experiment_keys)
        if df.empty:
            return df

        df = self.assign_modes(df)
        return df

    # ── Completeness validation ───────────────────────────────────────

    def validate_completeness(
        self,
        df_parents: pd.DataFrame,
        strategy: Literal["strict", "relaxed"] = "strict",
    ) -> pd.DataFrame:
        """Filter and deduplicate parent runs based on completeness.

        Parameters
        ----------
        strategy : {"strict", "relaxed"}
            - ``strict``: keep only parents where
              ``children_finished == expected_series[dataset]``,
              **plus** partial algos regardless of count.
              Deduplicates by keeping the latest parent per
              ``(experiment_id, dataset, algorithm)``.
            - ``relaxed``: keep all parents, without deduplication (grid
              experiments, deduplicated by parameters instead).
        """
        if df_parents.empty:
            return df_parents

        df = df_parents.copy()

        if strategy == "relaxed":
            return df.sort_values("start_time", ascending=False)

        # Strict: classify completeness
        def _is_complete(row: pd.Series) -> bool:
            ds = row.get("dataset", "")
            expected = self.expected_series.get(ds)
            if expected is None:
                return False
            return int(row["children_finished"]) == expected

        df["is_complete"] = df.apply(_is_complete, axis=1)
        is_partial = df["algorithm"].isin(self.partial_algos)

        # Keep if complete OR partial (whitelisted despite incompleteness)
        df = df[df["is_complete"] | is_partial].copy()

        # Flag partial algos explicitly
        df["is_partial"] = df["algorithm"].isin(self.partial_algos) & ~df["is_complete"]

        # Deduplicate: latest parent per (experiment_id, dataset, algorithm)
        df = df.sort_values("start_time", ascending=False).drop_duplicates(
            subset=["experiment_id", "dataset", "algorithm"], keep="first"
        )

        return df

    # ── Metric fetching + pivoting ────────────────────────────────────

    def fetch_metrics(
        self,
        valid_parents: pd.DataFrame,
        metric_keys: list[str] | None = None,
        deduplicate_series: bool = False,
    ) -> pd.DataFrame:
        """Fetch child metrics and merge with parent metadata.

        Returns a **wide** DataFrame with one row per child run and one
        column per metric (e.g. ``covering_score``, ``f1_score_score``).

        Parameters
        ----------
        valid_parents : DataFrame
            Output of ``validate_completeness`` (must have ``run_id``, ``algorithm``,
            ``dataset`` columns).
        metric_keys : list of str, optional
            Restrict to these metric keys.  ``None`` → all metrics.
        deduplicate_series : bool
            If True, keep only the latest child per
            ``(mode, algorithm, dataset, trial_index)``.
        """
        if valid_parents.empty:
            return pd.DataFrame()

        parent_ids = list(valid_parents["run_id"].unique())

        # Fetch long-form metrics
        df_long = self.manager.fetch_child_metrics(parent_ids, metric_keys)
        if df_long.empty:
            return pd.DataFrame()

        # Pivot to wide: one row per child, one column per metric
        index_cols = ["run_id", "parent_run_id"]
        if "start_time" in df_long.columns:
            index_cols.append("start_time")
        if "trial_index" in df_long.columns:
            index_cols.append("trial_index")

        # pivot_table with aggfunc='first' since latest_metrics guarantees
        # one value per (run, key) — this is just a safety net.
        df_wide = df_long.pivot_table(
            index=index_cols,
            columns="metric_key",
            values="metric_value",
            aggfunc="first",
        ).reset_index()
        # Flatten MultiIndex columns
        df_wide.columns.name = None

        # Merge parent metadata
        meta_cols = ["run_id", "experiment_id", "algorithm", "dataset"]
        if "mode" in valid_parents.columns:
            meta_cols.append("mode")
        if "is_partial" in valid_parents.columns:
            meta_cols.append("is_partial")

        parent_meta = valid_parents[meta_cols].rename(columns={"run_id": "parent_run_id"})
        df = df_wide.merge(parent_meta, on="parent_run_id", suffixes=("_child", ""))

        # Deduplication
        if deduplicate_series and "trial_index" in df.columns and "start_time" in df.columns:
            subset = ["algorithm", "dataset", "trial_index"]
            if "mode" in df.columns:
                subset.insert(0, "mode")

            has_index = df["trial_index"].notna()
            df_valid = df[has_index].copy()
            df_noindex = df[~has_index].copy()

            if not df_noindex.empty:
                n = len(df_noindex)
                print(f"  ⚠ {n} children without trial_index — kept but not deduplicated")

            before = len(df_valid)
            df_valid = df_valid.sort_values("start_time", ascending=False).drop_duplicates(
                subset=subset, keep="first"
            )
            if before != len(df_valid):
                print(f"  Deduplication: {before} → {len(df_valid)} children")

            df = pd.concat([df_valid, df_noindex], ignore_index=True)

        return df

    # ── Grid search helpers ───────────────────────────────────────────

    def fetch_grid_metrics(
        self,
        base_mode: str,
        metric_keys: list[str] | None = None,
    ) -> pd.DataFrame:
        """Fetch all grid search child metrics for a base mode.

        Maps ``"default"`` → ``"grid_default"``, ``"guided"`` → ``"grid_guided"``.
        Uses ``strategy='relaxed'`` (no completeness filter on parents),
        then deduplicates parents by parameter fingerprint so that reruns
        of the same config only count once (latest parent kept).
        """
        grid_mode = f"grid_{base_mode}"
        keys = self.manager.get_group_keys(grid_mode)
        if not keys:
            print(f"Grid mode '{grid_mode}' not found in config groups.")
            return pd.DataFrame()

        df_parents = self.fetch_parent_stats(keys)
        df_valid = self.validate_completeness(df_parents, strategy="relaxed")
        if df_valid.empty:
            return pd.DataFrame()

        # Deduplicate grid parents: keep only the latest parent per
        # (algorithm, dataset, param_fingerprint) to eliminate reruns.
        df_valid = self._deduplicate_grid_parents(df_valid)

        return self.fetch_metrics(df_valid, metric_keys=metric_keys, deduplicate_series=False)

    def _deduplicate_grid_parents(
        self,
        df_parents: pd.DataFrame,
        score_metric: str = "bidirectional_covering_score",
    ) -> pd.DataFrame:
        """Remove rerun duplicates from grid parents using param fingerprints.

        Two parents with the same (algorithm, dataset) and identical
        hyperparameter values are considered reruns.  Among reruns the
        parent with the **most finished children** is kept first (maximises
        coverage), then by **highest mean child score**, then by
        **most recent start_time**.
        """
        if df_parents.empty:
            return df_parents

        parent_ids = df_parents["run_id"].tolist()
        df_params = self.manager.fetch_run_params(
            parent_ids, exclude_keys={"algorithm_name", "dataset_name"}
        )

        if df_params.empty:
            return df_parents

        # Build a fingerprint (MD5 hash of sorted param dict) per parent
        def _fingerprint(sub: pd.DataFrame) -> str:
            d = dict(zip(sub["key"], sub["value"]))
            return hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]

        fps = df_params.groupby("run_id").apply(_fingerprint).reset_index(name="_param_fp")

        df = df_parents.merge(fps, on="run_id", how="left")

        # Check if there are actually any duplicates to resolve
        dup_mask = df.duplicated(subset=["algorithm", "dataset", "_param_fp"], keep=False)
        if not dup_mask.any():
            return df_parents

        # Fetch mean child score only for parents involved in reruns
        rerun_pids = df.loc[dup_mask, "run_id"].tolist()
        df_children = self.manager.fetch_child_metrics(rerun_pids, metric_keys=[score_metric])

        if not df_children.empty:
            parent_scores = (
                df_children[df_children["metric_key"] == score_metric]
                .groupby("parent_run_id")["metric_value"]
                .mean()
                .reset_index()
                .rename(columns={"parent_run_id": "run_id", "metric_value": "_mean_score"})
            )
            df = df.merge(parent_scores, on="run_id", how="left")
        else:
            df["_mean_score"] = float("nan")

        # Sort: most children first, then best score, then latest
        before = len(df)
        df = (
            df.sort_values(
                ["children_finished", "_mean_score", "start_time"],
                ascending=[False, False, False],
            )
            .drop_duplicates(subset=["algorithm", "dataset", "_param_fp"], keep="first")
            .drop(columns=["_param_fp", "_mean_score"])
        )

        removed = before - len(df)
        if removed:
            print(f"  Grid parent dedup: {before} → {len(df)} (removed {removed} reruns)")

        return df
