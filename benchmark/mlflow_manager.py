"""
MLflow Benchmark Manager — SQL-first data access layer.

Provides clean, config-driven access to MLflow experiment data stored in
a SQLite backend.  **No dependency on the ``mlflow`` Python package at
runtime** — all queries go through raw SQL on the SQLite file.

This avoids the common pitfall where ``mlflow.set_tracking_uri()`` triggers
Alembic schema migrations on snapshot databases.

Design principles:
- **One source of truth**: ``configs/benchmark_config_v3.yaml`` defines experiment mapping,
  groups, and expected series counts.
- **SQL-first**: all data fetching uses ``latest_metrics`` (not ``metrics``),
  so there is exactly one value per (run, key) — no silent averaging.
- **No silent data loss**: missing data is surfaced, not hidden.
- **Read-only**: the DB connection is opened in read-only mode to prevent
  accidental creation of empty files.
"""

from __future__ import annotations

import fnmatch
import sqlite3
from pathlib import Path

import pandas as pd
import yaml


class MLflowBenchmarkManager:
    """Config-driven manager for accessing MLflow benchmark data.

    Parameters
    ----------
    config_path : str or Path
        Path to ``configs/benchmark_config_v3.yaml``.
    tracking_uri : str, optional
        Tracking URI of the results database (``sqlite:///path/to/db``).
    """

    # ── Construction ──────────────────────────────────────────────────

    def __init__(self, config_path: str | Path, tracking_uri: str | None = None):
        with open(config_path) as f:
            self.config: dict = yaml.safe_load(f)

        if tracking_uri is None:
            raise ValueError("tracking_uri is required (e.g. 'sqlite:///path/to/db')")
        self.tracking_uri = tracking_uri

        # {logical_key: [experiment_name, ...]}
        self._exp_name_map: dict[str, list[str]] = self._resolve_experiment_names()
        # {experiment_name: experiment_id}  (pre-computed for fast lookup)
        self._name_to_id: dict[str, str] = self._build_name_id_map()

    # ── Experiment resolution ─────────────────────────────────────────

    def _resolve_experiment_names(self) -> dict[str, list[str]]:
        """Map each logical key in config to real MLflow experiment names.

        Supports wildcard patterns (``tsseg-experiment-*``).
        Uses direct SQL — no ``mlflow`` import needed.
        """
        conn = self._connect()
        try:
            rows = pd.read_sql_query(
                "SELECT name FROM experiments WHERE lifecycle_stage = 'active'", conn
            )
            all_names = rows["name"].tolist()
        finally:
            conn.close()

        resolved: dict[str, list[str]] = {}
        for key, pattern in self.config.get("experiments", {}).items():
            if "*" in pattern:
                resolved[key] = fnmatch.filter(all_names, pattern)
            elif pattern in all_names:
                resolved[key] = [pattern]
            else:
                resolved[key] = []

        return resolved

    def _build_name_id_map(self) -> dict[str, str]:
        """Build a {name → experiment_id} lookup from the DB."""
        conn = self._connect()
        try:
            rows = pd.read_sql_query("SELECT experiment_id, name FROM experiments", conn)
            return dict(zip(rows["name"], rows["experiment_id"].astype(str)))
        finally:
            conn.close()

    def get_experiment_ids(self, keys: list[str] | None = None) -> list[str]:
        """Return MLflow experiment IDs for the given logical keys.

        Parameters
        ----------
        keys : list of str, optional
            Logical keys (e.g. ``["unsupervised"]``).  ``None`` → all keys.
        """
        if keys is None:
            keys = list(self._exp_name_map.keys())

        target_names: set[str] = set()
        for k in keys:
            target_names.update(self._exp_name_map.get(k, []))

        return [self._name_to_id[n] for n in target_names if n in self._name_to_id]

    def get_group_keys(self, group_name: str) -> list[str]:
        """Return the logical experiment keys for a config group.

        Example: ``get_group_keys("default")`` → ``["unsupervised"]``
        """
        return self.config.get("groups", {}).get(group_name, [])

    def get_all_group_keys(self) -> list[str]:
        """Flatten all logical keys across all groups."""
        keys: list[str] = []
        for k_list in self.config.get("groups", {}).values():
            keys.extend(k_list)
        return list(set(keys))

    # ── SQLite connection ─────────────────────────────────────────────

    def get_db_path(self) -> Path:
        """Extract the file-system path from the tracking URI."""
        prefix = "sqlite:///"
        if not self.tracking_uri.startswith(prefix):
            raise ValueError(f"Tracking URI is not SQLite: {self.tracking_uri}")
        return Path(self.tracking_uri[len(prefix) :])

    def _connect(self) -> sqlite3.Connection:
        """Open a read-only SQLite connection.

        Raises ``FileNotFoundError`` if the DB file does not exist (avoids
        silently creating an empty file).
        """
        db_path = self.get_db_path()
        if not db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")
        return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _in_clause(items: list) -> str:
        """Build a SQL ``IN (...)`` value list.

        Safe for MLflow UUIDs and integer experiment IDs (no user input).
        """
        return ",".join(f"'{v}'" for v in items)

    # ── Fetch parent runs with child stats ────────────────────────────

    def fetch_parent_stats(self, experiment_keys: list[str] | None = None) -> pd.DataFrame:
        """Fetch parent runs with aggregated child counts.

        Returns one row per parent run with columns::

            run_id, experiment_id, start_time, end_time, status,
            algorithm, dataset,
            total_children, children_finished, children_failed

        Notes
        -----
        - *Parent* = run without ``mlflow.parentRunId`` tag.
        - ``algorithm`` / ``dataset`` resolved from ``params.algorithm_name``
          with fallback to ``tags.algorithm_name``.
        """
        exp_ids = self.get_experiment_ids(experiment_keys)
        if not exp_ids:
            return pd.DataFrame()

        ids = self._in_clause(exp_ids)
        conn = self._connect()

        try:
            q_children = f"""
                SELECT
                    t.value                                                 AS parent_run_id,
                    COUNT(*)                                                AS total_children,
                    SUM(CASE WHEN r.status = 'FINISHED' THEN 1 ELSE 0 END) AS children_finished,
                    SUM(CASE WHEN r.status != 'FINISHED' THEN 1 ELSE 0 END) AS children_failed
                FROM runs r
                JOIN tags t ON r.run_uuid = t.run_uuid AND t.key = 'mlflow.parentRunId'
                WHERE r.experiment_id IN ({ids})
                GROUP BY t.value
            """
            df_children = pd.read_sql_query(q_children, conn)

            q_parents = f"""
                SELECT
                    r.run_uuid       AS run_id,
                    r.experiment_id,
                    r.start_time,
                    r.end_time,
                    r.status,
                    COALESCE(pa.value, ta.value) AS algorithm,
                    COALESCE(pd.value, td.value) AS dataset
                FROM runs r
                LEFT JOIN params pa ON r.run_uuid = pa.run_uuid AND pa.key = 'algorithm_name'
                LEFT JOIN tags   ta ON r.run_uuid = ta.run_uuid AND ta.key = 'algorithm_name'
                LEFT JOIN params pd ON r.run_uuid = pd.run_uuid AND pd.key = 'dataset_name'
                LEFT JOIN tags   td ON r.run_uuid = td.run_uuid AND td.key = 'dataset_name'
                WHERE r.experiment_id IN ({ids})
                  AND NOT EXISTS (
                      SELECT 1 FROM tags tc
                      WHERE tc.run_uuid = r.run_uuid AND tc.key = 'mlflow.parentRunId'
                  )
            """
            df_parents = pd.read_sql_query(q_parents, conn)
        finally:
            conn.close()

        if df_parents.empty:
            return pd.DataFrame()

        # Merge child stats
        df = df_parents.merge(df_children, left_on="run_id", right_on="parent_run_id", how="left")
        for col in ("total_children", "children_finished", "children_failed"):
            df[col] = df[col].fillna(0).astype(int)
        df["parent_run_id"] = df["run_id"]

        # Normalise types
        df["start_time"] = pd.to_datetime(df["start_time"], unit="ms", utc=True)
        df["end_time"] = pd.to_datetime(df["end_time"], unit="ms", utc=True)
        df["dataset"] = df["dataset"].str.lower()

        return df

    # ── Fetch child metrics ───────────────────────────────────────────

    def fetch_child_metrics(
        self,
        parent_ids: list[str],
        metric_keys: list[str] | None = None,
    ) -> pd.DataFrame:
        """Fetch latest metric values for *finished* child runs.

        Returns a **long** DataFrame::

            run_id | parent_run_id | start_time | trial_index | metric_key | metric_value

        Uses ``latest_metrics`` (one row per run × key — no duplicates).
        """
        if not parent_ids:
            return pd.DataFrame()

        conn = self._connect()
        CHUNK = 500
        chunks: list[pd.DataFrame] = []

        try:
            for i in range(0, len(parent_ids), CHUNK):
                batch = parent_ids[i : i + CHUNK]
                ids = self._in_clause(batch)

                metric_filter = ""
                if metric_keys:
                    metric_filter = f"AND lm.key IN ({self._in_clause(metric_keys)})"

                q = f"""
                    SELECT
                        r.run_uuid     AS run_id,
                        tp.value       AS parent_run_id,
                        r.start_time,
                        ti.value       AS trial_index,
                        lm.key         AS metric_key,
                        lm.value       AS metric_value
                    FROM runs r
                    JOIN tags tp           ON r.run_uuid = tp.run_uuid  AND tp.key = 'mlflow.parentRunId'
                    JOIN latest_metrics lm ON r.run_uuid = lm.run_uuid {metric_filter}
                    LEFT JOIN tags ti      ON r.run_uuid = ti.run_uuid AND ti.key = 'dataset_trial_index'
                    WHERE tp.value IN ({ids})
                      AND r.status = 'FINISHED'
                """
                chunks.append(pd.read_sql_query(q, conn))
        finally:
            conn.close()

        if not chunks:
            return pd.DataFrame()

        df = pd.concat(chunks, ignore_index=True)
        df["start_time"] = pd.to_datetime(df["start_time"], unit="ms", utc=True)
        return df

    # ── Fetch run params ──────────────────────────────────────────────

    def fetch_run_params(
        self,
        run_ids: list[str],
        exclude_keys: set | None = None,
    ) -> pd.DataFrame:
        """Fetch all parameter key-value pairs for the given runs.

        Returns a DataFrame with columns ``run_id``, ``key``, ``value``.

        Parameters
        ----------
        run_ids : list of str
            Run UUIDs to fetch params for.
        exclude_keys : set of str, optional
            Param keys to exclude (e.g. ``{"algorithm_name", "dataset_name"}``).
        """
        if not run_ids:
            return pd.DataFrame(columns=["run_id", "key", "value"])

        conn = self._connect()
        CHUNK = 500
        chunks: list[pd.DataFrame] = []

        try:
            for i in range(0, len(run_ids), CHUNK):
                batch = run_ids[i : i + CHUNK]
                ids = self._in_clause(batch)
                q = f"""
                    SELECT run_uuid AS run_id, key, value
                    FROM params
                    WHERE run_uuid IN ({ids})
                """
                chunks.append(pd.read_sql_query(q, conn))
        finally:
            conn.close()

        if not chunks:
            return pd.DataFrame(columns=["run_id", "key", "value"])

        df = pd.concat(chunks, ignore_index=True)
        if exclude_keys:
            df = df[~df["key"].isin(exclude_keys)]
        return df
