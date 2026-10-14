#!/usr/bin/env python
"""Rebuild, from the published tables, the SQLite database the analysis reads.

The TSB-SEG results are published as Parquet tables. The analysis code
(``benchmark/``) was written against the MLflow tracking database of the
benchmark; this script rebuilds the subset of that database it reads
(``experiments``, ``runs``, ``tags``, ``params``, ``latest_metrics``), with the
original run and experiment identifiers, so that the code runs unchanged::

    python scripts/build_results_db.py --results results
    export TSSEG_ANALYSIS_DB=$PWD/results/results.db

Tables (``results/``):

``runs.parquet``
    One row per evaluated series (an MLflow child run, any status): its
    configuration (``config_id``), experiment, algorithm, dataset, series
    index (1-based), status, error message, timestamps and one column per
    metric.
``run_params.parquet``
    Parameters logged by each series run (long format).
``configs.parquet``
    One row per configuration (an MLflow parent run): algorithm, dataset,
    experiment, number of series run and finished, and the aggregates it
    logged.
``config_params.parquet``
    Hyperparameters of each configuration (long format).
``sms_error_breakdown.parquet``
    SMS penalty of every Default state-detection run, split by error type
    (Figures 5c and 6c).
``manifest.json``
    Experiment names and identifiers, provenance and sha256 of every file.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

# Non-metric columns of runs.parquet and configs.parquet; every other column
# is a metric.
RUN_COLS = [
    "run_id",
    "config_id",
    "experiment",
    "algorithm",
    "dataset",
    "series",
    "run_name",
    "status",
    "lifecycle_stage",
    "error_message",
    "start_time",
    "end_time",
    "duration_s",
]
CONFIG_COLS = [
    "config_id",
    "experiment",
    "algorithm",
    "dataset",
    "run_name",
    "status",
    "lifecycle_stage",
    "start_time",
    "end_time",
    "n_series",
    "n_finished",
]

SCHEMA = """
CREATE TABLE experiments (
    experiment_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    lifecycle_stage TEXT
);
CREATE TABLE runs (
    run_uuid TEXT PRIMARY KEY,
    name TEXT,
    experiment_id INTEGER,
    status TEXT,
    start_time INTEGER,
    end_time INTEGER,
    lifecycle_stage TEXT
);
CREATE TABLE tags (
    key TEXT NOT NULL,
    value TEXT,
    run_uuid TEXT NOT NULL,
    PRIMARY KEY (key, run_uuid)
);
CREATE TABLE params (
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    run_uuid TEXT NOT NULL,
    PRIMARY KEY (key, run_uuid)
);
CREATE TABLE latest_metrics (
    key TEXT NOT NULL,
    value REAL,
    timestamp INTEGER,
    step INTEGER,
    is_nan BOOLEAN,
    run_uuid TEXT NOT NULL,
    PRIMARY KEY (key, run_uuid)
);
"""

INDEXES = """
CREATE INDEX index_runs_experiment_id ON runs (experiment_id);
CREATE INDEX index_tags_run_uuid ON tags (run_uuid);
CREATE INDEX idx_tags_key_value ON tags (key, value);
CREATE INDEX index_params_run_uuid ON params (run_uuid);
CREATE INDEX index_latest_metrics_run_uuid ON latest_metrics (run_uuid);
"""


def to_ms(ts: pd.Series) -> list:
    """UTC timestamps to integer milliseconds (MLflow), ``None`` for NaT."""
    ms = ts.dt.as_unit("ms").array.asi8
    return [None if missing else int(v) for v, missing in zip(ms, ts.isna())]


def tag_rows(df: pd.DataFrame, id_col: str, columns: dict[str, str]) -> pd.DataFrame:
    """Long ``(key, value, run_uuid)`` rows of the non-missing ``columns``."""
    frames = []
    for col, key in columns.items():
        sub = df.loc[df[col].notna(), [id_col, col]]
        frames.append(
            pd.DataFrame({"key": key, "value": sub[col].astype(str), "run_uuid": sub[id_col]})
        )
    return pd.concat(frames, ignore_index=True)


def metric_rows(df: pd.DataFrame, id_col: str, fixed: list[str]) -> pd.DataFrame:
    metrics = [c for c in df.columns if c not in fixed]
    long = df[[id_col, *metrics]].melt(id_vars=id_col, var_name="key").dropna()
    return pd.DataFrame(
        {
            "key": long["key"],
            "value": long["value"].astype(float),
            "timestamp": 0,
            "step": 0,
            "is_nan": np.isnan(long["value"].astype(float)).astype(int),
            "run_uuid": long[id_col],
        }
    )


def build(results: Path, out: Path) -> None:
    manifest = json.loads((results / "manifest.json").read_text())
    exp_id = {key: e["id"] for key, e in manifest["experiments"].items()}
    runs = pd.read_parquet(results / "runs.parquet")
    configs = pd.read_parquet(results / "configs.parquet")
    params = pd.concat(
        [
            pd.read_parquet(results / "config_params.parquet").rename(
                columns={"config_id": "run_id"}
            ),
            pd.read_parquet(results / "run_params.parquet"),
        ],
        ignore_index=True,
    )

    experiments = pd.DataFrame(
        [
            {"experiment_id": e["id"], "name": e["name"], "lifecycle_stage": "active"}
            for e in manifest["experiments"].values()
        ]
    )
    run_table = pd.concat(
        [
            pd.DataFrame(
                {
                    "run_uuid": df[id_col],
                    "name": df["run_name"],
                    "experiment_id": df["experiment"].map(exp_id),
                    "status": df["status"],
                    "start_time": to_ms(df["start_time"]),
                    "end_time": to_ms(df["end_time"]),
                    "lifecycle_stage": df["lifecycle_stage"],
                }
            )
            for df, id_col in ((configs, "config_id"), (runs, "run_id"))
        ],
        ignore_index=True,
    )
    tags = pd.concat(
        [
            tag_rows(
                configs,
                "config_id",
                {
                    "run_name": "mlflow.runName",
                    "algorithm": "algorithm_name",
                    "dataset": "dataset_name",
                },
            ),
            tag_rows(
                runs.assign(series=runs["series"].astype("Int64")),
                "run_id",
                {
                    "config_id": "mlflow.parentRunId",
                    "run_name": "mlflow.runName",
                    "algorithm": "algorithm_name",
                    "dataset": "dataset_name",
                    "series": "dataset_trial_index",
                },
            ),
        ],
        ignore_index=True,
    )
    metrics = pd.concat(
        [
            metric_rows(configs, "config_id", CONFIG_COLS),
            metric_rows(runs, "run_id", RUN_COLS),
        ],
        ignore_index=True,
    )

    if out.exists():
        out.unlink()
    conn = sqlite3.connect(out)
    try:
        conn.executescript(SCHEMA)
        for name, df in (
            ("experiments", experiments),
            ("runs", run_table),
            ("tags", tags),
            (
                "params",
                params.rename(columns={"run_id": "run_uuid"})[["key", "value", "run_uuid"]],
            ),
            ("latest_metrics", metrics),
        ):
            df.to_sql(name, conn, if_exists="append", index=False, chunksize=100_000)
            print(f"  {name}: {len(df)} rows")
        conn.executescript(INDEXES)
        conn.execute("ANALYZE")
        conn.commit()
    finally:
        conn.close()
    print(f"written {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, help="default: <results>/results.db")
    args = parser.parse_args()
    build(args.results, args.out or args.results / "results.db")


if __name__ == "__main__":
    main()
