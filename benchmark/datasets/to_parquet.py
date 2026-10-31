#!/usr/bin/env python
"""Convert the benchmark series to Parquet, one file per dataset (TSB-SEG v1.0).

Each file holds exactly the series of the paper, as the data loader returns
them (same channels, same values and dtypes, same labels), in the order of the
benchmark runs (``data_loading/series.csv``). One row per series:

``series``        1-based index, the ``series`` / ``dataset_trial_index`` of the runs
``series_id``     stable name of the series in its source
``source``        loader parameters that select it in the source layout (JSON)
``n_timepoints``, ``n_channels``
``labels``        state label of every time point
``channel_<k>``   values of channel k (null when the series has fewer channels)

The loader reads ``<data_root>/<dataset>.parquet`` when it exists, and the
source layout otherwise. Every converted series is read back through the loader
and compared with the source (values, NaN, dtypes); nothing is written if one
differs.

    python benchmark/datasets/to_parquet.py --out release/data
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from tsbseg.data_loading.loaders import SERIES_FILE, load_dataset

ROOT = Path(__file__).resolve().parents[2]

FORMAT_VERSION = "1.0"


def series_id(dataset: str, p: dict) -> str:
    if dataset == "pamap2":
        return f"subject10{p['subject_number']}"
    if dataset == "usc-had":
        return f"Subject{p['subject_number']}/t{p['target_number']}"
    if dataset == "actrectut":
        return f"subject{p['subject_number']}_walk"
    if dataset == "pump":
        return f"{p['version']}/{Path(p['filename']).stem}"
    if dataset == "has":
        return f"ts_{p['ts_id']}"
    if dataset == "mocap":
        return f"trial_{p['trial']}"
    return p["ts_name"]


def convert(dataset: str, data_root: Path) -> pa.Table:
    series = pd.read_csv(SERIES_FILE)
    series = series[series["dataset"] == dataset].sort_values("series")
    rows, label_dtype = [], None
    for s in series.itertuples():
        p = json.loads(s.source)
        X, y = load_dataset(dataset, data_root=str(data_root), return_X_y=True, **p)
        X, y = np.asarray(X), np.asarray(y)
        if X.dtype != np.float64 or X.ndim != 2:
            raise SystemExit(f"{dataset} {s.series}: X is {X.dtype}, ndim {X.ndim}")
        if label_dtype is None:
            label_dtype = y.dtype
        elif y.dtype != label_dtype:
            raise SystemExit(f"{dataset}: label dtype {y.dtype} != {label_dtype}")
        rows.append((s.series, series_id(dataset, p), s.source, X, y))

    width = max(X.shape[1] for *_, X, _ in rows)
    label_type = pa.from_numpy_dtype(label_dtype)
    columns = {
        "series": pa.array([r[0] for r in rows], pa.int32()),
        "series_id": pa.array([r[1] for r in rows], pa.string()),
        "source": pa.array([r[2] for r in rows], pa.string()),
        "n_timepoints": pa.array([len(r[3]) for r in rows], pa.int32()),
        "n_channels": pa.array([r[3].shape[1] for r in rows], pa.int16()),
        "labels": pa.array([r[4] for r in rows], pa.list_(label_type)),
    }
    for k in range(width):
        columns[f"channel_{k}"] = pa.array(
            [X[:, k] if k < X.shape[1] else None for *_, X, _ in rows],
            pa.list_(pa.float64()),
        )
    meta = {
        "tsb_seg_format": FORMAT_VERSION,
        "dataset": dataset,
        "label_dtype": str(label_dtype),
    }
    return pa.table(columns).replace_schema_metadata(meta)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check(dataset: str, data_root: Path, out: Path) -> None:
    """Read every series back through the loader and compare with the source."""
    Xs, ys = load_dataset(dataset, data_root=str(data_root), return_X_y=True)
    Xp, yp = load_dataset(dataset, data_root=str(out), return_X_y=True)
    if len(Xs) != len(Xp):
        raise SystemExit(f"{dataset}: {len(Xs)} series in the source, {len(Xp)} read back")
    for i, (a, b, u, v) in enumerate(zip(Xs, Xp, ys, yp), 1):
        a, b, u, v = map(np.asarray, (a, b, u, v))
        same = (
            a.dtype == b.dtype
            and u.dtype == v.dtype
            and np.array_equal(a, b, equal_nan=True)
            and np.array_equal(u, v)
        )
        if not same:
            raise SystemExit(f"{dataset} series {i}: read back differs from the source")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("datasets", nargs="*")
    args = parser.parse_args()

    datasets = args.datasets or sorted(pd.read_csv(SERIES_FILE)["dataset"].unique())
    args.out.mkdir(parents=True, exist_ok=True)
    checksums_file = args.out / "checksums.json"
    checksums = json.loads(checksums_file.read_text()) if checksums_file.is_file() else {}
    for ds in datasets:
        table = convert(ds, args.data)
        path = args.out / f"{ds}.parquet"
        tmp = path.with_suffix(".parquet.tmp")
        pq.write_table(table, tmp, compression="zstd", row_group_size=16)
        tmp.replace(path)
        try:
            check(ds, args.data, args.out)
        except SystemExit:
            path.unlink()
            raise
        checksums[path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
        print(f"{ds}: {table.num_rows} series, {path.stat().st_size / 1e6:.1f} MB, checked")
    checksums_file.write_text(json.dumps(checksums, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
