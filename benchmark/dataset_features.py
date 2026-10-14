"""Features of every benchmark series: ``datasets/dataset_features.csv``.

One row per series, in the order of the benchmark (``series``, from 1, as in
the results): length, number of channels, of change points and of states, and
whether a state recurs after another one (A -> B -> A). The series are read
from ``data/`` (``python scripts/download_data.py``).

    python benchmark/dataset_features.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from tsbseg.data_loading.loaders import SERIES_FILE, load_dataset

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark" / "datasets" / "dataset_features.csv"


def reoccurring(y: np.ndarray) -> bool:
    """Whether a state comes back after another one."""
    segments = y[np.concatenate(([True], y[1:] != y[:-1]))]
    return len(segments) > len(np.unique(segments))


def main() -> None:
    rows = []
    series = pd.read_csv(SERIES_FILE)
    for row in series.itertuples():
        X, y = load_dataset(
            row.dataset, data_root=str(ROOT / "data"), return_X_y=True, **json.loads(row.source)
        )
        X, y = np.asarray(X), np.asarray(y).astype(int)
        rows.append(
            {
                "dataset": row.dataset,
                "series": row.series,
                "length": X.shape[0],
                "dimensions": 1 if X.ndim == 1 else X.shape[1],
                "n_change_points": int(np.count_nonzero(y[1:] != y[:-1])),
                "n_states": len(np.unique(y)),
                "reoccurring": reoccurring(y),
            }
        )
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"{len(rows)} series -> {OUT}")


if __name__ == "__main__":
    main()
