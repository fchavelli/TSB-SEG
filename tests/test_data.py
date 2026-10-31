"""The registry covers every dataset; downloaded datasets hold the benchmark series."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tsbseg.data_loading.loaders import SERIES_FILE, load_dataset

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SERIES = pd.read_csv(SERIES_FILE)
DATASETS = sorted(SERIES["dataset"].unique())


def test_registry_covers_every_dataset():
    registry = json.loads((DATA / "registry.json").read_text())
    assert set(DATASETS) <= set(registry)
    for name, entry in registry.items():
        assert entry["license"], name
        files = entry.get("files", {})
        digests = [f["sha256"] if isinstance(f, dict) else f for f in files.values()]
        digests += [entry["sha256"]] if "sha256" in entry else []
        assert digests and all(len(d) == 64 for d in digests), name


@pytest.mark.parametrize("dataset", DATASETS)
def test_downloaded_dataset_matches_series(dataset):
    present = (DATA / f"{dataset}.parquet").is_file() or (DATA / dataset).is_dir()
    if not present:
        pytest.skip(f"{dataset} not downloaded")
    X, y = load_dataset(dataset, data_root=str(DATA), return_X_y=True)
    expected = SERIES[SERIES["dataset"] == dataset].sort_values("series")
    assert len(X) == len(expected)
    for x, labels, row in zip(X, y, expected.itertuples()):
        x, labels = np.asarray(x), np.asarray(labels)
        assert x.shape == (row.length, row.channels)
        assert np.count_nonzero(np.diff(labels)) == row.change_points
