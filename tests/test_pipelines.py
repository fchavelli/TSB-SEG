"""Every configuration composes; a Default run goes through the pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
ALGORITHMS = sorted(p.stem for p in (CONFIGS / "algorithm").glob("*.yaml"))
DATASETS = sorted(p.stem for p in (CONFIGS / "dataset").glob("*.yaml"))
EXPERIMENTS = [
    "unsupervised",
    "semi_supervised",
    "grid_unsupervised_v3",
    "grid_supervised_v3",
]
# Synthetic series instead of a dataset (no data download needed).
MOCK_DATA = [
    "dataset=pump",
    "dataset.loader._target_=tsbseg.utils.test_utils.mock_loader",
    "~dataset.loader.dataset_name",
    "~dataset.loader.return_X_y",
]


@pytest.fixture
def hydra_configs():
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
        yield
    GlobalHydra.instance().clear()


@pytest.mark.parametrize("experiment", EXPERIMENTS)
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_algorithm_config_composes(hydra_configs, algorithm, experiment):
    cfg = compose(
        config_name="config",
        overrides=[f"algorithm={algorithm}", f"experiment={experiment}"],
    )
    assert cfg.algorithm.instance._target_.startswith("tsseg.algorithms.")


@pytest.mark.parametrize("dataset", DATASETS)
def test_dataset_config_composes(hydra_configs, dataset):
    cfg = compose(config_name="config", overrides=[f"dataset={dataset}"])
    assert cfg.dataset.loader._target_ == "tsbseg.data_loading.loaders.load_dataset"


def test_default_run_is_logged(hydra_configs, tmp_path, monkeypatch):
    import mlflow

    from tsbseg.pipelines import run_default_pipeline

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")
    monkeypatch.chdir(tmp_path)
    cfg = compose(
        config_name="config",
        overrides=["algorithm=pelt", "experiment=unsupervised", *MOCK_DATA],
    )
    run_default_pipeline(cfg)
    runs = mlflow.search_runs(experiment_names=["non-guided-full-time-3"])
    assert len(runs) > 0
    assert (runs["status"] == "FINISHED").all()
    assert runs["metrics.bidirectional_covering_score"].notna().any()
