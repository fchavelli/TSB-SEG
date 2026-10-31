"""Main entrypoint dispatching to the configured pipeline."""

from __future__ import annotations

import os
import warnings

# Suppress warnings and noisy logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# Abseil (absl) logging commonly used by TensorFlow/JAX
os.environ["GLOG_minloglevel"] = "3"
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", message=".*Filesystem tracking backend.*")

import hydra
from omegaconf import DictConfig, open_dict

from tsbseg.pipelines import get_pipeline_runner
from tsbseg.utils.main_helpers import normalize_supervision_mode, resolve_experiment_cfg


def _resolve_pipeline_name(cfg: DictConfig) -> str:
    """Extract the pipeline name from config (string or sub-dict)."""
    if "pipeline" not in cfg:
        return "default"

    pipeline_value = cfg.pipeline
    if isinstance(pipeline_value, DictConfig):
        name = pipeline_value.get("name")
        if not name:
            return "default"
        return str(name)

    if pipeline_value is None:
        return "default"

    return str(pipeline_value)


def _apply_guided_instance(cfg: DictConfig) -> None:
    """Guided runs: overwrite ``algorithm.instance`` with ``algorithm.guided_instance``.

    For a Non-Guided stopping rule whose default moved after the guided runs: the
    instance holds the new Non-Guided default, ``guided_instance`` the value the
    guided runs used (e.g. ChangeFinder's threshold factor, which also filters the
    peaks among which the top ``n_cps`` are kept). Applied before any pipeline
    logs the configuration, so the logged parameters are the ones used.
    """
    algorithm = cfg.get("algorithm")
    if algorithm is None or not algorithm.get("guided_instance"):
        return
    experiment = resolve_experiment_cfg(cfg)
    mode = normalize_supervision_mode(experiment.get("supervision_mode", "unsupervised"))
    if mode not in {"semi_supervised", "supervised"}:
        return
    with open_dict(cfg):
        for key, value in algorithm.guided_instance.items():
            cfg.algorithm.instance[key] = value


@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint that resolves and runs the requested pipeline."""

    _apply_guided_instance(cfg)
    pipeline_name = _resolve_pipeline_name(cfg).strip().lower()

    try:
        runner = get_pipeline_runner(pipeline_name)
        runner(cfg)
    except Exception as e:
        print(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    main()
