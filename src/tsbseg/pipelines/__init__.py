"""Experiment pipelines of the TSB-SEG benchmark."""

from collections.abc import Callable

from omegaconf import DictConfig

from .default_pipeline import (
    ExperimentResult,
    RunContext,
    run_default_pipeline,
)
from .grid_pipeline import run_grid_pipeline

# "default_v2" and "grid_v2" are the pipeline names that the runs of the paper
# log (configs/pipeline/default.yaml, configs/experiment/grid_*_v3.yaml).
PIPELINE_REGISTRY: dict[str, Callable[[DictConfig], None]] = {
    "default": run_default_pipeline,
    "default_v2": run_default_pipeline,
    "grid": run_grid_pipeline,
    "grid_v2": run_grid_pipeline,
}


def get_pipeline_runner(name: str) -> Callable[[DictConfig], None]:
    """Return the callable associated with the requested pipeline name."""

    normalized = str(name).strip().lower()
    if normalized not in PIPELINE_REGISTRY:
        available = ", ".join(sorted(PIPELINE_REGISTRY))
        raise KeyError(f"Unknown pipeline '{name}'. Available pipelines: {available}.")
    return PIPELINE_REGISTRY[normalized]


__all__ = [
    "PIPELINE_REGISTRY",
    "ExperimentResult",
    "RunContext",
    "get_pipeline_runner",
    "run_default_pipeline",
    "run_grid_pipeline",
]
