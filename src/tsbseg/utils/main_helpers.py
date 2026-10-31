"""Utility helpers for the main experiment entrypoint."""

from __future__ import annotations

import logging
import math
import os
import time
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
from omegaconf import DictConfig

log = logging.getLogger(__name__)


def configure_mlflow(cfg: DictConfig) -> None:
    """Use the tracking URI of ``MLFLOW_TRACKING_URI`` (MLflow's default otherwise).

    For SQLite backends, many concurrent writers (one job per series on a
    cluster) need the pragmas of :func:`harden_sqlite_backend`.
    """
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        log.info("[MLflow] Tracking URI: %s", uri)
    harden_sqlite_backend()


# ── SQLite concurrency hardening ──────────────────────────────────────

_SQLITE_PRAGMAS_INSTALLED = False


def harden_sqlite_backend(
    busy_timeout_ms: int = 30_000,
    journal_mode: str = "WAL",
) -> None:
    """Install SQLAlchemy event listeners that set SQLite pragmas.

    Called once from :func:`configure_mlflow`.  The pragmas are applied
    to **every** new raw DBAPI connection that SQLAlchemy opens, which
    means they cover both the MLflow server and any in-process store.

    Parameters
    ----------
    busy_timeout_ms : int
        Milliseconds to wait when the database is locked before raising
        ``sqlite3.OperationalError``.  Default 30 s (SQLite default is
        only 5 s, too low for concurrent jobs).
    journal_mode : str
        ``"WAL"`` enables Write-Ahead Logging, which allows concurrent
        readers while a single writer is active.  This is the single
        most impactful change for SQLite concurrency.
    """
    global _SQLITE_PRAGMAS_INSTALLED
    if _SQLITE_PRAGMAS_INSTALLED:
        return

    try:
        from sqlalchemy import event
        from sqlalchemy.pool import Pool

        @event.listens_for(Pool, "connect")
        def _set_sqlite_pragmas(dbapi_conn, connection_record):
            """Set pragmas on every new SQLite connection."""
            # Only act on SQLite connections (duck-type check)
            module_name = type(dbapi_conn).__module__
            if "sqlite" not in module_name:
                return
            cursor = dbapi_conn.cursor()
            cursor.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)}")
            cursor.execute(f"PRAGMA journal_mode = {journal_mode}")
            cursor.execute("PRAGMA synchronous = NORMAL")
            cursor.close()

        _SQLITE_PRAGMAS_INSTALLED = True
        log.info(
            "[SQLite] Pragmas installed: busy_timeout=%dms, journal_mode=%s, synchronous=NORMAL",
            busy_timeout_ms,
            journal_mode,
        )
    except ImportError:
        log.warning("[SQLite] SQLAlchemy not available — pragmas not set.")


def _as_positive_float(value: Any) -> float | None:
    """Return a positive finite float if conversion succeeds, else ``None``."""

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(numeric) or numeric <= 0:
        return None

    return numeric


def _as_monotonic_deadline(seconds: float | None) -> float | None:
    """Translate a seconds budget into an absolute monotonic deadline."""

    if seconds is None:
        return None

    return time.monotonic() + seconds


def resolve_parent_deadline(
    *,
    timeout_seconds: Any = None,
    timeout_hours: Any = None,
) -> tuple[float | None, float | None]:
    """Resolve parent timeouts from seconds and hour hints.

    Returns a tuple ``(budget_seconds, absolute_deadline)`` where either value may
    be ``None`` if no positive budget could be derived.
    """

    budget_seconds = _as_positive_float(timeout_seconds)
    if budget_seconds is None:
        hours_budget = _as_positive_float(timeout_hours)
        if hours_budget is not None:
            budget_seconds = hours_budget * 3600.0

    if budget_seconds is None:
        return None, None

    return budget_seconds, _as_monotonic_deadline(budget_seconds)


def is_deadline_exceeded(deadline: float | None, *, slack: float = 0.0) -> bool:
    """Return True when the monotonic clock passed the given deadline minus slack."""

    if deadline is None:
        return False

    threshold = deadline - max(float(slack), 0.0)
    return time.monotonic() >= threshold


def labels_to_change_points(labels: np.ndarray) -> list[int]:
    """Convert a sequence of integer labels into change-point indices.

    Always returns a list containing ``0`` and ``len(labels)`` so the resulting
    segmentation spans the whole sequence.
    """

    labels = np.asarray(labels)
    if labels.size == 0:
        return [0]

    if labels.ndim != 1:
        raise ValueError("Expected a 1D array of labels when deriving change points.")

    cps = [0]
    change_locs = np.where(np.diff(labels) != 0)[0] + 1
    cps.extend(change_locs.tolist())
    cps.append(len(labels))
    return sorted(dict.fromkeys(cps))


def normalize_change_points(change_points: Iterable[int], n_timepoints: int) -> list[int]:
    """Sanitize change points ensuring bounds and sentinel markers."""

    if n_timepoints < 0:
        raise ValueError("Number of time points must be non-negative.")

    normalized = {0, int(n_timepoints)}
    for cp in change_points:
        cp_int = int(cp)
        if 0 < cp_int < n_timepoints:
            normalized.add(cp_int)
    return sorted(normalized)


def change_points_to_labels(change_points: Sequence[int], n_timepoints: int) -> np.ndarray:
    """Convert change-point indices into a sequence of integer labels."""

    if n_timepoints <= 0:
        return np.array([], dtype=int)

    cps = normalize_change_points(change_points, n_timepoints)
    labels = np.zeros(n_timepoints, dtype=int)
    current_label = 0
    for start, end in zip(cps[:-1], cps[1:]):
        labels[start:end] = current_label
        current_label += 1
    return labels


CHANGE_POINT_PARAM_ALIASES = {
    "n_cps",
    "n_change_points",
    "num_change_points",
    "n_breakpoints",
    "num_breakpoints",
    "n_cp",
    "n_changepoints",
    "k_max",
    "max_cps",
}

SEGMENT_COUNT_PARAM_ALIASES = {
    "n_segments",
    "num_segments",
    "segments",
    "n_seg",
    "num_seg",
}

STATE_COUNT_PARAM_ALIASES = {
    "n_states",
    "n_regimes",
    "num_states",
    "num_regimes",
    "n_classes",
    "num_classes",
    "n_clusters",
    "num_clusters",
    "n_max_states",
    "alphabet_size",
    "K_states",
}


def count_change_points(labels: np.ndarray) -> int:
    """Return the number of change points implied by the provided labels."""

    labels_arr = np.asarray(labels)
    if labels_arr.size == 0:
        return 0

    cps = labels_to_change_points(labels_arr)
    return max(len(cps) - 2, 0)


def count_unique_states(labels: np.ndarray) -> int:
    """Return the number of unique states (distinct labels) in the provided sequence."""

    labels_arr = np.asarray(labels)
    if labels_arr.size == 0:
        return 0

    return int(np.unique(labels_arr).size)


def count_segments(labels: np.ndarray) -> int:
    """Return the number of contiguous segments implied by the provided labels."""

    labels_arr = np.asarray(labels)
    if labels_arr.size == 0:
        return 0

    cps = labels_to_change_points(labels_arr)
    return max(len(cps) - 1, 0)


def build_supervision_param_overrides(labels: np.ndarray) -> dict[str, object]:
    """Map supervision-sensitive parameter names to the counts of the labels."""

    labels_arr = np.asarray(labels)
    if labels_arr.size == 0:
        return {}

    change_point_count = count_change_points(labels_arr)
    state_count = count_unique_states(labels_arr)
    segment_count = count_segments(labels_arr)

    overrides: dict[str, object] = {}
    for alias in CHANGE_POINT_PARAM_ALIASES:
        overrides[alias] = change_point_count
    for alias in STATE_COUNT_PARAM_ALIASES:
        overrides[alias] = state_count
    for alias in SEGMENT_COUNT_PARAM_ALIASES:
        overrides[alias] = segment_count
    return overrides


def infer_trial_modality(X: np.ndarray) -> str:
    """Infer whether a trial is univariate or multivariate."""

    if not isinstance(X, np.ndarray) or X.size == 0:
        return "unknown"

    if X.ndim == 1:
        return "univariate"
    if X.ndim == 2 and X.shape[1] == 1:
        return "univariate"
    return "multivariate"


def get_algorithm_tags(algorithm: Any) -> dict[str, Any]:
    """Return estimator capability tags if available."""

    if hasattr(algorithm, "get_tags"):
        try:
            tags = algorithm.get_tags()  # type: ignore[call-arg]
            if isinstance(tags, dict):
                return tags
        except Exception:
            pass

    if hasattr(algorithm, "tags"):
        tags = algorithm.tags
        if isinstance(tags, dict):
            return tags

    return {}


def resolve_algorithm_tag(task_from_config: str | None, algo_tags: dict[str, Any]) -> str:
    """Determine algorithm task from config override or estimator tags."""

    if task_from_config is not None and str(task_from_config).lower() != "auto":
        return normalize_task_value(task_from_config)

    tagged_type = algo_tags.get("detector_type")
    return normalize_task_value(tagged_type, default="state")


def resolve_capabilities(
    algo_tags: dict[str, Any],
    default_univariate: bool = True,
    default_multivariate: bool = True,
) -> tuple[bool, bool]:
    """Extract capability flags from estimator tags with fallbacks."""

    supports_univariate = bool(algo_tags.get("capability:univariate", default_univariate))
    supports_multivariate = bool(algo_tags.get("capability:multivariate", default_multivariate))
    return supports_univariate, supports_multivariate


def extract_prediction_components(
    prediction: Any,
    series_length: int,
    *,
    returns_dense: bool = False,
    algorithm_task: str = "state",
) -> tuple[np.ndarray, list[int], dict[str, Any]]:
    """Extract label and change-point representations from a prediction output."""

    artifacts: dict[str, Any] = {}
    labels_pred: np.ndarray | None = None
    change_points_pred: list[int] | None = None
    payload = prediction

    if isinstance(payload, dict):
        artifacts = {
            k: v
            for k, v in payload.items()
            if k not in {"labels", "change_points", "dense_change_points"}
        }
        if "labels" in payload:
            labels_pred = np.asarray(payload["labels"], dtype=int)
        if "change_points" in payload:
            change_points_pred = normalize_change_points(payload["change_points"], series_length)
        if change_points_pred is None and payload.get("dense_change_points") is not None:
            dense_array = np.asarray(payload["dense_change_points"], dtype=float)
            dense_array = dense_array.squeeze()
            if dense_array.ndim == 1 and dense_array.size == series_length:
                cp_indices = np.flatnonzero(np.asarray(dense_array) > 0)
                change_points_pred = normalize_change_points(cp_indices.tolist(), series_length)
        if labels_pred is not None:
            payload = labels_pred
        elif change_points_pred is not None:
            payload = change_points_pred

    if labels_pred is None:
        if isinstance(payload, np.ndarray):
            if payload.ndim == 1 and payload.size == series_length:
                if algorithm_task == "change_point" and returns_dense:
                    cp_indices = np.flatnonzero(payload)
                    change_points_pred = normalize_change_points(cp_indices.tolist(), series_length)
                else:
                    labels_pred = payload.astype(int)
            elif payload.ndim == 2 and payload.shape[0] == series_length:
                labels_pred = np.argmax(payload, axis=1).astype(int)
            else:
                change_points_pred = normalize_change_points(
                    payload.flatten().tolist(), series_length
                )
        elif isinstance(payload, (list, tuple)):
            change_points_pred = normalize_change_points(payload, series_length)
        else:
            raise TypeError(
                "Unsupported prediction type. Expected array, list, tuple or dict with 'labels'/'change_points'."
            )

    if change_points_pred is None and labels_pred is not None:
        change_points_pred = labels_to_change_points(labels_pred)

    if labels_pred is None and change_points_pred is not None:
        labels_pred = change_points_to_labels(change_points_pred, series_length)

    if labels_pred is None or change_points_pred is None:
        raise ValueError("Could not derive both labels and change points from prediction output.")

    return labels_pred, change_points_pred, artifacts


def _flatten_time_series(trial: np.ndarray) -> np.ndarray | None:
    """Reduce an arbitrary-shaped trial to a 1D series for spectral analysis."""

    if trial is None:
        return None

    trial_arr = np.asarray(trial, dtype=float)
    if trial_arr.size < 2:
        return None

    if trial_arr.ndim == 1:
        return trial_arr

    time_axis = int(np.argmax(trial_arr.shape))
    aligned = np.moveaxis(trial_arr, time_axis, 0)
    reshaped = aligned.reshape(aligned.shape[0], -1)
    with np.errstate(invalid="ignore"):
        signal = np.nanmean(reshaped, axis=1)
    return signal


def _estimate_single_series_window(signal: np.ndarray, fallback: int = 10) -> int | None:
    """Estimate window size for a single 1D array using dominant FFT frequency."""

    if signal is None or signal.size < 2:
        return None

    if np.all(np.isnan(signal)):
        return None

    signal = signal - np.nanmean(signal)
    signal = np.nan_to_num(signal)

    if np.allclose(signal, 0.0):
        return None

    spectrum = np.fft.rfft(signal)
    magnitudes = np.abs(spectrum)
    if magnitudes.size <= 1:
        return None

    magnitudes[0] = 0.0
    dominant_idx = int(np.argmax(magnitudes))
    if dominant_idx <= 0:
        return None

    period = int(round(signal.size / dominant_idx))
    if period <= 0:
        return None

    if period % 2 != 0:
        period = period - 1 if period > 1 else period + 1
    if period <= 0:
        return None

    if period >= signal.size * 0.05:
        return fallback

    return period


def estimate_window_size_fft(trials: Sequence[np.ndarray], fallback: int = 10) -> list[int] | None:
    """Estimate a window size per trial based on dominant FFT frequency."""

    estimates: list[int] = []
    for trial in trials:
        signal = _flatten_time_series(trial)
        period = _estimate_single_series_window(signal, fallback=fallback)
        if period is not None:
            estimates.append(period)
        else:
            estimates.append(-1)

    if not estimates:
        return None

    return estimates


_TASK_SYNONYMS = {
    "state": "state",
    "state_detection": "state",
    "segmentation": "state",
    "segmentation_detection": "state",
    "state_detector": "state",
    "states": "state",
    "change_point": "change_point",
    "changepoint": "change_point",
    "change_point_detection": "change_point",
    "changepoint_detection": "change_point",
    "change-point-detection": "change_point",
    "change-point": "change_point",
    "cpd": "change_point",
    "breakpoint_detection": "change_point",
    "both": "both",
    "hybrid": "both",
    "state_and_change_point": "both",
    "state-change": "both",
}


def normalize_task_value(value: str | None, default: str = "state") -> str:
    """Normalize various task aliases to canonical values.

    Parameters
    ----------
    value : Optional[str]
        Raw task string provided by config or tags.
    default : str, default="state"
        Fallback value if the input is missing or unrecognized.
    """

    if value is None:
        return default

    normalized = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    return _TASK_SYNONYMS.get(normalized, default)


# ── Supervision mode and experiment config ─────────────────────────────


def normalize_supervision_mode(raw: Any) -> str:
    """Return a canonical supervision mode string.

    Accepted values are ``"unsupervised"``, ``"semi_supervised"``, and
    ``"supervised"``.  Anything else (including ``None``) falls back to
    ``"unsupervised"``.
    """
    if raw is None:
        return "unsupervised"
    normalised = str(raw).strip().lower().replace(" ", "_").replace("-", "_")
    if normalised in {"semi_supervised", "supervised"}:
        return normalised
    return "unsupervised"


def resolve_experiment_cfg(cfg: DictConfig) -> DictConfig:
    """Return the experiment sub-config regardless of nesting style.

    Supports both ``cfg.experiment.*`` and flattened ``cfg.*`` layouts
    produced by ``@package _global_``.
    """
    if "experiment" in cfg:
        return cfg.experiment
    return cfg


# ── Per-trial deadline budget ─────────────────────────────────────────


def per_trial_deadline(
    *,
    budget_seconds: float | None,
    n_trials: int,
    elapsed: float = 0.0,
    min_seconds: float = 30.0,
    weight: float | None = None,
) -> tuple[float | None, float | None]:
    """Compute a per-trial budget from a global envelope.

    Parameters
    ----------
    budget_seconds : Optional[float]
        Total seconds allocated for the full run.  ``None`` disables the
        time constraint entirely.
    n_trials : int
        Number of remaining trials (must be >= 1).
    elapsed : float
        Seconds already consumed since the run started.
    min_seconds : float
        Minimum per-trial budget (prevents absurdly short deadlines).
    weight : Optional[float]
        Fraction of the remaining budget to allocate to this trial
        (value in ``(0, 1]``).  When ``None``, the remaining budget is
        divided equally among ``n_trials``.

        Typical usage: ``weight = trial_length / sum_of_remaining_lengths``
        so that longer time series receive proportionally more time.

    Returns
    -------
    trial_budget_seconds : Optional[float]
        Per-trial budget, or ``None`` when unconstrained.
    trial_deadline : Optional[float]
        Absolute monotonic deadline for the upcoming trial.
    """
    if budget_seconds is None:
        return None, None

    remaining = max(budget_seconds - elapsed, 0.0)
    if weight is not None and 0.0 < weight <= 1.0:
        per_trial = remaining * weight
    else:
        n = max(n_trials, 1)
        per_trial = remaining / n
    per_trial = max(per_trial, min_seconds)
    # Cap so that a single trial cannot eat more than the remaining budget.
    per_trial = min(per_trial, remaining)
    return per_trial, _as_monotonic_deadline(per_trial)
