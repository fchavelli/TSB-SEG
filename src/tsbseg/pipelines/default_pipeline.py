"""Default pipeline: one configuration of one algorithm on every series of a dataset.

*  One level of nesting: a parent run (the configuration) and one child run
   per series (``nested=True``).
*  Explicit context (:class:`RunContext`), no reliance on the implicit
   ``mlflow.active_run()`` stack.
*  Structured return (:class:`ExperimentResult`): the caller decides where
   to log, so that the grid pipeline and this one never log a parameter
   twice.
*  A timed-out series returns ``ExperimentResult(timed_out=True, ...)``
   instead of raising; the caller decides how to propagate it.
"""

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any

import hydra
import mlflow
import numpy as np
import pandas as pd
from mlflow.data import from_pandas as mlflow_from_pandas
from omegaconf import DictConfig, OmegaConf, open_dict

from tsbseg.utils.main_helpers import (
    build_supervision_param_overrides,
    configure_mlflow,
    count_change_points,
    count_unique_states,
    estimate_window_size_fft,
    extract_prediction_components,
    get_algorithm_tags,
    infer_trial_modality,
    is_deadline_exceeded,
    labels_to_change_points,
    normalize_supervision_mode,
    per_trial_deadline,
    resolve_algorithm_tag,
    resolve_capabilities,
    resolve_experiment_cfg,
    resolve_parent_deadline,
)

# ── Styled logging helpers ────────────────────────────────────────────

_BOLD = "\033[1m"
_DIM = "\033[2m"
_GREEN = "\033[32m"
_CYAN = "\033[36m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_RESET = "\033[0m"
_CHECK = "✓"
_CROSS = "✗"
_ARROW = "→"
_BULLET = "•"


def _header(title: str, run_id: str = "") -> None:
    width = 72
    print(f"\n{_BOLD}{'━' * width}{_RESET}")
    print(f"{_BOLD}  {title}{_RESET}")
    if run_id:
        print(f"{_DIM}  MLflow Run ID: {run_id}{_RESET}")
    print(f"{_BOLD}{'━' * width}{_RESET}")


def _footer(title: str, duration: float) -> None:
    width = 72
    mins, secs = divmod(duration, 60)
    time_str = f"{int(mins)}m {secs:.1f}s" if mins else f"{secs:.1f}s"
    print(f"{_DIM}{'─' * width}{_RESET}")
    print(f"{_DIM}  {_CHECK} {title} completed in {time_str}{_RESET}\n")


def _log_step(msg: str) -> None:
    print(f"  {_CYAN}{_ARROW}{_RESET} {msg}")


def _log_info(msg: str) -> None:
    print(f"  {_DIM}{_BULLET} {msg}{_RESET}")


def _log_warn(msg: str) -> None:
    print(f"  {_YELLOW}⚠ {msg}{_RESET}")


def _log_error(msg: str) -> None:
    print(f"  {_RED}{_CROSS} {msg}{_RESET}")


def _log_success(msg: str) -> None:
    print(f"  {_GREEN}{_CHECK} {msg}{_RESET}")


def _format_metrics_table(
    metrics: dict[str, float],
    *,
    title: str = "Metrics",
    max_rows: int | None = None,
) -> str:
    if not metrics:
        return f"  {_DIM}(no metrics){_RESET}"
    items = sorted(metrics.items())
    if max_rows is not None and len(items) > max_rows:
        items = items[:max_rows]
    name_width = max(len(k) for k, _ in items)
    lines = [f"  {_BOLD}{title}{_RESET}"]
    for name, value in items:
        bar = _metric_bar(value)
        lines.append(f"    {name:<{name_width}}  {value:>8.4f}  {bar}")
    return "\n".join(lines)


def _metric_bar(value: float, width: int = 16) -> str:
    clamped = max(0.0, min(1.0, value))
    filled = int(round(clamped * width))
    return f"{_DIM}{'█' * filled}{'░' * (width - filled)}{_RESET}"


# ── Exception ─────────────────────────────────────────────────────────


class TimeoutException(Exception):
    """Raised when a trial or run exceeds the configured wall time."""


# ── Data classes ──────────────────────────────────────────────────────


@dataclass
class RunContext:
    """Explicit context threaded through the call stack.

    Replaces the previous reliance on ``mlflow.active_run()`` so that
    every function knows *exactly* which run it is operating in and what
    parameters / tags have already been logged at the parent level.
    """

    parent_run_id: str
    experiment_id: str
    supervision_mode: str
    algorithm_task: str
    is_supervised: bool
    returns_dense: bool
    supports_semi: bool
    allowed_metric_tasks: set[str] = field(default_factory=set)
    timeout_seconds: float | None = None
    deadline: float | None = None
    proportional_timeout: bool = False
    extra_params: dict[str, Any] | None = None
    extra_tags: dict[str, Any] | None = None


@dataclass
class ExperimentResult:
    """Structured return value from :func:`run_single_experiment`.

    The caller is responsible for logging ``metrics`` and setting
    tags / status on the appropriate MLflow run — this function never
    logs metrics at the parent level itself.
    """

    metrics: dict[str, float] = field(default_factory=dict)
    algorithm: object = None
    timed_out: bool = False
    skipped: bool = False
    trials_completed: int = 0
    trials_total: int = 0


# ── Internal helpers ──────────────────────────────────────────────────


def _resolve_modality_for_trial(
    dataset_modality_cfg: str,
    inferred_trials: list[str],
    idx: int,
) -> str:
    configured = dataset_modality_cfg
    if configured in {"univariate", "multivariate"}:
        return configured
    if configured in {"mixed", "infer", "auto"}:
        return inferred_trials[idx]
    _log_warn(f"Unknown dataset modality '{configured}'. Falling back to inference.")
    return inferred_trials[idx]


def _log_dataset_overview(
    dataset_name: str,
    overview_rows: list[dict[str, float]],
) -> None:
    if not overview_rows:
        return
    overview_df = pd.DataFrame(overview_rows)
    numeric_cols = ["trial_index", "n_timepoints", "n_channels", "label_cardinality"]
    overview_df[numeric_cols] = overview_df[numeric_cols].astype(float)
    try:
        dataset_ds = mlflow_from_pandas(overview_df, name=dataset_name)
        mlflow.log_input(dataset_ds, context="dataset")
    except Exception as exc:
        _log_warn(f"Failed to log dataset to MLflow: {exc}")


def _prepare_trial_data(
    single_X: np.ndarray,
    single_y_true: np.ndarray,
    cfg: DictConfig,
) -> np.ndarray:
    """Apply optional preprocessing and return the (possibly transformed) X."""
    if "preprocessing" not in cfg or cfg.preprocessing is None:
        return single_X

    _log_info("Preprocessing...")
    preprocessor = hydra.utils.instantiate(cfg.preprocessing)

    if np.isnan(single_X).any() or np.isinf(single_X).any():
        _log_warn("NaN/Inf detected — applying linear interpolation.")
        single_X = single_X.copy()
        single_X[np.isinf(single_X)] = np.nan
        df = pd.DataFrame(single_X)
        df.interpolate(method="linear", axis=0, inplace=True, limit_direction="both")
        if df.isnull().values.any():
            _log_warn("NaNs persist after interpolation; filling with 0.")
            df.fillna(0, inplace=True)
        single_X = df.to_numpy()

    single_X_3d = single_X.T.reshape(1, single_X.shape[1], single_X.shape[0])
    transformed = preprocessor.fit_transform(single_X_3d)
    return transformed.squeeze(axis=0).T


def _instantiate_trial_algorithm(
    cfg: DictConfig,
    *,
    trial_window: int | None,
    supervision_overrides: dict[str, int],
    is_supervised_mode: bool,
):
    """Build and return a fresh algorithm instance with per-trial config."""
    algorithm_cfg_container = OmegaConf.to_container(
        cfg.algorithm.instance,
        resolve=True,
        throw_on_missing=True,
    )
    trial_algorithm_cfg = OmegaConf.create(algorithm_cfg_container)

    with open_dict(trial_algorithm_cfg):
        # Window override
        if trial_window is not None and "window_size" in trial_algorithm_cfg:
            trial_algorithm_cfg.window_size = trial_window
        elif trial_window is not None:
            _log_warn("window_size override skipped (parameter not declared).")

        # Supervision overrides
        if supervision_overrides:
            applied: dict[str, object] = {}
            for pname, pval in supervision_overrides.items():
                if pname in trial_algorithm_cfg:
                    # Most overrides are counts, but some carry change-point
                    # POSITIONS (oracle ablations): coercing those to int would
                    # raise on a list.
                    value = [int(v) for v in pval] if isinstance(pval, (list, tuple)) else int(pval)
                    trial_algorithm_cfg[pname] = value
                    applied[pname] = value
            if applied:
                items_str = ", ".join(
                    f"{n}={len(v)} positions" if isinstance(v, list) else f"{n}={v}"
                    for n, v in sorted(applied.items(), key=lambda kv: kv[0])
                )
                _log_info(f"Supervision overrides: {items_str}")
                # Log the size of list-valued overrides, not the list: MLflow
                # caps parameter length and a full change-point list would blow it.
                mlflow.log_params(
                    {
                        f"trial_supervision_override_{k}": (len(v) if isinstance(v, list) else v)
                        for k, v in applied.items()
                    }
                )
            else:
                _log_warn("Supervision overrides did not match any algorithm parameters.")

    return hydra.utils.instantiate(trial_algorithm_cfg), trial_algorithm_cfg


# ── Single trial execution ────────────────────────────────────────────


def _run_single_trial(
    *,
    cfg: DictConfig,
    ctx: RunContext,
    trial_idx: int,
    single_X: np.ndarray,
    single_y_true: np.ndarray,
    trial_modality: str,
    trial_window: int | None,
    trial_deadline: float | None,
) -> dict[str, float]:
    """Execute fit → predict → evaluate for one trial.

    Returns the metric dict (may be empty on failure).
    Raises ``TimeoutException`` if the per-trial deadline is exceeded.

    **Requires** an active nested MLflow run (asserted at entry).
    """
    # ── Guard: must be inside an active run ────────────────────────────
    active = mlflow.active_run()
    if active is None:
        raise RuntimeError("_run_single_trial must be called within an active MLflow run context.")

    def _check_budget(stage: str) -> None:
        if is_deadline_exceeded(trial_deadline):
            _log_warn(f"Trial {trial_idx + 1}: deadline reached during {stage}.")
            mlflow.set_tag("status", "timeout")
            raise TimeoutException(
                f"Per-trial deadline reached while {stage} (trial {trial_idx + 1})."
            )

    # ── Tags / params (trial-level only — no parent-level duplication) ─
    mlflow.set_tag("algorithm_name", cfg.algorithm.name)
    mlflow.set_tag("modality", trial_modality)
    mlflow.set_tag("dataset_name", cfg.dataset.name)
    mlflow.set_tag("dataset_trial_index", trial_idx + 1)
    mlflow.set_tag("algorithm_task", ctx.algorithm_task)
    mlflow.set_tag("status", "running")
    mlflow.set_tag("supervision_mode", ctx.supervision_mode)
    mlflow.log_param("algorithm_name", cfg.algorithm.name)
    mlflow.log_param("dataset_name", cfg.dataset.name)
    mlflow.log_param("modality", trial_modality)
    mlflow.log_param("trial_index", trial_idx + 1)

    use_labels = ctx.is_supervised and ctx.supports_semi
    mlflow.log_param("algorithm_semi_supervised", use_labels)

    if ctx.extra_params:
        mlflow.log_params({k: str(v) for k, v in ctx.extra_params.items()})
    if ctx.extra_tags:
        for k, v in ctx.extra_tags.items():
            mlflow.set_tag(k, str(v))

    trial_start_time = time.monotonic()
    try:
        trial_X_arr = np.asarray(single_X)
        if trial_X_arr.ndim == 1:
            trial_X_arr = trial_X_arr.reshape(-1, 1)

        # Log trial-level dataset info
        trial_info = pd.DataFrame(
            [
                {
                    "dataset": cfg.dataset.name,
                    "trial_index": float(trial_idx + 1),
                    "modality": trial_modality,
                    "n_timepoints": float(trial_X_arr.shape[0]),
                    "n_channels": float(trial_X_arr.shape[1]),
                    "label_cardinality": float(np.unique(np.asarray(single_y_true)).size),
                }
            ]
        )
        try:
            ds = mlflow_from_pandas(trial_info, name=f"{cfg.dataset.name}_trial_{trial_idx + 1}")
            mlflow.log_input(ds, context="trial")
        except Exception as exc:
            _log_warn(f"Failed to log trial dataset: {exc}")

        # Preprocessing
        single_X = _prepare_trial_data(single_X, single_y_true, cfg)

        # Algorithm instantiation
        _check_budget("algorithm setup")
        supervision_overrides = (
            build_supervision_param_overrides(single_y_true) if ctx.is_supervised else {}
        )
        algorithm, _ = _instantiate_trial_algorithm(
            cfg,
            trial_window=trial_window,
            supervision_overrides=supervision_overrides,
            is_supervised_mode=ctx.is_supervised,
        )

        if trial_window is not None:
            mlflow.log_param("trial_window_size", trial_window)
        if ctx.is_supervised:
            mlflow.log_param("trial_supervision_change_points", count_change_points(single_y_true))
            mlflow.log_param("trial_supervision_states", count_unique_states(single_y_true))

        # Fit
        _check_budget("algorithm fitting")
        fit_label = (
            "semi-supervised"
            if use_labels
            else "unsupervised — algo lacks semi-supervised"
            if ctx.is_supervised
            else "unsupervised"
        )
        _log_info(f"Fitting ({fit_label})...")
        algorithm.fit(single_X, axis=0)

        # Predict
        _check_budget("prediction")
        y_pred_raw = algorithm.predict(single_X, axis=0)
        seq_length = single_X.shape[0]

        try:
            y_pred, change_points_pred, prediction_artifacts = extract_prediction_components(
                y_pred_raw,
                seq_length,
                returns_dense=ctx.returns_dense,
                algorithm_task=ctx.algorithm_task,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to normalise prediction for trial {trial_idx + 1}: {exc}"
            ) from exc

        ground_truth_cps = labels_to_change_points(single_y_true)

        # Evaluate
        _check_budget("metric evaluation")
        _log_info("Evaluating metrics...")
        metrics: dict[str, float] = {}
        artifacts: dict[str, Any] = {
            "predicted_labels": y_pred,
            "predicted_change_points": np.asarray(change_points_pred, dtype=int),
            "ground_truth_change_points": np.asarray(ground_truth_cps, dtype=int),
        }
        for extra_k, extra_v in prediction_artifacts.items():
            artifacts[f"prediction_{extra_k}"] = extra_v

        for metric_name, metric_conf in cfg.metric.items():
            if not isinstance(metric_conf, DictConfig) or "_target_" not in metric_conf:
                continue
            metric_task = str(metric_conf.get("task", "state")).lower()
            # Symmetric filtering — each task gets *only* its own metrics
            if metric_task not in ctx.allowed_metric_tasks:
                continue

            compute_params = OmegaConf.to_container(
                metric_conf.get("compute_params", {}),
                resolve=True,
            )
            if compute_params is None:
                compute_params = {}
            instantiate_conf = {
                k: v for k, v in metric_conf.items() if k not in {"compute_params", "task"}
            }
            metric = hydra.utils.instantiate(instantiate_conf, **compute_params)
            result = metric.compute(single_y_true, y_pred)
            for key, value in result.items():
                # Avoid redundant suffixes (e.g. f1_score + score → f1_score)
                if metric_name.endswith(f"_{key}"):
                    full_key = metric_name
                else:
                    full_key = f"{metric_name}_{key}"
                if isinstance(value, (int, float)):
                    metrics[full_key] = value
                elif key != "segmentation_prediction":
                    artifacts[full_key] = value

        mlflow.log_metrics(metrics)

        # Artifact persistence
        with tempfile.TemporaryDirectory() as tmpdir:
            for name, data in artifacts.items():
                if isinstance(data, (list, tuple)):
                    data = np.asarray(data)
                if isinstance(data, np.ndarray):
                    np.save(os.path.join(tmpdir, f"{name}.npy"), data)
            if os.listdir(tmpdir):
                mlflow.log_artifacts(tmpdir)

        print(_format_metrics_table(metrics, title=f"Trial {trial_idx + 1}"))
        mlflow.set_tag("status", "completed")
        return metrics

    except TimeoutException:
        mlflow.set_tag("status", "timeout")
        raise
    except Exception as exc:
        mlflow.set_tag("status", "failed")
        mlflow.log_param("error_message", str(exc))
        raise
    finally:
        duration = float(max(time.monotonic() - trial_start_time, 0.0))
        mlflow.log_metric("execution_time_seconds", duration)


# ── Trial loop ────────────────────────────────────────────────────────


def _run_trials(
    cfg: DictConfig,
    ctx: RunContext,
    filtered: list[tuple[int, np.ndarray, np.ndarray, str, int | None]],
    trial_lengths: list[int],
    base_run_name: str,
) -> ExperimentResult:
    """Iterate over trials, each in a **nested** MLflow run.

    Returns an :class:`ExperimentResult` — never raises
    ``TimeoutException`` itself; instead it sets ``timed_out=True``.
    """
    n_total = len(filtered)
    all_metrics: list[dict[str, float]] = []
    last_algorithm = None
    run_start = time.monotonic()
    timed_out = False

    for position, (trial_idx, sX, sy, t_mod, t_win) in enumerate(filtered):
        # ── Check global budget before starting ──────────────────────
        if is_deadline_exceeded(ctx.deadline) or (
            ctx.timeout_seconds is not None
            and (time.monotonic() - run_start) >= ctx.timeout_seconds
        ):
            _log_warn("Global budget exhausted — skipping remaining trials.")
            timed_out = True
            break

        # ── Compute per-trial deadline ────────────────────────────────
        if ctx.proportional_timeout:
            # Proportional: budget weighted by series length
            elapsed = time.monotonic() - run_start
            remaining_trials = n_total - position
            remaining_lengths_sum = sum(trial_lengths[position:])
            weight = (
                trial_lengths[position] / remaining_lengths_sum
                if remaining_lengths_sum > 0
                else 1.0 / max(remaining_trials, 1)
            )
            trial_budget, trial_dl = per_trial_deadline(
                budget_seconds=ctx.timeout_seconds,
                n_trials=remaining_trials,
                elapsed=elapsed,
                weight=weight,
            )
            if trial_budget is not None:
                _log_info(
                    f"Trial {trial_idx + 1}: budget {trial_budget:.0f}s "
                    f"(len={trial_lengths[position]}, {remaining_trials} remaining)"
                )
        else:
            # Default: each trial gets the full remaining global budget
            trial_dl = None
            if ctx.timeout_seconds is not None:
                global_remaining = max(ctx.timeout_seconds - (time.monotonic() - run_start), 0.0)
                trial_dl = time.monotonic() + global_remaining

        # The parent deadline also applies (the tighter wins).
        if ctx.deadline is not None:
            parent_remaining = max(ctx.deadline - time.monotonic(), 0.0)
            if trial_dl is None or parent_remaining < (trial_dl - time.monotonic()):
                trial_dl = time.monotonic() + parent_remaining

        run_name = f"{base_run_name}_{trial_idx + 1}"
        with mlflow.start_run(run_name=run_name, nested=True):
            try:
                metrics = _run_single_trial(
                    cfg=cfg,
                    ctx=ctx,
                    trial_idx=trial_idx,
                    single_X=sX,
                    single_y_true=sy,
                    trial_modality=t_mod,
                    trial_window=t_win,
                    trial_deadline=trial_dl,
                )
                all_metrics.append(metrics)
            except TimeoutException:
                _log_warn(
                    f"Trial {trial_idx + 1} timed out — breaking to aggregate partial results."
                )
                timed_out = True
                break
            except Exception as exc:
                _log_error(f"Trial {trial_idx + 1} failed: {exc}")
                raise

    # ── Aggregate (even partial) ──────────────────────────────────────
    final_metrics: dict[str, float] = {}
    if all_metrics:
        all_keys = set().union(*(m.keys() for m in all_metrics))
        for key in all_keys:
            vals = [m[key] for m in all_metrics if key in m]
            if vals:
                final_metrics[key] = float(np.mean(vals))
        final_metrics["n_trials_completed"] = float(len(all_metrics))
        final_metrics["n_trials_total"] = float(n_total)

    if final_metrics:
        suffix = " (partial)" if timed_out else ""
        print(_format_metrics_table(final_metrics, title=f"Aggregated{suffix}"))

    return ExperimentResult(
        metrics=final_metrics,
        algorithm=last_algorithm,
        timed_out=timed_out,
        trials_completed=len(all_metrics),
        trials_total=n_total,
    )


# ── Main experiment function ──────────────────────────────────────────


def run_single_experiment(
    cfg: DictConfig,
    *,
    parent_run_id: str | None = None,
    extra_params: dict[str, Any] | None = None,
    extra_tags: dict[str, Any] | None = None,
    run_suffix: str = "",
    fallback_window: int = 10,
    deadline: float | None = None,
    timeout_seconds: float | None = None,
    proportional_timeout: bool = False,
) -> ExperimentResult:
    """Execute one dataset/algorithm experiment with per-trial timeout.

    Parameters
    ----------
    cfg : DictConfig
        Experiment configuration hydrated by Hydra.
    parent_run_id : Optional[str]
        When provided the function assumes a parent MLflow run is
        **already active** and will *not* create one, nor log parent-level
        parameters.  This is the mode used by the grid pipeline.
        When ``None`` the function creates its own parent run and logs
        all parent-level params/tags.
    extra_params, extra_tags : optional
        Additional MLflow params / tags forwarded to every *trial* run.
    run_suffix : str
        Extra suffix appended to the MLflow run name.
    fallback_window : int
        Fallback window size used when FFT estimation fails.
    deadline : Optional[float]
        Absolute monotonic deadline for the parent run.
    timeout_seconds : Optional[float]
        Global budget in seconds.  When ``proportional_timeout`` is True,
        this budget is split proportionally across trials.  Otherwise each
        trial may use the full remaining global budget.
        When *both* ``deadline`` and ``timeout_seconds`` are given, the
        tighter constraint wins.
    proportional_timeout : bool
        When True, the global budget is split proportionally across trials
        weighted by series length, so that a single slow trial cannot
        consume the entire budget.  When False (default), each trial gets
        the full remaining time.

    Returns
    -------
    ExperimentResult
        Aggregated metrics and execution metadata.  The **caller** is
        responsible for calling ``mlflow.log_metrics(result.metrics)``
        on the appropriate run.
    """
    # ── Determine whether the caller owns the parent run ──────────────
    caller_owns_parent = parent_run_id is not None

    # ── Load data ─────────────────────────────────────────────────────
    _log_step(f"Loading dataset: {_BOLD}{cfg.dataset.name}{_RESET}")
    X_list, y_list = hydra.utils.call(cfg.dataset.loader)
    if not isinstance(X_list, list):
        X_list, y_list = [X_list], [y_list]

    # ── Modality & periodicity ────────────────────────────────────────
    dataset_modality_cfg = str(cfg.dataset.get("modality", "infer")).lower()
    dataset_periodic_cfg = bool(cfg.dataset.get("periodic", False))
    inferred_trials = [infer_trial_modality(trial) for trial in X_list]

    # ── Supervision mode (single normalisation) ───────────────────────
    experiment_cfg = resolve_experiment_cfg(cfg)
    supervision_mode = normalize_supervision_mode(
        experiment_cfg.get("supervision_mode", "unsupervised")
    )
    is_supervised = supervision_mode in {"semi_supervised", "supervised"}

    # ── FFT window estimation ─────────────────────────────────────────
    per_trial_windows: list[int | None] = [None] * len(X_list)
    if dataset_periodic_cfg:
        estimated = estimate_window_size_fft(trials=X_list, fallback=fallback_window) or []
        if not estimated:
            _log_warn(f"FFT estimation failed; fallback={fallback_window} for all trials.")
            per_trial_windows = [fallback_window] * len(X_list)
        else:
            for idx in range(len(X_list)):
                est = estimated[idx] if idx < len(estimated) else None
                if isinstance(est, int) and est > 0:
                    per_trial_windows[idx] = int(est)
                else:
                    _log_warn(f"Trial {idx + 1}: invalid FFT estimate; fallback={fallback_window}.")
                    per_trial_windows[idx] = fallback_window
            _log_info("Using per-trial FFT-derived window sizes (periodic dataset).")
    else:
        _log_info("Non-periodic dataset; using configured window_size.")

    # ── Dataset overview ──────────────────────────────────────────────
    overview_rows: list[dict[str, float]] = []
    for idx, (tX, ty, mod) in enumerate(zip(X_list, y_list, inferred_trials), start=1):
        arr = np.asarray(tX)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        overview_rows.append(
            {
                "dataset": cfg.dataset.name,
                "trial_index": idx,
                "modality_inferred": mod,
                "n_timepoints": int(arr.shape[0]),
                "n_channels": int(arr.shape[1]),
                "label_cardinality": int(np.unique(np.asarray(ty)).size)
                if np.asarray(ty).size
                else 0,
            }
        )

    trial_modalities = [
        _resolve_modality_for_trial(dataset_modality_cfg, inferred_trials, i)
        for i in range(len(X_list))
    ]
    unique_modalities = sorted(set(trial_modalities))

    # ── Algorithm capability probing ──────────────────────────────────
    algorithm_conf = cfg.algorithm.instance
    algorithm_cfg_for_tags = OmegaConf.create(
        OmegaConf.to_container(algorithm_conf, resolve=True, throw_on_missing=True)
    )

    if is_supervised:
        seed_labels = next((np.asarray(lbl) for lbl in y_list if np.asarray(lbl).size), None)
        if seed_labels is not None:
            overrides = build_supervision_param_overrides(seed_labels)
            if overrides:
                with open_dict(algorithm_cfg_for_tags):
                    for pname, pval in overrides.items():
                        if (
                            pname in algorithm_cfg_for_tags
                            and algorithm_cfg_for_tags[pname] is None
                        ):
                            # Same list/scalar split as the trial-level override
                            # above: position-valued overrides (oracle ablations)
                            # must not go through int().
                            algorithm_cfg_for_tags[pname] = (
                                [int(v) for v in pval]
                                if isinstance(pval, (list, tuple))
                                else int(pval)
                            )

    try:
        algo_for_tags = hydra.utils.instantiate(algorithm_cfg_for_tags)
        algo_tags = get_algorithm_tags(algo_for_tags)
        del algo_for_tags
    except Exception as _tag_probe_exc:
        # Instantiation may fail when required supervision parameters
        # (e.g. n_cps, n_segments) are null and only provided at trial
        # time.  Fall back to reading class-level _tags directly.
        _log_warn(
            f"Algorithm instantiation failed during tag probing "
            f"({_tag_probe_exc!r}); reading class-level tags instead."
        )
        _target_path = str(algorithm_cfg_for_tags.get("_target_", ""))
        if _target_path:
            algo_cls = hydra.utils.get_class(_target_path)
            algo_tags = dict(getattr(algo_cls, "_tags", {}))
        else:
            _log_warn("No _target_ in algorithm config; assuming empty tags.")
            algo_tags = {}
    supports_uni, supports_multi = resolve_capabilities(algo_tags)
    supports_unsupervised = bool(algo_tags.get("capability:unsupervised", False))
    supports_semi = bool(algo_tags.get("capability:semi_supervised", False))
    returns_dense = bool(algo_tags.get("returns_dense", False))

    configured_semi = bool(algorithm_conf.get("semi_supervised", False))

    raw_algorithm_task = cfg.algorithm.get("task", "auto")
    algorithm_task = resolve_algorithm_tag(raw_algorithm_task, algo_tags)

    # CP detectors → CP metrics only ; state detectors → state + CP metrics
    if algorithm_task == "change_point":
        allowed_metric_tasks: set[str] = {"change_point"}
    else:
        allowed_metric_tasks = {"state", "change_point"}

    # ── Log parent-level params ONLY when we own the parent run ───────
    if not caller_owns_parent:
        mlflow.log_param("supervision_mode", supervision_mode)
        mlflow.set_tag("supervision_mode", supervision_mode)
        mlflow.log_param("modality", ",".join(unique_modalities))
        mlflow.set_tag("modality", ",".join(unique_modalities))
        _log_dataset_overview(cfg.dataset.name, overview_rows)
        mlflow.set_tag("dataset_name", cfg.dataset.name)
        mlflow.log_param("dataset_name", cfg.dataset.name)
        mlflow.log_param("algorithm_name", cfg.algorithm.name)
        mlflow.log_param("algorithm_config_semi_supervised", configured_semi)
        mlflow.log_param("algorithm_supports_unsupervised", supports_unsupervised)
        mlflow.log_param("algorithm_supports_semi_supervised", supports_semi)
        mlflow.set_tag("algorithm_task", algorithm_task)
        mlflow.log_param("dataset_modality_config", dataset_modality_cfg)
        mlflow.log_param("algorithm_supports_univariate", supports_uni)
        mlflow.log_param("algorithm_supports_multivariate", supports_multi)
        mlflow.log_param("algorithm_returns_dense", returns_dense)
    else:
        # Even when the caller owns the parent, log the dataset overview
        # as an MLflow input (it's idempotent via context= key).
        _log_dataset_overview(cfg.dataset.name, overview_rows)

    # ── Filter compatible trials ──────────────────────────────────────
    filtered: list[tuple[int, np.ndarray, np.ndarray, str, int | None]] = []
    for idx, (sX, sy, mod) in enumerate(zip(X_list, y_list, trial_modalities)):
        if mod == "univariate" and not supports_uni:
            _log_warn(f"Skipping trial {idx + 1}: univariate not supported.")
            continue
        if mod == "multivariate" and not supports_multi:
            _log_warn(f"Skipping trial {idx + 1}: multivariate not supported.")
            continue
        if not is_supervised and not supports_unsupervised:
            _log_warn(f"Skipping trial {idx + 1}: algorithm requires supervision.")
            continue
        tw = per_trial_windows[idx] if dataset_periodic_cfg else None
        filtered.append((idx, sX, sy, mod, tw))

    if not filtered:
        _log_warn(
            f"No compatible trials for {cfg.algorithm.name} on "
            f"{cfg.dataset.name} [{supervision_mode}] — skipping."
        )
        if not caller_owns_parent:
            mlflow.set_tag("status", "skipped")
            mlflow.log_param("skip_reason", "no_compatible_trials")
        return ExperimentResult(
            metrics={},
            algorithm=None,
            timed_out=False,
            skipped=True,
            trials_completed=0,
            trials_total=len(X_list),
        )

    n_total = len(filtered)
    _log_step(f"{n_total} trial(s) to run.")

    # ── Pre-compute trial lengths for weighted budget allocation ───────
    trial_lengths = []
    for _, sX_tmp, _, _, _ in filtered:
        arr_tmp = np.asarray(sX_tmp)
        trial_lengths.append(arr_tmp.shape[0] if arr_tmp.ndim >= 1 else 1)

    # ── Build RunContext (explicit, passed to every trial) ────────────
    active = mlflow.active_run()
    if active is None:
        raise RuntimeError(
            "run_single_experiment must be called within an active MLflow run "
            "(either self-managed or provided by the caller via parent_run_id)."
        )

    ctx = RunContext(
        parent_run_id=active.info.run_id,
        experiment_id=active.info.experiment_id,
        supervision_mode=supervision_mode,
        algorithm_task=algorithm_task,
        is_supervised=is_supervised,
        returns_dense=returns_dense,
        supports_semi=supports_semi,
        allowed_metric_tasks=allowed_metric_tasks,
        timeout_seconds=timeout_seconds,
        deadline=deadline,
        proportional_timeout=proportional_timeout,
        extra_params=extra_params,
        extra_tags=extra_tags,
    )

    # ── Run trial loop (single nesting level) ─────────────────────────
    suffix_frag = run_suffix or ""
    mode_frag = f"_{supervision_mode}" if supervision_mode else ""
    base_run_name = f"{cfg.algorithm.name}_{cfg.dataset.name}{mode_frag}{suffix_frag}"

    return _run_trials(cfg, ctx, filtered, trial_lengths, base_run_name)


# ── Entry point ───────────────────────────────────────────────────────


def run_default_pipeline(cfg: DictConfig) -> None:
    """Entry point for the default pipeline.

    Creates its own parent run, logs parent-level parameters, then
    delegates to :func:`run_single_experiment` which opens one nested
    run per trial (single nesting level).
    """
    configure_mlflow(cfg)

    experiment_cfg = resolve_experiment_cfg(cfg)
    experiment_name = experiment_cfg.get("name", "tsseg-experiment-default")
    mlflow.set_experiment(experiment_name)

    supervision_mode = normalize_supervision_mode(
        experiment_cfg.get("supervision_mode", "unsupervised")
    )

    base_run_name = cfg.get("run_name", f"{cfg.algorithm.name}_{cfg.dataset.name}")
    run_name = f"{base_run_name}_{supervision_mode}"

    with mlflow.start_run(run_name=run_name) as run:
        _header(
            f"{cfg.algorithm.name}  {_ARROW}  {cfg.dataset.name}  [{supervision_mode}]",
            run_id=run.info.run_id,
        )
        run_start_time = time.monotonic()

        # Log params once at parent level
        params_to_log: dict[str, object] = {}
        algo_params = OmegaConf.to_container(
            cfg.algorithm.instance,
            resolve=True,
            throw_on_missing=True,
        )
        params_to_log.update({f"algo_{k}": v for k, v in algo_params.items()})

        if "preprocessing" in cfg and cfg.preprocessing is not None:
            preproc_params = OmegaConf.to_container(
                cfg.preprocessing,
                resolve=True,
                throw_on_missing=True,
            )
            params_to_log.update({f"preproc_{k}": v for k, v in preproc_params.items()})

        dataset_params = OmegaConf.to_container(
            cfg.dataset.loader,
            resolve=True,
            throw_on_missing=True,
        )
        params_to_log.update({f"dataset_{k}": v for k, v in dataset_params.items()})

        params_to_log["experiment_supervision_mode"] = supervision_mode
        mlflow.log_params(params_to_log)
        mlflow.set_tag("algorithm_name", cfg.algorithm.name)
        mlflow.set_tag("supervision_mode", supervision_mode)
        mlflow.set_tag("pipeline", "default_v2")

        timeout_seconds, deadline = resolve_parent_deadline(
            timeout_seconds=experiment_cfg.get("timeout_seconds"),
            timeout_hours=experiment_cfg.get("timeout_hours"),
        )
        proportional_timeout = bool(experiment_cfg.get("proportional_timeout", False))
        if timeout_seconds is not None:
            hours = timeout_seconds / 3600.0
            mode_label = "per-trial proportional" if proportional_timeout else "global"
            _log_info(f"Time budget: {int(timeout_seconds)}s (~{hours:.2f}h)  [{mode_label}]")
            mlflow.log_param("timeout_seconds", timeout_seconds)
            mlflow.log_param("proportional_timeout", proportional_timeout)

        try:
            # parent_run_id=None → run_single_experiment logs its own
            # parent-level params (algorithm_name, supervision_mode, …)
            result = run_single_experiment(
                cfg,
                deadline=deadline,
                timeout_seconds=timeout_seconds,
                proportional_timeout=proportional_timeout,
            )
            mlflow.log_metrics(result.metrics)
            if result.skipped:
                _log_warn("Experiment skipped (no compatible trials).")
                mlflow.set_tag("status", "skipped")
            elif result.timed_out:
                disp = int(timeout_seconds) if timeout_seconds else "configured budget"
                _log_error(f"Run timed out after {disp}s (partial metrics saved).")
                mlflow.set_tag("status", "timeout")
            else:
                mlflow.set_tag("status", "completed")
        except Exception as exc:
            _log_error(f"Run failed: {exc}")
            import traceback

            traceback.print_exc()
            mlflow.set_tag("status", "failed")
            mlflow.log_param("error_message", str(exc))
        finally:
            duration = float(max(time.monotonic() - run_start_time, 0.0))
            mlflow.log_metric("execution_time_seconds", duration)
            _footer(run_name, duration)


__all__ = [
    "ExperimentResult",
    "RunContext",
    "TimeoutException",
    "run_default_pipeline",
    "run_single_experiment",
    # Logging helpers shared with grid_pipeline
    "_header",
    "_footer",
    "_log_step",
    "_log_info",
    "_log_warn",
    "_log_error",
    "_log_success",
    "_format_metrics_table",
    "_BOLD",
    "_DIM",
    "_ARROW",
    "_RESET",
]
