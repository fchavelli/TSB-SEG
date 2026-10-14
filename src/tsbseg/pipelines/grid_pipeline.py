"""Grid pipeline: every configuration of the grid of an algorithm on a dataset.

*  The grid is the Cartesian product of the algorithm's
   ``tunable_parameters``; each configuration is one parent run, with one
   child run per series directly under it.
*  The grid pipeline owns the parent run and logs its parameters;
   ``run_single_experiment(parent_run_id=...)`` only logs the series runs.
*  ``run_single_experiment`` returns :class:`ExperimentResult` instead of
   raising on timeout; the controller decides how to propagate it.
*  On SLURM, one job per configuration (submitit); if the controller fails,
   the remaining jobs are cancelled.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, ListConfig, OmegaConf

try:  # pragma: no cover
    import submitit
except ImportError:  # pragma: no cover
    submitit = None

from tsbseg.utils.main_helpers import (
    CHANGE_POINT_PARAM_ALIASES,
    SEGMENT_COUNT_PARAM_ALIASES,
    STATE_COUNT_PARAM_ALIASES,
    configure_mlflow,
    normalize_supervision_mode,
    resolve_experiment_cfg,
    resolve_parent_deadline,
)

from .default_pipeline import (
    _ARROW,
    ExperimentResult,
    _footer,
    _header,
    _log_error,
    _log_info,
    _log_success,
    _log_warn,
    run_single_experiment,
)

# ── Supervision-aware grid pruning ────────────────────────────────────

_SUPERVISION_ALIASES = {
    alias.lower()
    for alias in (
        list(CHANGE_POINT_PARAM_ALIASES)
        + list(SEGMENT_COUNT_PARAM_ALIASES)
        + list(STATE_COUNT_PARAM_ALIASES)
    )
}


def _is_supervised_mode(cfg: DictConfig) -> bool:
    experiment_cfg = resolve_experiment_cfg(cfg)
    mode = normalize_supervision_mode(experiment_cfg.get("supervision_mode", "unsupervised"))
    return mode in {"semi_supervised", "supervised"}


def _prune_supervision_parameters(cfg: DictConfig, tunable_spec: dict[str, Any]) -> dict[str, Any]:
    if not tunable_spec:
        return tunable_spec

    def _names(key: str) -> set:
        return {str(name).strip().lower() for name in cfg.algorithm.get(key, None) or []}

    if _is_supervised_mode(cfg):
        # Parameters that the supervision overrides make irrelevant, declared by the
        # algorithm config: e.g. the penalty of a detector once n_cps is given.
        dropped = _SUPERVISION_ALIASES | _names("unsupervised_only_parameters")
    else:
        # Axes swept in guided mode only; non-guided runs keep the instance value
        # (e.g. an axis that calibration showed to be flat, dropped from a larger
        # non-guided grid while the guided grid already ran with it).
        dropped = _names("guided_only_parameters")
        if not dropped:
            return tunable_spec
    retained: dict[str, Any] = {}
    removed: list[str] = []
    for raw_name, spec in tunable_spec.items():
        name = str(raw_name).strip().lower()
        if name in dropped:
            removed.append(str(raw_name))
            continue
        retained[str(raw_name)] = spec

    if removed:
        _log_info(f"Pruned supervision-controlled params from grid: {', '.join(sorted(removed))}")
    return retained


# ── Grid building utilities ───────────────────────────────────────────


def _normalise_tunable_parameters(spec: Any) -> dict[str, Any]:
    if spec is None:
        return {}
    if isinstance(spec, (DictConfig, ListConfig)):
        spec = OmegaConf.to_container(spec, resolve=True, throw_on_missing=False)
    if isinstance(spec, dict):
        return spec
    if isinstance(spec, list):
        normalised: dict[str, Any] = {}
        for entry in spec:
            if isinstance(entry, (DictConfig, ListConfig)):
                entry = OmegaConf.to_container(entry, resolve=True, throw_on_missing=False)
            if not isinstance(entry, dict) or len(entry) != 1:
                raise ValueError(
                    "Each 'tunable_parameters' list entry must map a single name to a specification."
                )
            name, value = next(iter(entry.items()))
            normalised[str(name)] = value
        return normalised
    raise ValueError(
        "Unsupported 'tunable_parameters' structure; expected dict or list of mappings."
    )


def _expand_parameter_axis(name: str, spec: Any) -> list[Any]:
    if isinstance(spec, (list, tuple)):
        return list(spec)
    if isinstance(spec, DictConfig):
        spec = OmegaConf.to_container(spec, resolve=True, throw_on_missing=False)
    if isinstance(spec, dict):
        if "values" in spec:
            return list(spec["values"])
        if "grid" in spec:
            g = spec["grid"]
            start, stop = float(g.get("start", 0.0)), float(g.get("stop", 1.0))
            num = int(g.get("num", 5))
            base = float(g.get("base", math.e))
            if num <= 1:
                return [base**start]
            return list(base ** np.linspace(start, stop, num=num))
        if {"min", "max", "step"}.issubset(spec.keys()):
            cur, mx, step = float(spec["min"]), float(spec["max"]), float(spec["step"])
            vals = []
            while cur <= mx + 1e-12:
                vals.append(cur)
                cur += step
            return vals
    raise ValueError(f"Unsupported specification for parameter '{name}'.")


def _build_parameter_grid(cfg: DictConfig) -> list[dict[str, Any]]:
    tunable_spec = _normalise_tunable_parameters(cfg.algorithm.get("tunable_parameters"))
    tunable_spec = _prune_supervision_parameters(cfg, tunable_spec)
    if not tunable_spec:
        return [{}]
    axes: list[list[tuple[str, Any]]] = []
    for pname, pspec in tunable_spec.items():
        axis_values = _expand_parameter_axis(pname, pspec)
        if axis_values:
            axes.append([(pname, v) for v in axis_values])
    if not axes:
        return [{}]
    return [{n: v for n, v in combo} for combo in _cartesian_product(*axes)]


def _cartesian_product(*axes: Iterable[tuple[str, Any]]) -> Iterable[list[tuple[str, Any]]]:
    if not axes:
        return []

    def recurse(idx: int, acc: list[tuple[str, Any]]):
        if idx == len(axes):
            yield list(acc)
            return
        for item in axes[idx]:
            acc.append(item)
            yield from recurse(idx + 1, acc)
            acc.pop()

    return recurse(0, [])


def _format_run_suffix(params: dict[str, Any]) -> str:
    if not params:
        return ""
    parts = [f"{n}_{str(v).replace(' ', '')}" for n, v in sorted(params.items())]
    return "_" + "_".join(parts)


def _apply_parameter_overrides(cfg: DictConfig, overrides: dict[str, Any]) -> DictConfig:
    cfg_copy = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    for key, value in overrides.items():
        cfg_copy.algorithm.instance[key] = value
    return cfg_copy


def _sanitize_param_key(key: str) -> str:
    return str(key).strip("_").replace(".", "_")


def _stringify_param_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return json.dumps([_stringify_param_value(v) for v in value])
    if isinstance(value, dict):
        return json.dumps({str(k): _stringify_param_value(v) for k, v in value.items()})
    if isinstance(value, (set, frozenset)):
        return json.dumps(sorted(_stringify_param_value(v) for v in value))
    return value


def _prefixed_params(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{_sanitize_param_key(k)}": v for k, v in values.items()}


# ── Grid controller ──────────────────────────────────────────────────


class GridPipeline:
    """Coordinate distributed or sequential grid execution."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.grid_cfg = cfg.get("grid", {})
        self.parameter_grid = _build_parameter_grid(cfg)
        self.base_run_name = cfg.get("run_name", f"{cfg.algorithm.name}_{cfg.dataset.name}")
        self.execution_mode = self._resolve_execution_mode()

    def run(self) -> None:
        _log_info(f"Grid: {len(self.parameter_grid)} parameter combination(s).")
        if self.execution_mode == "worker":
            self._run_worker()
            return

        if not self.parameter_grid:
            _log_info("No tunable parameters — running single default config.")
            self._run_single_variant(self.cfg, {}, role="worker")
            return

        executor = self._create_submitit_executor()
        if executor is None or self.grid_cfg.get("force_sequential", False):
            if executor is None:
                _log_info("Submitit unavailable; running grid sequentially.")
            else:
                _log_info("Sequential mode forced; running grid locally.")
            self._run_sequential()
            return

        self._run_submitit(executor)

    # ── Controller helpers ────────────────────────────────────────────

    def _create_submitit_executor(self) -> Any | None:
        if submitit is None:
            return None
        try:
            hydra_cfg = HydraConfig.get()
        except Exception:
            return None
        launcher_cfg = getattr(hydra_cfg, "launcher", None)
        if not launcher_cfg:
            return None
        launcher_dict = (
            OmegaConf.to_container(launcher_cfg, resolve=True, throw_on_missing=False) or {}
        )
        output_dir = hydra_cfg.runtime.output_dir or os.getcwd()
        submitit_dir = self.grid_cfg.get(
            "submitit_folder",
            str(Path(output_dir) / "submitit_grid_jobs"),
        )
        folder = Path(submitit_dir).resolve()
        folder.mkdir(parents=True, exist_ok=True)
        executor = submitit.AutoExecutor(folder=folder)
        exec_params = self._extract_submitit_parameters(launcher_dict)
        executor.update_parameters(**exec_params)
        return executor

    def _run_submitit(self, executor: Any) -> None:
        """Submit one SLURM job per grid combo.

        Behaviour depends on ``grid.wait_for_results`` (default: auto):

        *  ``True``  — block until all jobs finish (good when the
           controller runs on a login node or lightweight allocation).
        *  ``False`` — submit all jobs and return immediately
           (**fire-and-forget**).  Use this when the controller is
           itself a SLURM job to avoid wasting an allocation slot.
        *  ``"auto"`` (default) — ``True`` if running outside a SLURM
           allocation (login node), ``False`` if ``SLURM_JOB_ID`` is set.

        A manifest of submitted job IDs is written to
        ``<submitit_folder>/grid_jobs.json`` for later monitoring.
        """
        wait_cfg = self.grid_cfg.get("wait_for_results", "auto")
        if isinstance(wait_cfg, str) and wait_cfg.lower() == "auto":
            wait_for_results = "SLURM_JOB_ID" not in os.environ
        else:
            wait_for_results = bool(wait_cfg)

        base_command = self._build_base_command()
        working_dir = self._working_directory()
        base_overrides = self._base_overrides()

        jobs: list[tuple[dict[str, Any], Any]] = []
        for index, combo in enumerate(self.parameter_grid):
            overrides = self._build_job_overrides(list(base_overrides), combo)
            command = base_command + overrides
            job_name = self._job_name(combo, index)
            executor.update_parameters(name=job_name)
            job = executor.submit(_run_cli_command, command, working_dir)
            jobs.append((combo, job))
            _log_info(f"Submitted: {job_name}  (SLURM job {job.job_id})")

        # ── Persist manifest for monitoring / debugging ───────────────
        manifest = {
            "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "n_combos": len(jobs),
            "wait_for_results": wait_for_results,
            "jobs": [
                {
                    "index": i,
                    "combo": {k: _stringify_param_value(v) for k, v in combo.items()},
                    "slurm_job_id": str(getattr(job, "job_id", "unknown")),
                }
                for i, (combo, job) in enumerate(jobs)
            ],
        }
        manifest_dir = self.grid_cfg.get(
            "submitit_folder",
            str(Path(working_dir) / "submitit_grid_jobs"),
        )
        manifest_path = Path(manifest_dir) / "grid_jobs.json"
        try:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(manifest, indent=2))
            _log_info(f"Job manifest: {manifest_path}")
        except Exception as exc:
            _log_warn(f"Could not write manifest: {exc}")

        if not wait_for_results:
            _log_success(
                f"Fire-and-forget: {len(jobs)} job(s) submitted.  "
                f"Monitor via: squeue -u $USER | grep {self.base_run_name[:12]}"
            )
            return

        # ── Wait mode: block until all jobs complete ──────────────────
        _log_info("Waiting for all jobs to complete...")
        first_error: Exception | None = None
        for combo, job in jobs:
            try:
                job.result()
                _log_success(f"Job {combo} completed.")
            except Exception as exc:
                _log_error(f"Job {combo} failed: {exc}")
                first_error = exc
                break

        if first_error is not None:
            _log_warn("Cancelling remaining SLURM jobs...")
            for _, j in jobs:
                try:
                    j.cancel()
                except Exception:
                    pass
            raise first_error

    def _extract_submitit_parameters(self, cfg_dict: dict[str, Any]) -> dict[str, Any]:
        params: dict[str, Any] = {}
        for key in (
            "timeout_min",
            "cpus_per_task",
            "gpus_per_node",
            "tasks_per_node",
            "mem_gb",
            "nodes",
            "stderr_to_stdout",
            "name",
        ):
            if key in cfg_dict:
                params[key] = cfg_dict[key]

        slurm_keys = {
            "partition",
            "qos",
            "account",
            "gres",
            "constraint",
            "comment",
            "mail_type",
            "mail_user",
            "mem",
            "mem_per_gpu",
            "mem_per_cpu",
            "num_gpus",
            "ntasks_per_node",
            "srun_args",
            "exclude",
            "time",
            "gpus_per_task",
            "signal_delay_s",
            "cpus_per_gpu",
            "dependency",
            "use_srun",
            "array_parallelism",
            "additional_parameters",
            "setup",
            "wckey",
            "exclusive",
            "nodelist",
        }
        for key in slurm_keys:
            if key not in cfg_dict:
                continue
            value = cfg_dict[key]
            if key == "array_parallelism":
                params["slurm_array_parallelism"] = value
                continue
            if key == "setup":
                setup_cmds: list[str] = (
                    [str(c) for c in value]
                    if isinstance(value, (list, tuple))
                    else []
                    if value is None
                    else [str(value)]
                )
                for cmd in ("unset SLURM_ARRAY_JOB_ID", "unset SLURM_ARRAY_TASK_ID"):
                    if cmd not in setup_cmds:
                        setup_cmds.append(cmd)
                params["slurm_setup"] = setup_cmds
                continue
            params[f"slurm_{key}"] = value
        return params

    def _build_base_command(self) -> list[str]:
        entrypoint = self.grid_cfg.get("entrypoint")
        if entrypoint:
            script_path = Path(entrypoint)
            if not script_path.is_absolute():
                script_path = Path(self._working_directory()) / script_path
            script = str(script_path)
        else:
            script = str(Path(__file__).resolve().parent.parent / "main.py")
        return [sys.executable, script]

    def _working_directory(self) -> str:
        try:
            return HydraConfig.get().runtime.cwd
        except Exception:
            return os.getcwd()

    def _base_overrides(self) -> list[str]:
        try:
            overrides = list(HydraConfig.get().overrides.task)
        except Exception:
            return []
        return [
            ov
            for ov in overrides
            if not ov.startswith("grid.execution_mode")
            and not ov.startswith("grid.run_suffix")
            and not ov.startswith("run_name=")
            and not ov.startswith("+run_name=")
        ]

    def _build_job_overrides(self, base: list[str], combo: dict[str, Any]) -> list[str]:
        overrides = list(base)
        overrides.append("+grid.execution_mode=worker")
        run_suffix = _format_run_suffix(combo)
        target = f"{self.base_run_name}{run_suffix}" if run_suffix else self.base_run_name
        overrides.append(f"+run_name={self._quote_override_value(target)}")
        for name, value in combo.items():
            overrides.append(f"algorithm.instance.{name}={self._quote_override_value(value)}")
        return overrides

    def _job_name(self, combo: dict[str, Any], index: int) -> str:
        base = self.grid_cfg.get("job_name", self.base_run_name) or "grid"
        suffix = _format_run_suffix(combo)
        return f"{base}{suffix or ''}_{index:03d}".replace("/", "_")

    def _run_sequential(self) -> None:
        for combo in self.parameter_grid:
            variant_cfg = _apply_parameter_overrides(self.cfg, combo)
            self._run_single_variant(variant_cfg, combo, role="sequential")

    # ── Worker helpers ────────────────────────────────────────────────

    def _run_worker(self) -> None:
        combo = self._extract_active_parameters()
        self._run_single_variant(self.cfg, combo, role="worker")

    def _run_single_variant(
        self,
        variant_cfg: DictConfig,
        combo: dict[str, Any],
        role: str = "worker",
    ) -> None:
        """Execute one grid combo as a **single parent run**.

        Trial runs are nested directly under this parent (one level of
        nesting only).  ``run_single_experiment`` receives
        ``parent_run_id`` so it does **not** log parent-level params.
        """
        configure_mlflow(variant_cfg)

        experiment_cfg = resolve_experiment_cfg(variant_cfg)
        experiment_name = experiment_cfg.get("name", "default_experiment")
        mlflow.set_experiment(experiment_name)

        run_suffix = _format_run_suffix(combo)
        run_name = variant_cfg.get(
            "run_name", f"{variant_cfg.algorithm.name}_{variant_cfg.dataset.name}"
        )
        if run_suffix and run_suffix not in run_name:
            run_name = f"{run_name}{run_suffix}"

        timeout_seconds, deadline = resolve_parent_deadline(
            timeout_seconds=experiment_cfg.get("timeout_seconds"),
            timeout_hours=experiment_cfg.get("timeout_hours"),
        )
        proportional_timeout = bool(experiment_cfg.get("proportional_timeout", False))

        with mlflow.start_run(run_name=run_name) as parent_run:
            _header(
                f"{variant_cfg.algorithm.name}  {_ARROW}  "
                f"{variant_cfg.dataset.name}  [grid/{role}]",
                run_id=parent_run.info.run_id,
            )
            if timeout_seconds is not None:
                mode_label = "per-trial proportional" if proportional_timeout else "global"
                _log_info(
                    f"Time budget: {int(timeout_seconds)}s "
                    f"(~{timeout_seconds / 3600:.2f}h)  [{mode_label}]"
                )
            run_start = time.monotonic()

            # ── Parent-level params (logged ONCE, HERE) ───────────────
            params_to_log: dict[str, Any] = {}
            algo_params = OmegaConf.to_container(
                variant_cfg.algorithm.instance,
                resolve=True,
                throw_on_missing=True,
            )
            prefixed_algo = _prefixed_params("algo", algo_params)
            params_to_log.update(prefixed_algo)

            prefixed_preproc: dict[str, Any] = {}
            if "preprocessing" in variant_cfg and variant_cfg.preprocessing is not None:
                preproc_params = OmegaConf.to_container(
                    variant_cfg.preprocessing,
                    resolve=True,
                    throw_on_missing=True,
                )
                prefixed_preproc = _prefixed_params("preproc", preproc_params)
                params_to_log.update(prefixed_preproc)

            dataset_params = OmegaConf.to_container(
                variant_cfg.dataset.loader,
                resolve=True,
                throw_on_missing=True,
            )
            params_to_log.update(_prefixed_params("dataset", dataset_params))

            prefixed_grid: dict[str, Any] = {}
            if combo:
                prefixed_grid = _prefixed_params("grid", combo)
                params_to_log.update(prefixed_grid)

            params_to_log["pipeline"] = "grid_v2"
            params_to_log["grid_execution_role"] = role

            # Also log algorithm-level params that run_single_experiment
            # would have logged — they belong to the parent.
            supervision_mode = normalize_supervision_mode(
                experiment_cfg.get("supervision_mode", "unsupervised")
            )
            params_to_log["supervision_mode"] = supervision_mode
            params_to_log["dataset_name"] = variant_cfg.dataset.name
            params_to_log["algorithm_name"] = variant_cfg.algorithm.name

            serialised = {k: _stringify_param_value(v) for k, v in params_to_log.items()}
            mlflow.log_params({k: str(v) for k, v in serialised.items()})

            # Tags
            mlflow.set_tag("algorithm_name", variant_cfg.algorithm.name)
            mlflow.set_tag("pipeline", "grid_v2")
            mlflow.set_tag("grid_execution_role", role)
            mlflow.set_tag("supervision_mode", supervision_mode)

            if timeout_seconds is not None:
                mlflow.log_param("timeout_seconds", timeout_seconds)

            for source in (prefixed_algo, prefixed_preproc, prefixed_grid):
                for k, v in source.items():
                    mlflow.set_tag(k, str(_stringify_param_value(v)))

            grid_parameters_payload = (
                json.dumps(
                    {_sanitize_param_key(k): _stringify_param_value(v) for k, v in combo.items()}
                )
                if combo
                else "{}"
            )
            mlflow.set_tag("grid_parameters", grid_parameters_payload)

            # Tags to forward to child trial runs
            child_tags: dict[str, str] = {
                "pipeline": "grid_v2",
                "grid_execution_role": role,
                "grid_parameters": grid_parameters_payload,
            }
            for source in (prefixed_algo, prefixed_preproc, prefixed_grid):
                for k, v in source.items():
                    child_tags[k] = str(_stringify_param_value(v))

            try:
                # ── KEY CHANGE: pass parent_run_id ────────────────────
                # run_single_experiment will NOT create a parent run
                # and will NOT log parent-level params (it sees
                # parent_run_id != None).  Trial runs are nested
                # directly under *this* parent run → single nesting
                # level.
                result: ExperimentResult = run_single_experiment(
                    variant_cfg,
                    parent_run_id=parent_run.info.run_id,
                    extra_tags=child_tags,
                    run_suffix=run_suffix,
                    deadline=deadline,
                    timeout_seconds=timeout_seconds,
                    proportional_timeout=proportional_timeout,
                )

                # Log aggregated metrics on the parent run
                mlflow.log_metrics(result.metrics)

                if result.skipped:
                    _log_warn("Experiment skipped (no compatible trials).")
                    mlflow.set_tag("status", "skipped")
                elif result.timed_out:
                    disp = int(timeout_seconds) if timeout_seconds else "configured budget"
                    _log_error(
                        f"Grid run timed out after {disp}s "
                        f"({result.trials_completed}/{result.trials_total} trials, "
                        f"partial metrics saved)."
                    )
                    mlflow.set_tag("status", "timeout")
                else:
                    mlflow.set_tag("status", "completed")

            except Exception as exc:
                _log_error(f"Grid run failed: {exc}")
                import traceback

                traceback.print_exc()
                mlflow.set_tag("status", "failed")
                mlflow.log_param("error_message", str(exc))
            finally:
                duration = float(max(time.monotonic() - run_start, 0.0))
                mlflow.log_metric("execution_time_seconds", duration)
                _footer(run_name, duration)

    # ── Utilities ─────────────────────────────────────────────────────

    def _resolve_execution_mode(self) -> str:
        mode_cfg = str(self.grid_cfg.get("execution_mode", "auto")).lower()
        if mode_cfg in {"worker", "controller"}:
            return mode_cfg
        try:
            hydra_mode = HydraConfig.get().mode
        except Exception:
            hydra_mode = None
        if hydra_mode and str(hydra_mode).upper() == "MULTIRUN":
            return "worker"
        return "controller"

    def _extract_active_parameters(self) -> dict[str, Any]:
        spec = _normalise_tunable_parameters(self.cfg.algorithm.get("tunable_parameters"))
        active: dict[str, Any] = {}
        instance_cfg = self.cfg.algorithm.instance
        for name in spec:
            if name in instance_cfg:
                active[name] = instance_cfg.get(name)
        return active

    @staticmethod
    def _quote_override_value(value: Any) -> str:
        if isinstance(value, str):
            return f'"{value.replace(chr(34), chr(92) + chr(34))}"'
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)


# ── Entry point ───────────────────────────────────────────────────────


def run_grid_pipeline(cfg: DictConfig) -> None:
    """Entry point bridging Hydra config with the grid pipeline."""
    configure_mlflow(cfg)
    GridPipeline(cfg).run()


def _run_cli_command(command: list[str], workdir: str) -> int:
    _log_info(f"Executing: {' '.join(command[:3])}... (cwd={workdir})")
    completed = subprocess.run(command, cwd=workdir)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Grid worker command failed (rc={completed.returncode}): {' '.join(command)}"
        )
    return completed.returncode


__all__ = ["run_grid_pipeline", "GridPipeline"]
