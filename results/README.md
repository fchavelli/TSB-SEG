# Results

Results of every run of the TSB-SEG paper, released under CC BY 4.0.
`manifest.json` lists the sha256 of every file; `make results-db` rebuilds
from these tables the SQLite database (`results.db`, not versioned) that the
analysis reads.

| File | Rows | Content |
|---|---|---|
| `runs.parquet` | one per series run | `run_id`, `config_id`, `experiment`, `algorithm`, `dataset`, `series` (1-based), `run_name`, `status`, `lifecycle_stage`, `error_message`, `start_time`, `end_time`, `duration_s`, then one column per metric |
| `run_params.parquet` | one per (series run, parameter) | parameters logged by each series run, among them the ground truth given in Guided mode |
| `configs.parquet` | one per configuration | `config_id`, `experiment`, `algorithm`, `dataset`, `run_name`, `status`, timestamps, `n_series`, `n_finished`, then the aggregates the configuration logged |
| `config_params.parquet` | one per (configuration, parameter) | hyperparameters: `algo_*` the detector arguments, `grid_*` the swept ones, `*_target` the classes instantiated |
| `sms_error_breakdown.parquet` | one per Default state-detection run | SMS penalty per error type (delay, transition, isolation, missing) |
| `manifest.json` | | experiments (name, identifier, run count), provenance, sha256 |

`experiment` is one of `non_guided`, `guided` (Default configuration) and
`grid_non_guided`, `grid_guided` (grid search). The main metrics are
`bidirectional_covering_score` (BiC, change point detection) and
`state_matching_score` (SMS, state detection); `state_matching_score_score`
holds the SMS of the runs that logged it under that name.
