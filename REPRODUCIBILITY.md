# Reproducibility

Three levels, from the cheapest to the most expensive.

| Level | What you get | Needs | Time |
|---|---|---|---|
| 1. Paper from results | Every figure and table of the paper, from the published runs | `results/` (in the repository) | minutes |
| 2. One cell | One algorithm × dataset × mode rerun, to compare with its published runs | `data/`, `tsseg` | minutes to hours |
| 3. Full benchmark | All four experiments | a SLURM cluster | days |

## 1. Paper from results

`make results-db paper` rebuilds every figure and table from the
published runs. The analysis code was written against the MLflow tracking
database of the benchmark: `scripts/build_results_db.py` rebuilds, from the
Parquet tables, the part of that database it reads (runs, the tags that link a
series run to its configuration, all parameters, the last value of every
metric), with the original identifiers.

Before release, the rebuilt database was compared with the original one: the
parent runs, the metrics of every Default and grid run, the parameters of
every configuration and the series runs of all statuses read by the analysis
are identical (same rows, same values, same timestamps).

Two derived tables are not recomputed by `make paper`:

- `results/sms_error_breakdown.parquet` (Figures 5c, 6c) splits the SMS
  penalty of each Default state-detection run by error type. It needs the
  predicted labels of the runs, which the published results do not contain;
  `scripts/sms_error_breakdown.py` computed it from the run artifacts, and
  checked for every run that the ground truth matches the one the run logged
  and that the recomputed SMS equals the logged SMS.
- `benchmark/datasets/dataset_features.csv` (length, channels, change points
  and states of each series) is computed from the datasets by
  `benchmark/dataset_features.py`.

## 2 and 3. Rerunning experiments

### What was run

The four experiments of the paper (`benchmark/configs/benchmark_config_v3.yaml`):

| Experiment | Configurations | Series runs |
|---|---|---|
| `non-guided-full-time-3` (Default, Non-Guided) | one per algorithm and dataset | 5,336 |
| `guided-full-time-3` (Default, Guided) | one per algorithm and dataset | 8,543 |
| `grid-non-guided-v3` (Grid, Non-Guided) | one per grid cell and dataset | 335,270 |
| `grid-guided-v3` (Grid, Guided) | one per grid cell and dataset | 206,619 |

### Library version

The runs do not log the `tsseg` version they used. This repository pins
`tsseg` 0.3.0, which contains every correction above. Other corrections of
`tsseg` made between the first campaign and this version were not followed by
a rerun: a rerun with this version can differ slightly from the published runs
of the methods concerned.

### Nondeterministic methods

Methods that train neural networks or sample (E2USD, Time2State, TIRE,
TS-CP², tGLAD, HDP-HSMM, TICC initialisation, Random) give different
predictions from one run to the next unless their seed is fixed and the
hardware is the same. Their reruns match the published scores in
distribution, not run by run.

### Selection rules of the analysis

- Grid configurations run more than once (reruns after a timeout or a node
  failure) count once: the analysis keeps the run with the most finished
  series, then the highest mean score, then the latest
  (`benchmark_analysis.py`, `_deduplicate_grid_parents`). For a
  nondeterministic method this favours it slightly.
- 326 series runs carry the MLflow `deleted` flag and 93 never finished
  (`RUNNING`); both are published as logged, and the analysis treats them as
  any run of their status.

### Series order

`series` (the `dataset_trial_index` of the runs) is 1-based, in the order in
which the data loader listed the series on the cluster that ran the
benchmark. `src/tsbseg/data_loading/series.csv` gives, for each series, the
loader parameters that name its file, its length, channels and number of
change points; the loader lists the series in this order on every machine.
The file was built from the directory listing of the cluster and every series
checked against the runs.
