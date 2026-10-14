# TSB-SEG: A Systematic Time-Series Segmentation Benchmark

Code, configurations and results of the TSB-SEG benchmark (NeurIPS 2026,
Evaluations & Datasets track). TSB-SEG evaluates 26 segmentation algorithms on
592 univariate and multivariate time series from 8 datasets, for change point
detection (CPD) and state detection (SD), plus a scalability case study (1 dataset).

- Paper: TODO(release) arXiv DOI, once accepted.
- Algorithms: [`tsseg`](https://github.com/fchavelli/tsseg), the library that
  implements every method of the benchmark behind one `fit`/`predict` API.
- Results: `results/`, in this repository (CC BY 4.0).

This repository is frozen on the paper: it changes only through errata (tagged
`v1.0.x`). New results and methods go to the online benchmark, not here.

## What is in this repository

| Path | Content |
|---|---|
| `src/tsbseg/` | Experiment pipelines (Hydra + MLflow): Default and grid runs, Non-Guided and Guided modes, per-series timeouts |
| `configs/` | Hydra configurations: every algorithm (default instance and grid), dataset, metric and experiment of the paper |
| `benchmark/analysis/` | Code of every figure and table of the paper (`python -m analysis.paper`) |
| `benchmark/configs/`, `benchmark/typology.json` | Experiments read by the analysis, taxonomy of the methods |
| `benchmark/datasets/` | Per-series features and domains |
| `benchmark/results/` | Derived tables the figures read (SMS error breakdown) |
| `scripts/` | Rebuild of the results database, SMS error breakdown |
| `results/` | Results of every run of the paper (Parquet, CC BY 4.0) |
| `data/` | Datasets: `python scripts/download_data.py` (see `data/README.md`) |

## Reproduce the figures and tables of the paper

The results of every run of the paper (555,768 series runs of 10,259
configurations) are published as Parquet tables. No experiment needs to be
rerun to rebuild the paper.

```bash
conda env create -f environment.yml && conda activate tsbseg
pip install -e .
make results-db   # rebuilds results/results.db from the tables of results/
make paper        # writes every figure and table to figures/paper/
```

`results/README.md` describes the tables. They can also be read directly:

```python
import pandas as pd

runs = pd.read_parquet("results/runs.parquet")  # one row per series run
configs = pd.read_parquet("results/configs.parquet")  # one row per configuration
```

## Rerun experiments

Experiments run through Hydra and log to MLflow. One algorithm on one dataset,
in the Default configuration, Non-Guided:

```bash
python scripts/download_data.py mocap
export MLFLOW_TRACKING_URI=sqlite:///$PWD/mlruns.db
python src/tsbseg/main.py algorithm=pelt dataset=mocap experiment=unsupervised
```

The four experiments of the paper are:

| Experiment config | Logged experiment | Mode |
|---|---|---|
| `experiment=unsupervised` | `non-guided-full-time-3` | Default, Non-Guided |
| `experiment=semi_supervised` | `guided-full-time-3` | Default, Guided (true number of change points or states given) |
| `experiment=grid_unsupervised_v3` | `grid-non-guided-v3` | Grid, Non-Guided |
| `experiment=grid_supervised_v3` | `grid-guided-v3` | Grid, Guided |

A grid experiment expands the `tunable_parameters` of the algorithm config
into a Cartesian product and runs one job per configuration; on a SLURM
cluster, add `hydra/launcher=slurm` (edit `configs/hydra/launcher/slurm.yaml`
for your partitions). The paper ran every series with a 24 h timeout, on CPU
nodes of one partition so that runtimes are comparable.

Rerunning an experiment does not reproduce every published number bit for bit:
see `REPRODUCIBILITY.md` for the library version, the nondeterministic methods
and the known deviations.

## Software versions

- [`tsseg` 0.3.0](https://pypi.org/project/tsseg/0.3.0/), installed by `pip install -e .` (see `pyproject.toml`).
- Python 3.12; `environment.yml` pins the versions of the cluster environment
  that ran the experiments, `requirements-analysis.txt` those of the
  environment that drew the figures.

## Citation

```bibtex
@inproceedings{chavelli2026tsbseg,
  title     = {{TSB-SEG}: A Systematic Time-Series Segmentation Benchmark},
  author    = {Chavelli, F{\'e}lix and Ermshaus, Arik and Yang, Fan and
               Sch{\"a}fer, Patrick and Paparrizos, John and Boniol, Paul},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS),
               Evaluations and Datasets Track},
  year      = {2026}
}
```

## License

Code: GNU Affero General Public License v3.0 (`LICENSE`). The published
results: CC BY 4.0. Each dataset keeps its own license
(`data/README.md`).
