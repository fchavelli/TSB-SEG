#!/bin/bash
# Step 2 of scripts/sms_error_breakdown.py, on a SLURM cluster next to the run
# artifacts. Run from the directory that holds sms_manifest.csv (step 1):
#
#   REPO=/path/to/TSB-SEG ARTIFACTS="/path/to/mlartifacts" DATA=/path/to/data \
#       sbatch --chdir=$PWD /path/to/TSB-SEG/scripts/sms_error_breakdown.sbatch
#
# REPO: this repository, with tsbseg installed in the active environment;
# ARTIFACTS: one or more MLflow artifact roots (space-separated); DATA: the
# datasets in the layout of the cluster that ran the benchmark.
#SBATCH --job-name=sms-breakdown
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
: "${REPO:?set REPO to the TSB-SEG checkout}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1

# shellcheck disable=SC2086
python "$REPO/scripts/sms_error_breakdown.py" compute sms_manifest.csv \
    -o sms_error_breakdown.parquet \
    --artifacts $ARTIFACTS \
    --data "$DATA" \
    --workers "$SLURM_CPUS_PER_TASK"
