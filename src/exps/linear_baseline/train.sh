#!/usr/bin/env bash
set -euo pipefail

# Train Method 1 on every pre-split dataset.
# By default this runs grid search. Set GRID_SEARCH=0 for a single config.
# Override DATASETS, REGRESSOR, POOLING, GRID_*, CV, N_JOBS, or SEED from the shell if needed.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${ROOT_DIR}"

if [[ -n "${DATASETS:-}" ]]; then
  read -r -a DATASET_LIST <<< "${DATASETS}"
else
  DATASET_LIST=(FAAH CB2 PARP1 PBP2a USP7)
fi

REGRESSOR="${REGRESSOR:-ridge}"
POOLING="${POOLING:-sum_mean_max}"
SEED="${SEED:-42}"
GRID_SEARCH="${GRID_SEARCH:-1}"
GRID_REGRESSORS="${GRID_REGRESSORS:-ridge,lasso,elasticnet}"
GRID_POOLINGS="${GRID_POOLINGS:-sum_mean,sum_mean_max}"
GRID_FP_RADII="${GRID_FP_RADII:-2,3}"
GRID_FP_BITS="${GRID_FP_BITS:-2048,4096}"
CV="${CV:-5}"
N_JOBS="${N_JOBS:--1}"
MAX_ITER="${MAX_ITER:-100000}"

for dataset in "${DATASET_LIST[@]}"; do
  echo "Training Method 1 linear baseline for dataset: ${dataset}"
  if [[ "${GRID_SEARCH}" == "1" ]]; then
    python3 -m src.exps.linear_baseline.train \
      --dataset "${dataset}" \
      --grid_search \
      --grid_regressors "${GRID_REGRESSORS}" \
      --grid_poolings "${GRID_POOLINGS}" \
      --grid_fp_radii "${GRID_FP_RADII}" \
      --grid_fp_bits "${GRID_FP_BITS}" \
      --cv "${CV}" \
      --n_jobs "${N_JOBS}" \
      --max_iter "${MAX_ITER}" \
      --seed "${SEED}"
  else
    python3 -m src.exps.linear_baseline.train \
      --dataset "${dataset}" \
      --regressor "${REGRESSOR}" \
      --pooling "${POOLING}" \
      --n_jobs "${N_JOBS}" \
      --max_iter "${MAX_ITER}" \
      --seed "${SEED}"
  fi
done
