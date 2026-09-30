#!/usr/bin/env bash
set -euo pipefail

# Train Method 3 on every pre-split dataset.
# Requires a pretrained GNN checkpoint. Override GNN_WEIGHTS if needed.
# Override DATASETS, EPOCHS, BATCH_SIZE, LR, HIDDEN_DIM, or SEED from the shell if needed.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${ROOT_DIR}"

if [[ -n "${DATASETS:-}" ]]; then
  read -r -a DATASET_LIST <<< "${DATASETS}"
else
  DATASET_LIST=(FAAH CB2 PARP1 PBP2a USP7)
fi

GNN_WEIGHTS="${GNN_WEIGHTS:-${ROOT_DIR}/weights/gnn/gnn_pretrain.pt}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-0.001}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
SEED="${SEED:-42}"

if [[ ! -f "${GNN_WEIGHTS}" ]]; then
  echo "GNN checkpoint not found: ${GNN_WEIGHTS}"
  echo "Run src/tools/gnn_pretrain/train_gnn.py first or set GNN_WEIGHTS."
  exit 1
fi

for dataset in "${DATASET_LIST[@]}"; do
  echo "Training Method 3 GNN substructure attention for dataset: ${dataset}"
  python3 -m src.exps.gnn_sub_attn.train \
    --dataset "${dataset}" \
    --gnn_weights "${GNN_WEIGHTS}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --hidden_dim "${HIDDEN_DIM}" \
    --seed "${SEED}"
done
