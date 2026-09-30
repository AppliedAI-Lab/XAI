#!/usr/bin/env bash
set -euo pipefail

# Train Method 2 and Method 4 on every pre-split dataset.
# Method 2: FP substructure attention pooling.
# Method 4: Set Transformer over substructure embeddings.
# Override DATASETS, EPOCHS, BATCH_SIZE, LR, HIDDEN_DIM, or SEED from the shell if needed.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${ROOT_DIR}"

if [[ -n "${DATASETS:-}" ]]; then
  read -r -a DATASET_LIST <<< "${DATASETS}"
else
  DATASET_LIST=(FAAH CB2 PARP1 PBP2a USP7)
fi

EPOCHS="${EPOCHS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-0.0001}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
SEED="${SEED:-42}"

for dataset in "${DATASET_LIST[@]}"; do
  echo "Training Method 2 FP substructure attention for dataset: ${dataset}"
  python3 -m src.exps.gnn_attn.train \
    --dataset "${dataset}" \
    --encoder fp \
    --decompose brics \
    --model_type attn_pool \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --hidden_dim "${HIDDEN_DIM}" \
    --seed "${SEED}"

  echo "Training Method 4 Set Transformer for dataset: ${dataset}"
  python3 -m src.exps.gnn_attn.train \
    --dataset "${dataset}" \
    --encoder fp \
    --decompose brics \
    --model_type set_transformer \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --hidden_dim "${HIDDEN_DIM}" \
    --seed "${SEED}"
done
