#!/usr/bin/env bash
set -euo pipefail

# Method 5 has no learnable parameters.
# Train a base predictor first, then run masking inference with that checkpoint.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${ROOT_DIR}"

cat <<'EOF'
Method 5 (masking) does not require training.

Train one of these base predictors per dataset first:
  bash src/exps/linear_baseline/train.sh
  bash src/exps/gnn_attn/train.sh
  bash src/exps/gnn_sub_attn/train.sh

Then run masking inference, for example:
  python3 -m src.exps.masking.infer \
    --predictor attn \
    --model weights/FAAH/attn/attn_model_fp_brics_attn_pool.pt \
    --smiles "CCO" \
    --out_heatmap outputs/masking_heatmap.png
EOF
