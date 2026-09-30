#!/usr/bin/env bash
# Run target-validation infer for all methods x targets in data/targets.csv.
# Discovers checkpoints under weights/; skips missing ones.
# Heatmaps + CSVs land under outputs/target_validation/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

TARGETS_CSV="${TARGETS_CSV:-${ROOT_DIR}/data/targets.csv}"
WEIGHTS_DIR="${WEIGHTS_DIR:-${ROOT_DIR}/weights}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/outputs/target_validation}"
GNN_WEIGHTS="${GNN_WEIGHTS:-}"

if [[ ! -f "${TARGETS_CSV}" ]]; then
  echo "Targets CSV not found: ${TARGETS_CSV}"
  exit 1
fi

mapfile -t TARGETS < <(
  python3 - <<PY
import pandas as pd
df = pd.read_csv("${TARGETS_CSV}")
for t in df["target"].drop_duplicates().tolist():
    print(t)
PY
)

if [[ ${#TARGETS[@]} -eq 0 ]]; then
  echo "No targets found in ${TARGETS_CSV}"
  exit 1
fi

# Prefer explicit GNN_WEIGHTS, else first matching pretrained GNN under weights/.
if [[ -z "${GNN_WEIGHTS}" ]]; then
  if [[ -f "${WEIGHTS_DIR}/gnn/gnn_pretrain.pt" ]]; then
    GNN_WEIGHTS="${WEIGHTS_DIR}/gnn/gnn_pretrain.pt"
  elif [[ -f "${WEIGHTS_DIR}/gnn_pretrain.pt" ]]; then
    GNN_WEIGHTS="${WEIGHTS_DIR}/gnn_pretrain.pt"
  else
    GNN_WEIGHTS="$(find "${WEIGHTS_DIR}" -type f -name 'gnn_pretrain.pt' 2>/dev/null | head -n 1 || true)"
  fi
fi

mkdir -p "${OUT_ROOT}"

# Resolve a checkpoint for (target, kind). Prints path or empty.
resolve_ckpt() {
  local target="$1"
  local kind="$2"
  local candidates=()
  local c

  case "${kind}" in
    linear)
      candidates=(
        "${WEIGHTS_DIR}/linear_baseline/grid/${target}/linear_agg_ecfp_grid_${target}.joblib"
        "${WEIGHTS_DIR}/linear_baseline/${target}/linear_agg_ecfp_grid_${target}.joblib"
      )
      ;;
    attn_pool)
      candidates=(
        "${WEIGHTS_DIR}/${target}/attn/attn_model_fp_brics_attn_pool.pt"
        "${WEIGHTS_DIR}/exp2_4_gnn_attn/${target}/attn/attn_model_fp_brics_attn_pool.pt"
      )
      ;;
    set_transformer)
      candidates=(
        "${WEIGHTS_DIR}/${target}/attn/attn_model_fp_brics_set_transformer.pt"
        "${WEIGHTS_DIR}/exp2_4_gnn_attn/${target}/attn/attn_model_fp_brics_set_transformer.pt"
      )
      ;;
    gnn_sub_attn)
      candidates=(
        "${WEIGHTS_DIR}/${target}/gnn_sub_attn/gnn_sub_attn.pt"
        "${WEIGHTS_DIR}/exp3_gnn_sub_attn/${target}/gnn_sub_attn/gnn_sub_attn.pt"
      )
      ;;
    *)
      echo "Unknown checkpoint kind: ${kind}" >&2
      return 1
      ;;
  esac

  for c in "${candidates[@]}"; do
    if [[ -f "${c}" ]]; then
      echo "${c}"
      return 0
    fi
  done

  # Fallback: search under weights/ for files whose path contains the target name.
  case "${kind}" in
    linear)
      find "${WEIGHTS_DIR}" -type f -path "*${target}*" -name "linear_agg_ecfp_grid_${target}.joblib" 2>/dev/null | sort | head -n 1
      ;;
    attn_pool)
      find "${WEIGHTS_DIR}" -type f -path "*${target}*" -name "attn_model_fp_brics_attn_pool.pt" 2>/dev/null | sort | head -n 1
      ;;
    set_transformer)
      find "${WEIGHTS_DIR}" -type f -path "*${target}*" -name "attn_model_fp_brics_set_transformer.pt" 2>/dev/null | sort | head -n 1
      ;;
    gnn_sub_attn)
      find "${WEIGHTS_DIR}" -type f -path "*${target}*" -name "gnn_sub_attn.pt" 2>/dev/null | sort | head -n 1
      ;;
  esac
}

run_tv() {
  local method="$1"
  local model="$2"
  local target="$3"
  local tag="$4"
  shift 4

  if [[ -z "${model}" || ! -f "${model}" ]]; then
    echo "[SKIP] ${tag} | ${target} | missing checkpoint"
    return 0
  fi

  local out_dir="${OUT_ROOT}/${tag}/${target}"
  mkdir -p "${out_dir}/heatmaps"

  echo "=== ${tag} | ${target} | ${method} ==="
  echo "    model: ${model}"
  python3 -m src.exps.target_validation \
    --method "${method}" \
    --model "${model}" \
    --target "${target}" \
    --targets_csv "${TARGETS_CSV}" \
    --out_csv "${out_dir}/predictions.csv" \
    --heatmap_dir "${out_dir}/heatmaps" \
    "$@"
}

echo "Targets CSV : ${TARGETS_CSV}"
echo "Targets     : ${TARGETS[*]}"
echo "Weights dir : ${WEIGHTS_DIR}"
echo "GNN weights : ${GNN_WEIGHTS:-<none>}"
echo "Output root : ${OUT_ROOT}"
echo

for target in "${TARGETS[@]}"; do
  linear_ckpt="$(resolve_ckpt "${target}" linear || true)"
  attn_ckpt="$(resolve_ckpt "${target}" attn_pool || true)"
  set_ckpt="$(resolve_ckpt "${target}" set_transformer || true)"
  gnn_sub_ckpt="$(resolve_ckpt "${target}" gnn_sub_attn || true)"

  # Method 1: Linear Baseline
  run_tv linear "${linear_ckpt}" "${target}" "linear_baseline"

  # Method 2: Substructure Attention (FP + attn_pool)
  run_tv attn "${attn_ckpt}" "${target}" "attn_pool"

  # Method 3: GNN + Substructure Attention
  if [[ -n "${gnn_sub_ckpt}" && -f "${gnn_sub_ckpt}" ]]; then
    if [[ -z "${GNN_WEIGHTS}" || ! -f "${GNN_WEIGHTS}" ]]; then
      echo "[SKIP] gnn_sub_attn | ${target} | missing GNN pretrained weights"
    else
      run_tv attn "${gnn_sub_ckpt}" "${target}" "gnn_sub_attn" --gnn_weights "${GNN_WEIGHTS}"
    fi
  else
    echo "[SKIP] gnn_sub_attn | ${target} | missing checkpoint"
  fi

  # Method 4: Set Transformer
  run_tv attn "${set_ckpt}" "${target}" "set_transformer"

  # Method 5a: Masking on attn_pool
  run_tv masking "${attn_ckpt}" "${target}" "masking_attn" --masking_predictor attn

  # Method 5b: Masking on linear
  run_tv masking "${linear_ckpt}" "${target}" "masking_linear" --masking_predictor linear
done

echo
echo "Done. Results under: ${OUT_ROOT}"
