# Experiment Guide

This guide describes how to train, validate, and run inference for the five
substructure-importance methods. All training commands use the pre-created
splits under `data/datasets/{dataset}/{dataset}_train.csv` and
`data/datasets/{dataset}/{dataset}_test.csv`.

## Datasets

Default dataset names used by the scripts:

```bash
FAAH FAAH_augmented CB2 PARP1 PBP2a USP7
```

To train only a subset, set `DATASETS`:

```bash
DATASETS="FAAH CB2 PARP1" bash src/exps/linear_baseline/train.sh
```

Training logs are saved next to checkpoints as `training.log`.

## Method 1: Linear Baseline

Architecture:

```text
Ligand
-> BRICS + Murcko scaffold + functional groups
-> ECFP per substructure
-> aggregation pooling (default: concat sum/mean/max)
-> Ridge/Lasso/ElasticNet regression
-> pIC50
```

Train all datasets:

```bash
bash src/exps/linear_baseline/train.sh
```

By default, this runs grid search over:

```text
regressor: ridge, lasso, elasticnet
pooling: sum_mean, sum_mean_max
fp_radius: 2, 3
fp_bits: 2048, 4096
```

Train one dataset manually:

```bash
python3 -m src.exps.linear_baseline.train \
  --dataset FAAH \
  --grid_search
```

Train one fixed config without grid search:

```bash
GRID_SEARCH=0 DATASETS="FAAH" REGRESSOR=ridge POOLING=sum_mean_max \
  bash src/exps/linear_baseline/train.sh
```

Checkpoint and log:

```text
weights/linear_baseline/grid/FAAH/linear_agg_ecfp_grid_FAAH.joblib
weights/linear_baseline/grid/FAAH/grid_results.csv
weights/linear_baseline/grid/FAAH/training.log
```

Infer one new SMILES and save heatmap:

```bash
python3 -m src.exps.linear_baseline.infer \
  --model weights/linear_baseline/grid/FAAH/linear_agg_ecfp_grid_FAAH.joblib \
  --smiles "CCO" \
  --out_heatmap outputs/linear_heatmap.png
```

## Method 2: Substructure Attention (FP-Based)

Architecture:

```text
Ligand
-> BRICS + Murcko scaffold + functional groups
-> ECFP per substructure
-> attention pooling
-> MLP regression
-> pIC50
```

Train all datasets:

```bash
bash src/exps/gnn_attn/train.sh
```

The script above trains both Method 2 and Method 4. To train only Method 2:

```bash
python3 -m src.exps.gnn_attn.train \
  --dataset FAAH \
  --encoder fp \
  --decompose brics \
  --model_type attn_pool
```

Checkpoint and log:

```text
weights/FAAH/attn/attn_model_fp_brics_attn_pool.pt
weights/FAAH/attn/training.log
```

Infer one new SMILES and save heatmap:

```bash
python3 -m src.exps.gnn_attn.infer \
  --model weights/FAAH/attn/attn_model_fp_brics_attn_pool.pt \
  --smiles "CCO" \
  --out_heatmap outputs/attn_pool_heatmap.png
```

## Method 3: GNN + Substructure Attention

Architecture:

```text
Ligand graph
-> pretrained/frozen GNN atom embeddings
-> mean-pool nodes into BRICS/scaffold/functional-group embeddings
-> attention pooling
-> MLP regression
-> pIC50
```

This method requires `weights/gnn/gnn_pretrain.pt`. If the checkpoint is elsewhere,
set `GNN_WEIGHTS`.

Train all datasets:

```bash
bash src/exps/gnn_sub_attn/train.sh
```

Train one dataset manually:

```bash
python3 -m src.exps.gnn_sub_attn.train \
  --dataset FAAH \
  --gnn_weights weights/gnn/gnn_pretrain.pt
```

Checkpoint and log:

```text
weights/FAAH/gnn_sub_attn/gnn_sub_attn.pt
weights/FAAH/gnn_sub_attn/training.log
```

Infer one new SMILES and save heatmap:

```bash
python3 -m src.exps.gnn_attn.infer \
  --model weights/FAAH/gnn_sub_attn/gnn_sub_attn.pt \
  --gnn_weights weights/gnn/gnn_pretrain.pt \
  --smiles "CCO" \
  --out_heatmap outputs/gnn_sub_attn_heatmap.png
```

## Method 4: Set Transformer

Architecture:

```text
Ligand
-> BRICS + Murcko scaffold + functional groups
-> ECFP per substructure
-> Set Transformer encoder
-> attention pooling
-> MLP regression
-> pIC50
```

Train all datasets:

```bash
bash src/exps/gnn_attn/train.sh
```

The script above trains both Method 2 and Method 4. To train only Method 4:

```bash
python3 -m src.exps.gnn_attn.train \
  --dataset FAAH \
  --encoder fp \
  --decompose brics \
  --model_type set_transformer
```

Checkpoint and log:

```text
weights/FAAH/attn/attn_model_fp_brics_set_transformer.pt
weights/FAAH/attn/training.log
```

Infer one new SMILES and save heatmap:

```bash
python3 -m src.exps.gnn_attn.infer \
  --model weights/FAAH/attn/attn_model_fp_brics_set_transformer.pt \
  --smiles "CCO" \
  --out_heatmap outputs/set_transformer_heatmap.png
```

## Method 5: Masking Attribution

Architecture:

```text
Ligand
-> predict pIC50_full with a trained base predictor
-> remove one substructure at a time
-> predict pIC50_masked
-> importance = pIC50_full - pIC50_masked
```

Masking has no trainable parameters. The `train.sh` in this method directory is
a reminder script:

```bash
bash src/exps/masking/train.sh
```

Run masking with an attention checkpoint:

```bash
python3 -m src.exps.masking.infer \
  --predictor attn \
  --model weights/FAAH/attn/attn_model_fp_brics_attn_pool.pt \
  --smiles "CCO" \
  --out_heatmap outputs/masking_attn_heatmap.png
```

Run masking with a linear checkpoint:

```bash
python3 -m src.exps.masking.infer \
  --predictor linear \
  --model weights/linear_baseline/grid/FAAH/linear_agg_ecfp_grid_FAAH.joblib \
  --smiles "CCO" \
  --out_heatmap outputs/masking_linear_heatmap.png
```

## Target Validation

Run README target-validation SMILES for one method and target:

```bash
python3 -m src.exps.target_validation \
  --method attn \
  --model weights/FAAH/attn/attn_model_fp_brics_attn_pool.pt \
  --target FAAH \
  --out_csv outputs/target_validation_FAAH_attn.csv \
  --heatmap_dir outputs/target_validation_FAAH_attn_heatmaps
```

For linear:

```bash
python3 -m src.exps.target_validation \
  --method linear \
  --model weights/linear_baseline/grid/FAAH/linear_agg_ecfp_grid_FAAH.joblib \
  --target FAAH
```

For masking:

```bash
python3 -m src.exps.target_validation \
  --method masking \
  --masking_predictor attn \
  --model weights/FAAH/attn/attn_model_fp_brics_attn_pool.pt \
  --target FAAH
```

## Useful Overrides

The bash scripts accept environment overrides:

```bash
DATASETS="FAAH USP7" EPOCHS=300 BATCH_SIZE=16 bash src/exps/gnn_attn/train.sh
DATASETS="CB2" GRID_SEARCH=1 bash src/exps/linear_baseline/train.sh
GRID_SEARCH=0 DATASETS="CB2" REGRESSOR=lasso POOLING=sum_mean_max bash src/exps/linear_baseline/train.sh
GNN_WEIGHTS=weights/gnn/gnn_pretrain.pt DATASETS="PARP1" bash src/exps/gnn_sub_attn/train.sh
```
