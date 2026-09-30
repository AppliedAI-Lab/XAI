# Target-Dependent Explainability in Molecular Activity Prediction

**A Structure-Based Benchmark for Target-Specific Model Selection**

Nhat-Hai Nguyen<sup>a</sup>, Manh-Tu Luong<sup>a</sup>, Khanh Huyen Thi Pham<sup>b</sup>, Thi-Thu Nguyen<sup>a,*</sup>, Tan Khanh Nguyen<sup>c,d,*</sup>

Code companion for the paper above: substructure-level pIC50 prediction and explainability across five protein targets, with a shared ligand decomposition and multiple attribution schemes for **target-specific** model selection.

---

## Overview

Whole-molecule QSAR scores do not expose which chemical motifs drive activity. This repository decomposes each ligand into overlapping substructure **views**, encodes them, aggregates them into a ligand-level prediction, and estimates motif importance. Because dataset size and chemistry differ by target, **no single architecture is assumed best for every target**—the benchmark is meant to guide per-target choice of predictor and attribution method.

**Shared pipeline**

```text
SMILES
  -> RDKit graph
  -> BRICS + Bemis–Murcko scaffold + SMARTS functional groups
  -> substructure embeddings (ECFP or GNN-pooled)
  -> aggregation / attention / Set Transformer
  -> predicted pIC50
  -> importance (linear coeffs | attention | masking)
```

Exact atom-set duplicates are dropped; partial overlaps are kept so each named view (scaffold, BRICS piece, functional group) remains interpretable when projected onto atoms.

---

## Methods

| ID | Name | Representation | Aggregation | Importance |
|----|------|----------------|-------------|------------|
| M1 | Linear baseline | ECFP per substructure | Sum / mean / max (concat) + Ridge / Lasso / Elastic Net | Coefficient × embedding |
| M2 | Substructure attention (FP) | ECFP per substructure | Attention pooling + MLP | Attention weight × embedding norm |
| M3 | GNN + substructure attention | Frozen pretrained GNN atom embeddings → motif mean-pool | Attention pooling + MLP | Attention weights |
| M4 | Set Transformer | ECFP per substructure | Set Transformer (self-attention over motifs) + MLP | Multi-head attention (averaged) |
| M5 | Masking | Any trained M1–M4 predictor | — (perturbation) | ΔpIC50 after removing motif *i* |

Masking (M5) has no trainable parameters; it wraps a trained predictor and reports signed prediction change after motif deletion.

---

## Datasets

Pre-split CSVs live under `data/datasets/{NAME}/{NAME}_{train,test}.csv` (`smiles`, `pIC50`).

| Dataset | Train | Test | Total |
|---------|------:|-----:|------:|
| FAAH | 1075 | 120 | 1195 |
| CB2 | 3511 | 391 | 3902 |
| PARP1 | 2208 | 246 | 2454 |
| PBP2a | 80 | 9 | 89 |
| USP7 | 458 | 51 | 509 |

Optional: `FAAH_augmented` (randomized SMILES augmentation). Crystal-structure ligands used for qualitative target validation are listed in `data/targets.csv` (PDB ID + SMILES).

---

## Setup

```bash
pip install -r requirements.txt
```

Main dependencies: `pandas`, `scikit-learn`, `torch`, `rdkit`, `joblib`, `matplotlib`, `scipy`.

Directory layout (high level):

```text
data/           # splits, raw CSVs, target-validation SMILES
src/
  data/         # decomposition + datasets
  models/       # linear / attention / GNN / masking
  exps/         # train, infer, target_validation scripts
  tools/        # split, augment, GNN pretrain helpers
weights/        # checkpoints (gitignored)
outputs/        # heatmaps / metrics (gitignored)
```

---

## Training

Default dataset list for bash runners: `FAAH FAAH_augmented CB2 PARP1 PBP2a USP7`. Restrict with `DATASETS="FAAH CB2 PARP1"`.

### M1 — Linear baseline

```bash
bash src/exps/linear_baseline/train.sh
```

Grid search (default): regressor ∈ {ridge, lasso, elasticnet}, pooling ∈ {sum_mean, sum_mean_max}, ECFP radius / bits. Single config:

```bash
GRID_SEARCH=0 DATASETS="FAAH" REGRESSOR=ridge POOLING=sum_mean_max \
  bash src/exps/linear_baseline/train.sh
```

Or:

```bash
python3 -m src.exps.linear_baseline.train --dataset FAAH --grid_search
```

Checkpoints: `weights/linear_baseline/grid/{DATASET}/`.

### M2 / M4 — FP attention & Set Transformer

Train both (script default):

```bash
bash src/exps/gnn_attn/train.sh
```

Attention only:

```bash
python3 -m src.exps.gnn_attn.train \
  --dataset FAAH --encoder fp --decompose brics --model_type attn_pool
```

Set Transformer only:

```bash
python3 -m src.exps.gnn_attn.train \
  --dataset FAAH --encoder fp --decompose brics --model_type set_transformer
```

### M3 — GNN + substructure attention

Requires a pretrained GNN encoder (`weights/gnn/gnn_pretrain.pt`, or set `GNN_WEIGHTS`).

```bash
bash src/exps/gnn_sub_attn/train.sh
# or
python3 -m src.exps.gnn_sub_attn.train \
  --dataset FAAH --gnn_weights weights/gnn/gnn_pretrain.pt
```

Useful overrides:

```bash
DATASETS="FAAH USP7" EPOCHS=300 BATCH_SIZE=16 bash src/exps/gnn_attn/train.sh
GNN_WEIGHTS=weights/gnn/gnn_pretrain.pt DATASETS="PARP1" bash src/exps/gnn_sub_attn/train.sh
```

More detail: [`experiments.md`](experiments.md).

---

## Inference & heatmaps

### Linear (M1)

```bash
python3 -m src.exps.linear_baseline.infer \
  --model weights/linear_baseline/grid/FAAH/linear_agg_ecfp_grid_FAAH.joblib \
  --smiles "CCO" \
  --out_heatmap outputs/linear_heatmap.png
```

### Attention / Set Transformer / GNN-sub-attn (M2–M4)

```bash
python3 -m src.exps.gnn_attn.infer \
  --model weights/FAAH/attn/attn_model_fp_brics_attn_pool.pt \
  --smiles "CCO" \
  --out_heatmap outputs/attn_pool_heatmap.png
```

For M3, also pass `--gnn_weights weights/gnn/gnn_pretrain.pt` and the corresponding `gnn_sub_attn` checkpoint.

### Masking (M5)

```bash
python3 -m src.exps.masking.infer \
  --predictor attn \
  --model weights/FAAH/attn/attn_model_fp_brics_attn_pool.pt \
  --smiles "CCO" \
  --out_heatmap outputs/masking_attn_heatmap.png

python3 -m src.exps.masking.infer \
  --predictor linear \
  --model weights/linear_baseline/grid/FAAH/linear_agg_ecfp_grid_FAAH.joblib \
  --smiles "CCO" \
  --out_heatmap outputs/masking_linear_heatmap.png
```

### Target validation (PDB ligands)

```bash
python3 -m src.exps.target_validation \
  --method attn \
  --model weights/FAAH/attn/attn_model_fp_brics_attn_pool.pt \
  --target FAAH \
  --out_csv outputs/target_validation_FAAH_attn.csv \
  --heatmap_dir outputs/target_validation_FAAH_attn_heatmaps
```

`--method` supports `linear`, `attn`, `masking` (with `--masking_predictor`), etc. Batch helper: `bash src/exps/validation.sh`.

---

## Citation

If you use this code or benchmark, please cite the paper:

```bibtex
@article{nguyen202Xtarget,
  title   = {Target-Dependent Explainability in Molecular Activity Prediction:
             A Structure-Based Benchmark for Target-Specific Model Selection},
  author  = {Nguyen, Nhat-Hai and Luong, Manh-Tu and Pham, Khanh Huyen Thi
             and Nguyen, Thi-Thu and Nguyen, Tan Khanh},
  year    = {202X},
  note    = {Update venue / year / DOI when published}
}
```

**Key prior work used in the methodology** (full BibTeX in [`references.bib`](references.bib)):

1. Rogers & Hahn, *Extended-Connectivity Fingerprints*, J. Chem. Inf. Model. (2010).
2. Gilmer et al., *Neural Message Passing for Quantum Chemistry*, ICML (2017).
3. Degen et al., *On the Art of Compiling and Using “Drug-Like” Chemical Fragment Spaces* (BRICS), J. Chem. Inf. Model. (2008).
4. Bemis & Murcko, *The Properties of Known Drugs. 1. Molecular Frameworks*, J. Med. Chem. (1996).
5. Lee et al., *Set Transformer*, ICML (2019).
6. Jain & Wallace, *Attention is not Explanation*, NAACL (2019).

---

## License / contact

For questions related to the paper or code, contact the corresponding authors marked with \* in the author list. Update this section with institutional affiliations and license terms before public release.
