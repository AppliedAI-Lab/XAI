"""
Method 1 inference: predict pIC50 with a fitted linear baseline and visualize
per-substructure coefficient contributions.
"""
import argparse
import os
import sys
from collections import defaultdict

import joblib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data.attn_dataset import get_substructure_features_fp
from src.models.masking import linear_per_fragment_importance
from src.exps.gnn_attn.infer import draw_ligand_heatmap
from src.exps.linear_baseline.train import aggregate_substructure_features


def load_linear_model(model_path):
    if not os.path.isfile(model_path):
        raise FileNotFoundError("Model file not found: " + model_path)
    return joblib.load(model_path)


def predict_and_explain(payload, smiles):
    fp_radius = payload.get("fp_radius", 2)
    fp_n_bits = payload.get("fp_n_bits", 2048)
    pooling = payload.get("pooling", "sum")
    features, labels, subs = get_substructure_features_fp(
        smiles,
        fp_radius=fp_radius,
        fp_n_bits=fp_n_bits,
    )
    if not features:
        return None, [], {}

    pooled = aggregate_substructure_features(features, pooling=pooling).reshape(1, -1)
    model = payload["model"]
    pred = float(model.predict(pooled)[0])
    reg = model.named_steps["reg"]
    scaler = model.named_steps.get("scaler")
    coef = payload.get("coef", reg.coef_)
    scaler_mean = getattr(scaler, "mean_", None)
    scaler_scale = getattr(scaler, "scale_", None)
    importances = linear_per_fragment_importance(
        coef, scaler_mean, scaler_scale, features, pooling=pooling
    )
    rows = [
        (sub.name, float(imp), sub.smiles, sub.atom_indices)
        for sub, imp in zip(subs, importances)
    ]
    atom_contrib = defaultdict(float)
    for _name, imp, _smi, atom_indices in rows:
        if not atom_indices:
            continue
        per_atom = imp / len(atom_indices)
        for atom_idx in atom_indices:
            atom_contrib[int(atom_idx)] += per_atom
    return pred, rows, dict(atom_contrib)


def main():
    parser = argparse.ArgumentParser(description="Linear baseline substructure inference")
    parser.add_argument("--smiles", required=True)
    parser.add_argument(
        "--model",
        default=os.path.join(ROOT, "weights", "linear_baseline", "grid", "FAAH", "linear_agg_ecfp_grid_FAAH.joblib"),
    )
    parser.add_argument("--out_heatmap", default=None)
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    try:
        payload = load_linear_model(args.model)
    except FileNotFoundError as exc:
        print(exc)
        sys.exit(1)

    pred, rows, atom_contrib = predict_and_explain(payload, args.smiles.strip())
    if pred is None:
        print("No substructures extracted for this SMILES.")
        sys.exit(1)

    print("SMILES:", args.smiles.strip())
    print("Predicted pIC50:", round(pred, 4))
    print("\nTop substructures by positive linear contribution:")
    for name, score, sub_smi, _atom_indices in sorted(rows, key=lambda x: -x[1])[: args.top_k]:
        print("  ", name, round(score, 6), sub_smi)
    print("\nTop substructures by negative linear contribution:")
    for name, score, sub_smi, _atom_indices in sorted(rows, key=lambda x: x[1])[: args.top_k]:
        print("  ", name, round(score, 6), sub_smi)

    out_path = args.out_heatmap or os.path.join(ROOT, "outputs", "linear_heatmap.png")
    draw_ligand_heatmap(args.smiles.strip(), atom_contrib, out_path)


if __name__ == "__main__":
    main()
