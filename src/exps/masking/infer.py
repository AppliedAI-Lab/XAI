"""
Method 5 inference: masking-based substructure attribution for a trained predictor.
"""
import argparse
import os
import sys
from collections import defaultdict

import joblib
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data.substructure_decompose import decompose_smiles
from src.data.attn_dataset import get_substructure_features_fp
from src.exps.gnn_attn.infer import draw_ligand_heatmap, load_attn_model, run_inference
from src.exps.linear_baseline.train import aggregate_substructure_features
from src.models.gnn_encoder import load_pretrained_gnn
from src.models.masking import masking_delta_pic50


def make_linear_predictor(model_path):
    payload = joblib.load(model_path)
    model = payload["model"]
    fp_radius = payload.get("fp_radius", 2)
    fp_n_bits = payload.get("fp_n_bits", 2048)
    pooling = payload.get("pooling", "sum")

    def predict(smiles):
        features, _labels, _subs = get_substructure_features_fp(
            smiles,
            fp_radius=fp_radius,
            fp_n_bits=fp_n_bits,
        )
        if not features:
            raise ValueError("No substructures extracted.")
        pooled = aggregate_substructure_features(features, pooling=pooling).reshape(1, -1)
        return float(model.predict(pooled)[0])

    return predict


def make_attn_predictor(model_path, gnn_weights=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_attn_model(model_path, device)
    gnn_model = None
    atom_dict = None
    if ckpt.get("encoder", "gnn") == "gnn":
        gnn_path = gnn_weights or ckpt.get("gnn_weights_path") or os.path.join(ROOT, "weights", "gnn_pretrain.pt")
        gnn_model, atom_dict = load_pretrained_gnn(gnn_path, device)

    def predict(smiles):
        pred, _contrib, _labels = run_inference(model, ckpt, smiles, gnn_model, atom_dict, device)
        if pred is None:
            raise ValueError("No substructures extracted.")
        return float(pred)

    return predict


def atom_contributions_from_masking(subs, rows):
    atom_contrib = defaultdict(float)
    for sub, (_name, delta, _masked_smiles) in zip(subs, rows):
        if not np.isfinite(delta) or not sub.atom_indices:
            continue
        per_atom = float(delta) / len(sub.atom_indices)
        for atom_idx in sub.atom_indices:
            atom_contrib[int(atom_idx)] += per_atom
    return dict(atom_contrib)


def main():
    parser = argparse.ArgumentParser(description="Masking-based substructure attribution")
    parser.add_argument("--smiles", required=True)
    parser.add_argument("--predictor", choices=["linear", "attn"], default="attn")
    parser.add_argument("--model", required=True)
    parser.add_argument("--gnn_weights", default=None)
    parser.add_argument("--out_heatmap", default=None)
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        print("Model not found:", args.model)
        sys.exit(1)

    predict_fn = (
        make_linear_predictor(args.model)
        if args.predictor == "linear"
        else make_attn_predictor(args.model, args.gnn_weights)
    )
    smiles = args.smiles.strip()
    subs = decompose_smiles(smiles)
    full_pred, rows = masking_delta_pic50(smiles, predict_fn, subs=subs)

    print("SMILES:", smiles)
    print("Predicted pIC50 full:", round(full_pred, 4))
    print("\nTop substructures by masking delta pIC50:")
    valid_rows = [row for row in rows if np.isfinite(row[1])]
    for name, delta, masked_smiles in sorted(valid_rows, key=lambda x: -abs(x[1]))[: args.top_k]:
        print("  ", name, round(delta, 6), masked_smiles)

    out_path = args.out_heatmap or os.path.join(ROOT, "outputs", "masking_heatmap.png")
    atom_contrib = atom_contributions_from_masking(subs, rows)
    draw_ligand_heatmap(smiles, atom_contrib, out_path)


if __name__ == "__main__":
    main()
