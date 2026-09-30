"""
Batch inference for README target-validation ligands.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data.substructure_decompose import decompose_smiles
from src.exps.dataset_utils import target_validation_frame
from src.exps.gnn_attn.infer import (
    atom_contributions_from_attention,
    atom_contributions_from_brics_fg,
    draw_ligand_heatmap,
    load_attn_model,
    run_inference,
)
from src.exps.linear_baseline.infer import load_linear_model, predict_and_explain
from src.exps.masking.infer import (
    atom_contributions_from_masking,
    make_attn_predictor,
    make_linear_predictor,
)
from src.models.gnn_encoder import load_pretrained_gnn
from src.models.masking import masking_delta_pic50


def safe_name(value):
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value))


def run_linear(payload, smiles):
    pred, rows, atom_contrib = predict_and_explain(payload, smiles)
    top = sorted(rows, key=lambda x: -abs(x[1]))[:5]
    return pred, [(name, score) for name, score, _smi, _atoms in top], atom_contrib


def run_attn(model, ckpt, smiles, gnn_model=None, atom_dict=None, device=None):
    pred, contrib, _labels = run_inference(model, ckpt, smiles, gnn_model, atom_dict, device)
    top = sorted(contrib, key=lambda x: -abs(x[1]))[:5]
    decompose = ckpt.get("decompose", "brics" if ckpt.get("encoder", "gnn") == "gnn" else "path")
    if decompose == "brics":
        atom_contrib = atom_contributions_from_brics_fg(smiles, contrib)
    else:
        atom_contrib = atom_contributions_from_attention(
            smiles,
            contrib,
            ckpt.get("min_len", 2),
            ckpt.get("max_len", 6),
        )
    return pred, top, atom_contrib


def run_masking(predict_fn, smiles):
    subs = decompose_smiles(smiles)
    pred, rows = masking_delta_pic50(smiles, predict_fn, subs=subs)
    valid = [(name, delta) for name, delta, _masked in rows if np.isfinite(delta)]
    top = sorted(valid, key=lambda x: -abs(x[1]))[:5]
    atom_contrib = atom_contributions_from_masking(subs, rows)
    return pred, top, atom_contrib


def main():
    parser = argparse.ArgumentParser(description="Run target-validation SMILES inference")
    parser.add_argument("--method", choices=["linear", "attn", "masking"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--target", default=None, help="Optional target filter: FAAH, CB2, PARP1, PBP2a, USP7")
    parser.add_argument(
        "--targets_csv",
        default=os.path.join(ROOT, "data", "targets.csv"),
        help="CSV with columns target,pdb_id,smiles",
    )
    parser.add_argument("--masking_predictor", choices=["linear", "attn"], default="attn")
    parser.add_argument("--gnn_weights", default=None)
    parser.add_argument("--out_csv", default=os.path.join(ROOT, "outputs", "target_validation_predictions.csv"))
    parser.add_argument("--heatmap_dir", default=os.path.join(ROOT, "outputs", "target_validation_heatmaps"))
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        print("Model not found:", args.model)
        sys.exit(1)

    df = target_validation_frame(args.target, csv_path=args.targets_csv)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    os.makedirs(args.heatmap_dir, exist_ok=True)

    linear_payload = None
    attn_model = None
    attn_ckpt = None
    gnn_model = None
    atom_dict = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.method == "linear":
        linear_payload = load_linear_model(args.model)
    elif args.method == "attn":
        attn_model, attn_ckpt = load_attn_model(args.model, device)
        if attn_ckpt.get("encoder", "gnn") == "gnn":
            gnn_path = args.gnn_weights or attn_ckpt.get("gnn_weights_path") or os.path.join(ROOT, "weights", "gnn_pretrain.pt")
            gnn_model, atom_dict = load_pretrained_gnn(gnn_path, device)
    elif args.masking_predictor == "linear":
        predict_fn = make_linear_predictor(args.model)
    else:
        predict_fn = make_attn_predictor(args.model, args.gnn_weights)

    out_rows = []
    for row in df.itertuples(index=False):
        heatmap_path = os.path.join(
            args.heatmap_dir,
            f"{safe_name(args.method)}_{safe_name(row.target)}_{safe_name(row.pdb_id)}.png",
        )
        try:
            if args.method == "linear":
                pred, top, atom_contrib = run_linear(linear_payload, row.smiles)
            elif args.method == "attn":
                pred, top, atom_contrib = run_attn(attn_model, attn_ckpt, row.smiles, gnn_model, atom_dict, device)
            else:
                pred, top, atom_contrib = run_masking(predict_fn, row.smiles)
            draw_ligand_heatmap(row.smiles, atom_contrib, heatmap_path)
        except Exception as exc:
            print(f"[SKIP] {row.target} {row.pdb_id}: {exc}")
            out_rows.append(
                {
                    "method": args.method,
                    "target": row.target,
                    "pdb_id": row.pdb_id,
                    "smiles": row.smiles,
                    "pred_pic50": np.nan,
                    "top_substructures": "",
                    "heatmap_path": "",
                    "error": str(exc),
                }
            )
            continue

        top_text = "; ".join(f"{name}:{score:.6f}" for name, score in top)
        out_rows.append(
            {
                "method": args.method,
                "target": row.target,
                "pdb_id": row.pdb_id,
                "smiles": row.smiles,
                "pred_pic50": pred,
                "top_substructures": top_text,
                "heatmap_path": heatmap_path,
                "error": "",
            }
        )
        print(row.target, row.pdb_id, "pIC50=", round(float(pred), 4))

    pd.DataFrame(out_rows).to_csv(args.out_csv, index=False)
    print("Saved target-validation predictions to:", args.out_csv)


if __name__ == "__main__":
    main()
