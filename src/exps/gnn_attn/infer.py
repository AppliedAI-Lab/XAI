"""
Load saved attention model and explain a ligand: predict pIC50 and show attention map
(contribution of each fragment to the prediction). Optionally draw heatmap on 2D structure.
"""
import os
import sys
import argparse
import numpy as np
import torch
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data.fragments import fragment_paths_with_indices
from src.data.substructure_decompose import decompose_smiles
from src.data.attn_dataset import (
    get_fragment_features_fp,
    get_fragment_features_gnn,
    get_substructure_features_fp,
    get_substructure_features_gnn,
    build_padded_fragment_tensor,
)
from src.models.attention_regression import (
    SubstructureAttentionRegression,
    AttentionPoolingRegression,
    SetTransformerRegression,
)
from src.models.gnn_encoder import load_pretrained_gnn


def load_attn_model(save_path, device=None):
    """Load checkpoint and rebuild model."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(save_path, map_location=device, weights_only=False)
    mt = ckpt.get("model_type", "mha")
    fd = ckpt["fragment_dim"]
    hd = ckpt["hidden_dim"]
    nh = ckpt.get("num_heads", 4)
    do = ckpt.get("dropout", 0.1)
    nel = ckpt.get("num_encoder_layers", 2)
    if mt == "mha":
        model = SubstructureAttentionRegression(
            fragment_dim=fd,
            hidden_dim=hd,
            num_heads=nh,
            dropout=do,
        ).to(device)
    elif mt == "attn_pool":
        model = AttentionPoolingRegression(
            fragment_dim=fd,
            hidden_dim=hd,
            dropout=do,
        ).to(device)
    elif mt == "set_transformer":
        model = SetTransformerRegression(
            fragment_dim=fd,
            hidden_dim=hd,
            num_heads=nh,
            num_encoder_layers=nel,
            dropout=do,
        ).to(device)
    elif mt == "gnn_sub_attn_pool":
        model = AttentionPoolingRegression(
            fragment_dim=fd,
            hidden_dim=hd,
            dropout=do,
        ).to(device)
    else:
        raise ValueError("Unknown model_type in checkpoint: " + str(mt))
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return model, ckpt


def get_features_and_labels(
    smiles,
    encoder,
    gnn_model=None,
    atom_dict=None,
    device=None,
    min_len=2,
    max_len=6,
    decompose="path",
):
    """Get fragment features and labels for one SMILES."""
    if device is None:
        device = torch.device("cpu")
    if decompose == "path":
        if encoder == "fp":
            features, labels = get_fragment_features_fp(smiles, min_len=min_len, max_len=max_len)
        else:
            features, labels = get_fragment_features_gnn(smiles, gnn_model, atom_dict, device, min_len, max_len)
    else:
        if encoder == "fp":
            features, labels, _subs = get_substructure_features_fp(smiles)
        else:
            features, labels, _subs = get_substructure_features_gnn(smiles, gnn_model, atom_dict, device)
    return features, labels


def run_inference(model, ckpt, smiles, gnn_model=None, atom_dict=None, device=None):
    """
    Predict pIC50 and return attention weights per fragment for one SMILES.
    Returns: pred (float), attention_weights (list of (fragment_label, weight)).
    """
    if device is None:
        device = next(model.parameters()).device
    encoder = ckpt.get("encoder", "gnn")
    decompose = ckpt.get("decompose", "brics")
    min_len = ckpt.get("min_len", 2)
    max_len = ckpt.get("max_len", 6)

    features, labels = get_features_and_labels(
        smiles, encoder, gnn_model, atom_dict, device, min_len, max_len, decompose=decompose
    )
    if not features:
        return None, [], []

    # Apply same feature normalization as used during training, if present
    feature_mean = ckpt.get("feature_mean")
    feature_std = ckpt.get("feature_std")
    if feature_mean is not None and feature_std is not None:
        # Ensure feature normalization keeps each fragment feature as a 1D vector
        fm = np.asarray(feature_mean, dtype=np.float32).reshape(-1)
        fs = np.asarray(feature_std, dtype=np.float32).reshape(-1)
        features = [
            ((np.asarray(f, dtype=np.float32).reshape(-1) - fm) / fs).astype(np.float32)
            for f in features
        ]

    features_list = [features]
    labels_list = [labels]
    X_t, mask_t, all_labels = build_padded_fragment_tensor(features_list, labels_list, device)
    with torch.no_grad():
        pred, attn = model(X_t, mask_t)
    pred_val = pred[0, 0].item()

    # Undo target normalization if training used normalized pIC50
    y_mean = ckpt.get("y_mean")
    y_std = ckpt.get("y_std")
    if y_mean is not None and y_std is not None:
        pred_val = pred_val * float(y_std) + float(y_mean)
    attn_np = attn[0].cpu().numpy()
    fragment_contrib = [(labels[i], float(attn_np[i])) for i in range(len(labels))]
    return pred_val, fragment_contrib, labels


def atom_contributions_from_attention(smiles, fragment_contrib, min_len=2, max_len=6):
    """
    Map fragment attention weights to per-atom contributions (for heatmap).
    """
    atom_contrib = defaultdict(float)
    contrib_dict = dict(fragment_contrib)
    for path_key, path_indices in fragment_paths_with_indices(smiles, min_len, max_len):
        w = contrib_dict.get(path_key, 0.0)
        per_atom = w / len(path_indices) if path_indices else 0
        for idx in path_indices:
            atom_contrib[idx] += per_atom
    return dict(atom_contrib)


def atom_contributions_from_brics_fg(smiles, fragment_contrib):
    """Map attention weights to atoms using BRICS+scaffold+FG decomposition."""
    atom_contrib = defaultdict(float)
    contrib_dict = dict(fragment_contrib)
    for s in decompose_smiles(smiles):
        w = contrib_dict.get(s.name, 0.0)
        n = len(s.atom_indices) if s.atom_indices else 1
        per_atom = w / n
        for idx in s.atom_indices:
            atom_contrib[idx] += per_atom
    return dict(atom_contrib)


def draw_ligand_heatmap(smiles, atom_contrib, out_path, size=(500, 500)):
    """Draw 2D structure with atoms colored by attention contribution."""
    from rdkit import Chem
    from rdkit.Chem import Draw
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES for drawing: " + str(smiles))
    Chem.rdDepictor.Compute2DCoords(mol)
    try:
        import matplotlib
    except ImportError:
        raise ImportError("matplotlib required for heatmap. pip install matplotlib")
    values = np.array([atom_contrib.get(i, 0.0) for i in range(mol.GetNumAtoms())])
    if values.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = values.min(), values.max()
        if vmax <= vmin:
            vmax = vmin + 1e-9
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    # Use modern colormap API to avoid deprecation warnings
    cmap = matplotlib.colormaps.get_cmap("YlOrRd")
    highlight_atoms = list(range(mol.GetNumAtoms()))
    highlight_colors = {}
    for i in range(mol.GetNumAtoms()):
        v = atom_contrib.get(i, 0.0)
        r, g, b, _ = cmap(norm(v))
        highlight_colors[i] = (float(r), float(g), float(b))
    d2d = Draw.MolDraw2DCairo(size[0], size[1])
    opts = d2d.drawOptions()
    try:
        opts.useBWAtomPalette()
    except Exception:
        pass
    d2d.DrawMolecule(mol, highlightAtoms=highlight_atoms, highlightAtomColors=highlight_colors)
    d2d.FinishDrawing()
    png = d2d.GetDrawingText()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(png)
    print("Heatmap saved to:", out_path)


def main():
    p = argparse.ArgumentParser(description="Attention-based fragment contribution and heatmap")
    p.add_argument("--smiles", required=True, help="SMILES of the molecule")
    p.add_argument("--model", default="weights/attn/attn_model_gnn.pt", help="Path to attn_model.pt")
    p.add_argument("--gnn_weights", default=None)
    p.add_argument("--out_heatmap", default=None, help="Output PNG path for heatmap")
    p.add_argument("--top_k", type=int, default=10)
    args = p.parse_args()

    if args.model is None:
        args.model = os.path.join(ROOT, "weights", "attn", "attn_model_fp.pt")
    if not os.path.isfile(args.model):
        print("Model not found:", args.model)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_attn_model(args.model, device)

    gnn_model = None
    atom_dict = None
    if ckpt.get("encoder", "gnn") == "gnn":
        gnn_path = args.gnn_weights or ckpt.get("gnn_weights_path") or os.path.join(ROOT, "weights", "gnn_pretrain.pt")
        if os.path.isfile(gnn_path):
            gnn_model, atom_dict = load_pretrained_gnn(gnn_path, device)
        else:
            print("GNN weights not found:", gnn_path)
            sys.exit(1)

    smi = args.smiles.strip()
    pred_val, fragment_contrib, labels = run_inference(
        model, ckpt, smi, gnn_model, atom_dict, device
    )
    if pred_val is None:
        print("No fragments extracted for this SMILES.")
        sys.exit(1)

    print("SMILES:", smi)
    print("Predicted pIC50:", round(pred_val, 4))
    # Aggregate by fragment type (same label can appear multiple times at different positions)
    type_to_weights = defaultdict(list)
    for lab, w in fragment_contrib:
        type_to_weights[lab].append(w)
    type_to_total = [(lab, sum(weights), len(weights)) for lab, weights in type_to_weights.items()]
    sorted_by_total = sorted(type_to_total, key=lambda x: -x[1])
    print("\nTop fragment types by total attention weight (contribution to prediction):")
    for lab, total_w, count in sorted_by_total[: args.top_k]:
        suffix = f" ({count} occurrences)" if count > 1 else ""
        print("  ", lab, ":", round(total_w, 6), suffix)

    if args.out_heatmap or (fragment_contrib and not args.out_heatmap):
        out_path = args.out_heatmap or os.path.join(ROOT, "outputs", "attn_heatmap.png")
        min_len = ckpt.get("min_len", 2)
        max_len = ckpt.get("max_len", 6)
        decompose = ckpt.get("decompose", "path")
        if decompose == "brics":
            print("Drawing heatmap (BRICS/scaffold/FG substructures)")
            atom_contrib = atom_contributions_from_brics_fg(smi, fragment_contrib)
        else:
            print("Drawing heatmap for fragments of length", min_len, "to", max_len)
            atom_contrib = atom_contributions_from_attention(smi, fragment_contrib, min_len, max_len)
        draw_ligand_heatmap(smi, atom_contrib, out_path)


if __name__ == "__main__":
    main()
