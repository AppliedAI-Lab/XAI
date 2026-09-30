"""
Method 3 — Frozen GNN node embeddings, mean-pool to substructures (BRICS + scaffold + FG),
attention pooling + MLP. (End-to-end GNN fine-tuning would require on-the-fly featurization per step.)
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data.attn_dataset import (
    build_padded_fragment_tensor,
    get_substructure_features_gnn,
)
from src.data.fragments import load_faah_csv
from src.exps.dataset_utils import load_split_frames, start_training_log
from src.models.attention_regression import AttentionPoolingRegression
from src.models.gnn_encoder import load_pretrained_gnn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="FAAH", help="Dataset folder under data/datasets")
    p.add_argument("--csv", default=None)
    p.add_argument("--train_csv", default=None)
    p.add_argument("--test_csv", default=None)
    p.add_argument("--save_dir", default=None)
    p.add_argument("--gnn_weights", default=None)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    save_dir = args.save_dir or os.path.join(ROOT, "weights", args.dataset, "gnn_sub_attn")
    os.makedirs(save_dir, exist_ok=True)
    log_path = start_training_log(save_dir)
    gnn_path = args.gnn_weights or os.path.join(ROOT, "weights", "gnn_pretrain.pt")
    if not os.path.isfile(gnn_path):
        print("GNN checkpoint not found:", gnn_path)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    split_boundary = None
    if args.csv:
        if not os.path.isfile(args.csv):
            print("Data file not found:", args.csv)
            sys.exit(1)
        df = load_faah_csv(args.csv)
        print("Using random split from:", args.csv)
    else:
        try:
            df_train, df_test, train_path, test_path = load_split_frames(
                load_faah_csv,
                dataset=args.dataset,
                train_csv=args.train_csv,
                test_csv=args.test_csv,
            )
        except FileNotFoundError as exc:
            print(exc)
            sys.exit(1)
        split_boundary = len(df_train)
        df = pd.concat([df_train, df_test], ignore_index=True)
        print("Train CSV:", train_path)
        print("Test CSV:", test_path)

    gnn_model, atom_dict = load_pretrained_gnn(gnn_path, device)
    gnn_model.eval()

    features_list = []
    labels_list = []
    for smi in tqdm(df["smiles"], desc="GNN substructure features"):
        feats, labels, _subs = get_substructure_features_gnn(smi, gnn_model, atom_dict, device)
        features_list.append([np.asarray(f, dtype=np.float32) for f in feats])
        labels_list.append(labels)

    valid = [i for i in range(len(features_list)) if len(features_list[i]) > 0]
    if not valid:
        print("No valid molecules.")
        sys.exit(1)
    df = df.iloc[valid].reset_index(drop=True)
    features_list = [features_list[i] for i in valid]
    labels_list = [labels_list[i] for i in valid]

    if split_boundary is None:
        idx = np.arange(len(df))
        train_idx, test_idx = train_test_split(idx, test_size=args.test_size, random_state=args.seed)
    else:
        train_idx = np.asarray([new_i for new_i, old_i in enumerate(valid) if old_i < split_boundary])
        test_idx = np.asarray([new_i for new_i, old_i in enumerate(valid) if old_i >= split_boundary])
    if len(train_idx) == 0 or len(test_idx) == 0:
        print("Train/test split has no valid molecules after decomposition.")
        sys.exit(1)

    all_feats = np.concatenate([np.stack(features_list[i]) for i in train_idx], axis=0)
    feature_mean = all_feats.mean(axis=0, keepdims=True).astype(np.float32)
    feature_std = all_feats.std(axis=0, keepdims=True).astype(np.float32) + 1e-6
    for i, feats in enumerate(features_list):
        arr = np.stack(feats)
        arr = (arr - feature_mean) / feature_std
        features_list[i] = [arr[j] for j in range(arr.shape[0])]

    X_t, mask_t, _ = build_padded_fragment_tensor(features_list, labels_list, device)
    fragment_dim = gnn_model.dim

    y = df["pic50"].values.astype(np.float32)
    y_mean = float(y[train_idx].mean())
    y_std = float(y[train_idx].std() + 1e-6)
    y_norm = (y - y_mean) / y_std
    y_t = torch.FloatTensor(y_norm).to(device).unsqueeze(1)

    model = AttentionPoolingRegression(
        fragment_dim=fragment_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    save_path = os.path.join(save_dir, "gnn_sub_attn.pt")
    best_r2 = -float("inf")
    ckpt = {
        "state_dict": model.state_dict(),
        "model_type": "gnn_sub_attn_pool",
        "fragment_dim": fragment_dim,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "gnn_weights_path": gnn_path,
        "atom_dict": atom_dict,
        "N_atoms": gnn_model.N_atoms,
        "dim": gnn_model.dim,
        "layer_hidden": gnn_model.layer_hidden,
        "layer_output": gnn_model.layer_output,
        "dataset": args.dataset,
        "train_csv": args.train_csv,
        "test_csv": args.test_csv,
        "training_log": log_path,
    }

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = np.random.permutation(len(train_idx))
        loss_sum = 0.0
        n_batches = 0
        for start in range(0, len(perm), args.batch_size):
            batch_idx = train_idx[perm[start : start + args.batch_size]]
            x_b = X_t[batch_idx]
            m_b = mask_t[batch_idx]
            y_b = y_t[batch_idx]
            pred, _alpha = model(x_b, m_b)
            loss = F.mse_loss(pred, y_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            pred_te, _ = model(X_t[test_idx], mask_t[test_idx])
            r2_test = 1 - F.mse_loss(pred_te, y_t[test_idx]).item() / (y_t[test_idx].var().item() + 1e-8)
        if r2_test > best_r2:
            best_r2 = r2_test
            ckpt["state_dict"] = model.state_dict()
            torch.save(ckpt, save_path)
        if epoch % 20 == 0 or epoch == 1:
            print("Epoch", epoch, "MSE train:", loss_sum / max(n_batches, 1), "R2 test:", r2_test)

    print("Best test R2 (approx):", best_r2)
    print("Saved:", save_path)


if __name__ == "__main__":
    main()
