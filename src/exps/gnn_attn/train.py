"""
Train attention regression over fragment features (fingerprint or GNN).
Saves model and config for infer; attention map is available at inference.
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data.fragments import load_faah_csv
from src.data.attn_dataset import (
    get_fragment_features_fp,
    get_fragment_features_gnn,
    get_substructure_features_fp,
    get_substructure_features_gnn,
    build_padded_fragment_tensor,
    FP_NBITS,
)
from src.models.attention_regression import (
    SubstructureAttentionRegression,
    AttentionPoolingRegression,
    SetTransformerRegression,
)
from src.exps.dataset_utils import load_split_frames, start_training_log
from src.models.gnn_encoder import load_pretrained_gnn


def main(
    dataset="FAAH",
    csv_path=None,
    train_csv=None,
    test_csv=None,
    save_dir=None,
    encoder="fp",
    decompose="path",
    model_type="mha",
    gnn_weights_path=None,
    min_len=2,
    max_len=6,
    hidden_dim=128,
    num_heads=4,
    num_encoder_layers=2,
    dropout=0.1,
    lr=1e-3,
    epochs=100,
    batch_size=32,
    test_size=0.2,
    random_state=42,
):
    if save_dir is None:
        save_dir = os.path.join(ROOT, "weights", dataset, "attn")
    os.makedirs(save_dir, exist_ok=True)
    log_path = start_training_log(save_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_boundary = None
    if csv_path is not None:
        if not os.path.isfile(csv_path):
            print("Data file not found:", csv_path)
            sys.exit(1)
        df = load_faah_csv(csv_path)
        print("Using random split from:", csv_path)
    else:
        try:
            df_train, df_test, train_path, test_path = load_split_frames(
                load_faah_csv,
                dataset=dataset,
                train_csv=train_csv,
                test_csv=test_csv,
            )
        except FileNotFoundError as exc:
            print(exc)
            sys.exit(1)
        split_boundary = len(df_train)
        df = pd.concat([df_train, df_test], ignore_index=True)
        print("Train CSV:", train_path)
        print("Test CSV:", test_path)

    # Build fragment / substructure features per molecule
    features_list = []
    labels_list = []
    if encoder == "fp":
        if decompose == "path":
            print("Building path-fragment fingerprint features...")
            for smi in tqdm(df["smiles"], desc="FP path fragments"):
                feats, labels = get_fragment_features_fp(smi, min_len=min_len, max_len=max_len)
                features_list.append(feats)
                labels_list.append(labels)
        else:
            print("Building BRICS+scaffold+FG fingerprint features...")
            for smi in tqdm(df["smiles"], desc="FP BRICS/FG"):
                feats, labels, _subs = get_substructure_features_fp(smi)
                features_list.append(feats)
                labels_list.append(labels)
        fragment_dim = FP_NBITS
    else:
        print("Building features with GNN encoder...")
        if gnn_weights_path is None:
            gnn_weights_path = os.path.join(ROOT, "weights", "gnn_pretrain.pt")
        if not os.path.isfile(gnn_weights_path):
            print("GNN weights not found. Run gnn_pretrain/train_gnn.py first:", gnn_weights_path)
            sys.exit(1)
        gnn_model, atom_dict = load_pretrained_gnn(gnn_weights_path, device)
        if decompose == "path":
            for smi in tqdm(df["smiles"], desc="GNN path fragments"):
                feats, labels = get_fragment_features_gnn(smi, gnn_model, atom_dict, device, min_len, max_len)
                features_list.append(feats)
                labels_list.append(labels)
        else:
            for smi in tqdm(df["smiles"], desc="GNN BRICS/FG"):
                feats, labels, _subs = get_substructure_features_gnn(smi, gnn_model, atom_dict, device)
                features_list.append(feats)
                labels_list.append(labels)
        fragment_dim = gnn_model.dim

    # Filter molecules with at least one fragment
    valid = [i for i in range(len(features_list)) if len(features_list[i]) > 0]
    if not valid:
        print("No valid fragment features.")
        sys.exit(1)
    df = df.iloc[valid].reset_index(drop=True)
    features_list = [features_list[i] for i in valid]
    labels_list = [labels_list[i] for i in valid]

    if split_boundary is None:
        idx = np.arange(len(df))
        train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=random_state)
    else:
        train_idx = np.asarray([new_i for new_i, old_i in enumerate(valid) if old_i < split_boundary])
        test_idx = np.asarray([new_i for new_i, old_i in enumerate(valid) if old_i >= split_boundary])

    if len(train_idx) == 0 or len(test_idx) == 0:
        print("Train/test split has no valid molecules after decomposition.")
        sys.exit(1)

    # Normalize fragment features from train molecules only to avoid leaking test data.
    all_feats = np.concatenate([np.stack(features_list[i]) for i in train_idx], axis=0)
    feature_mean = all_feats.mean(axis=0, keepdims=True).astype(np.float32)
    feature_std = all_feats.std(axis=0, keepdims=True).astype(np.float32) + 1e-6
    for i, feats in enumerate(features_list):
        if not feats:
            continue
        arr = np.stack(feats)
        arr = (arr - feature_mean) / feature_std
        features_list[i] = [arr[j] for j in range(arr.shape[0])]

    X_t, mask_t, all_labels = build_padded_fragment_tensor(features_list, labels_list, device)

    # Normalize pIC50 target for more stable regression
    y = df["pic50"].values.astype(np.float32)
    y_mean = float(y[train_idx].mean())
    y_std = float(y[train_idx].std() + 1e-6)
    y_norm = (y - y_mean) / y_std
    y_t = torch.FloatTensor(y_norm).to(device).unsqueeze(1)

    if model_type == "mha":
        model = SubstructureAttentionRegression(
            fragment_dim=fragment_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        ).to(device)
    elif model_type == "attn_pool":
        model = AttentionPoolingRegression(
            fragment_dim=fragment_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        ).to(device)
    elif model_type == "set_transformer":
        model = SetTransformerRegression(
            fragment_dim=fragment_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            dropout=dropout,
        ).to(device)
    else:
        raise ValueError("Unknown model_type: " + str(model_type))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_r2_test = -float("inf")
    save_path = os.path.join(save_dir, f"attn_model_{encoder}_{decompose}_{model_type}.pt")
    ckpt = {
        "state_dict": model.state_dict(),
        "encoder": encoder,
        "decompose": decompose,
        "model_type": model_type,
        "fragment_dim": fragment_dim,
        "hidden_dim": hidden_dim,
        "num_heads": num_heads,
        "num_encoder_layers": num_encoder_layers,
        "dropout": dropout,
        "min_len": min_len,
        "max_len": max_len,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "dataset": dataset,
        "train_csv": train_csv,
        "test_csv": test_csv,
        "training_log": log_path,
    }

    for epoch in range(1, epochs + 1):
        model.train()
        perm = np.random.permutation(len(train_idx))
        loss_sum = 0
        n_batches = 0
        for start in range(0, len(perm), batch_size):
            batch_idx = train_idx[perm[start : start + batch_size]]
            x_b = X_t[batch_idx]
            m_b = mask_t[batch_idx]
            y_b = y_t[batch_idx]
            pred, _ = model(x_b, m_b)
            loss = F.mse_loss(pred, y_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            n_batches += 1
        train_loss = loss_sum / max(n_batches, 1)
        model.eval()
        with torch.no_grad():
            pred_train, _ = model(X_t[train_idx], mask_t[train_idx])
            pred_test, _ = model(X_t[test_idx], mask_t[test_idx])
            r2_train = 1 - F.mse_loss(pred_train, y_t[train_idx]).item() / (y_t[train_idx].var().item() + 1e-8)
            r2_test = 1 - F.mse_loss(pred_test, y_t[test_idx]).item() / (y_t[test_idx].var().item() + 1e-8)
        if r2_test > best_r2_test:
            best_r2_test = r2_test
            ckpt["state_dict"] = model.state_dict()
            torch.save(ckpt, save_path)
        if epoch % 10 == 0 or epoch == 1:
            print("Epoch", epoch, "Train MSE:", train_loss, "R2 train:", r2_train, "R2 test:", r2_test)

    print("Best Test R2:", best_r2_test)
    print("Saved best checkpoint to", save_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="FAAH", help="Dataset folder under data/datasets")
    p.add_argument("--csv", default=None)
    p.add_argument("--train_csv", default=None)
    p.add_argument("--test_csv", default=None)
    p.add_argument("--save_dir", default=None)
    p.add_argument("--encoder", choices=["fp", "gnn"], default="fp")
    p.add_argument(
        "--decompose",
        choices=["path", "brics"],
        default="path",
        help="path: atom-path fragments; brics: BRICS+Murcko+functional groups",
    )
    p.add_argument(
        "--model_type",
        choices=["mha", "attn_pool", "set_transformer"],
        default="mha",
        help="mha: existing query self-attn; attn_pool: softmax pooling; set_transformer: encoder+pool",
    )
    p.add_argument("--gnn_weights", default=None)
    p.add_argument("--min_len", type=int, default=2)
    p.add_argument("--max_len", type=int, default=6)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--num_encoder_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    main(
        dataset=args.dataset,
        csv_path=args.csv,
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        save_dir=args.save_dir,
        encoder=args.encoder,
        decompose=args.decompose,
        model_type=args.model_type,
        gnn_weights_path=args.gnn_weights,
        min_len=args.min_len,
        max_len=args.max_len,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_encoder_layers,
        dropout=args.dropout,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        test_size=args.test_size,
        random_state=args.seed,
    )
