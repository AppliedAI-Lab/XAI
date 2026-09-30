"""
LSTM-based regression from SMILES (ReLeaSE/OpenChem style).
Input: sequence of token indices -> Embedding -> LSTM -> MLP -> scalar prediction.
"""
import torch
import torch.nn as nn


def build_vocab(smiles_list, pad_token=" ", add_pad=True):
    """
    Build character-level vocab from list of SMILES.
    Returns: tokens (list of chars, pad at index 0 if add_pad), char2idx (dict).
    """
    all_chars = set()
    for s in smiles_list:
        if isinstance(s, str) and s:
            all_chars.update(s)
    tokens = sorted(all_chars)
    if add_pad and pad_token not in tokens:
        tokens = [pad_token] + tokens
    elif add_pad:
        tokens = [pad_token] + [c for c in tokens if c != pad_token]
    char2idx = {c: i for i, c in enumerate(tokens)}
    return tokens, char2idx


def smiles_to_indices(smiles, char2idx, pad_idx=0, max_len=None, pad_right=True):
    """
    Encode one SMILES to list of indices. Optionally pad to max_len.
    Returns: list of int (length max_len if max_len else len(smiles)).
    """
    idx = [char2idx.get(c, pad_idx) for c in smiles]
    if max_len is not None:
        if len(idx) > max_len:
            idx = idx[:max_len]
        elif len(idx) < max_len:
            padding = [pad_idx] * (max_len - len(idx))
            idx = idx + padding if pad_right else padding + idx
    return idx


def encode_and_pad_batch(smiles_list, char2idx, pad_idx=0, max_len=None, device=None):
    """
    Encode list of SMILES to padded tensor (N, L) and lengths (N,).
    If max_len is None, use max length in batch.
    """
    if max_len is None:
        max_len = max(len(s) for s in smiles_list if isinstance(s, str))
    rows = []
    lengths = []
    for s in smiles_list:
        if not isinstance(s, str) or not s:
            rows.append([pad_idx] * max_len)
            lengths.append(0)
            continue
        lengths.append(min(len(s), max_len))
        rows.append(smiles_to_indices(s, char2idx, pad_idx, max_len, pad_right=True))
    t = torch.LongTensor(rows)
    if device is not None:
        t = t.to(device)
    lengths = torch.LongTensor(lengths)
    if device is not None:
        lengths = lengths.to(device)
    return t, lengths


class SmilesLSTMRegression(nn.Module):
    """
    ReLeaSE/OpenChem-style: Embedding -> LSTM -> MLP -> regression.
    Same layout as RecurrentQSAR-example-logp: 2-layer LSTM hidden 128, MLP 128->1.
    """

    def __init__(
        self,
        num_embeddings,
        embedding_dim=128,
        hidden_size=128,
        n_layers=2,
        dropout=0.0,
        padding_idx=0,
    ):
        super(SmilesLSTMRegression, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embed = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x, lengths=None):
        """
        x: (B, L) Long tensor of token indices.
        lengths: (B,) optional; if provided use last valid hidden, else use last step.
        Returns: (B, 1) predicted values.
        """
        B, L = x.shape
        emb = self.embed(x)
        if lengths is not None and lengths.max().item() > 0:
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                emb, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, (h_n, _) = self.lstm(packed)
            last_hidden = h_n[-1]
        else:
            out, (h_n, _) = self.lstm(emb)
            last_hidden = h_n[-1]
        return self.mlp(last_hidden)
