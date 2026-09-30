"""
Attention over fragment features (fingerprint or GNN) then regression to pIC50.
Returns attention weights for visualization (contribution of each fragment to output).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SubstructureAttentionRegression(nn.Module):
    """
    Fragment features (B, T, D) -> attention -> single vector -> regression -> pIC50.
    T = number of fragments, D = feature dim (fp or GNN). Returns pred and attention weights (B, T).
    """

    def __init__(self, fragment_dim, hidden_dim=128, num_heads=4, dropout=0.1):
        super(SubstructureAttentionRegression, self).__init__()
        self.proj = nn.Linear(fragment_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.regress = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, fragment_features, mask=None):
        """
        fragment_features: (B, T, D)
        mask: (B, T) True = valid fragment, False = padding. If None, all valid.
        Returns: pred (B, 1), attn_weights (B, T)
        """
        B, T, D = fragment_features.shape
        if mask is None:
            mask = torch.ones(B, T, dtype=torch.bool, device=fragment_features.device)
        # (B, T, H)
        h = self.proj(fragment_features)
        # key_padding_mask: (B, T) True = ignore
        key_padding = ~mask
        # self-attention: query = key = value = h; we want one vector per sample so we use a single query or mean.
        # Use mean of h as query to get attention over fragments: (B, 1, H) query, (B, T, H) key, value
        query = h.mean(dim=1, keepdim=True)
        out, attn_weights = self.attention(
            query, h, h, key_padding_mask=key_padding, need_weights=True
        )
        # out (B, 1, H), attn_weights (B, 1, T)
        out = out.squeeze(1)
        attn_weights = attn_weights.squeeze(1)
        # zero out attention on padding for clarity
        attn_weights = attn_weights.masked_fill(key_padding, 0.0)
        pred = self.regress(out)
        return pred, attn_weights


class AttentionPoolingRegression(nn.Module):
    """
    Method 2: softmax attention over substructures, pooled = sum_i alpha_i * h_i, then MLP -> pIC50.
    Importance proxy: alpha_i * ||embedding_i|| (use fragment_features norm, optional * gradient in caller).
    """

    def __init__(self, fragment_dim, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(fragment_dim, hidden_dim)
        self.attn_score = nn.Linear(hidden_dim, 1, bias=False)
        self.regress = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, fragment_features, mask=None):
        B, T, D = fragment_features.shape
        if mask is None:
            mask = torch.ones(B, T, dtype=torch.bool, device=fragment_features.device)
        h = torch.tanh(self.proj(fragment_features))
        logits = self.attn_score(h).squeeze(-1)
        logits = logits.masked_fill(~mask, float("-inf"))
        alpha = torch.softmax(logits, dim=-1)
        alpha = alpha.masked_fill(~mask, 0.0)
        pooled = (alpha.unsqueeze(-1) * h).sum(dim=1)
        pred = self.regress(pooled)
        return pred, alpha


class SetTransformerRegression(nn.Module):
    """
    Method 4: self-attention between substructures (set, unordered), then attention pooling + MLP.
    Returns pred and pool attention weights (B, T) for interpretability.
    """

    def __init__(self, fragment_dim, hidden_dim=128, num_heads=4, num_encoder_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.proj = nn.Linear(fragment_dim, hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_encoder_layers)
        self.pool_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.pool_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.regress = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, fragment_features, mask=None):
        B, T, _ = fragment_features.shape
        if mask is None:
            mask = torch.ones(B, T, dtype=torch.bool, device=fragment_features.device)
        h = self.proj(fragment_features)
        key_padding = ~mask
        h = self.encoder(h, src_key_padding_mask=key_padding)
        q = self.pool_query.expand(B, -1, -1)
        out, attn_w = self.pool_attn(q, h, h, key_padding_mask=key_padding, need_weights=True)
        out = out.squeeze(1)
        attn_w = attn_w.squeeze(1)
        attn_w = attn_w.masked_fill(key_padding, 0.0)
        pred = self.regress(out)
        return pred, attn_w
