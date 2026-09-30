"""
Method 5: masking-based importance — remove one substructure at a time and measure delta pIC50.
Applies to any predictor that maps SMILES -> scalar (torch or numpy).
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from src.data.substructure_decompose import Substructure, decompose_smiles


def smiles_after_removing_atoms(smiles: str, atom_indices: Sequence[int]) -> Optional[str]:
    """
    Remove atoms (reverse index order) and return canonical SMILES, or None if invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    to_remove = sorted({int(i) for i in atom_indices}, reverse=True)
    rw = Chem.RWMol(mol)
    for idx in to_remove:
        if 0 <= idx < rw.GetNumAtoms():
            rw.RemoveAtom(idx)
    try:
        Chem.SanitizeMol(rw)
    except Exception:
        return None
    out = rw.GetMol()
    try:
        return Chem.MolToSmiles(out)
    except Exception:
        return None


def masking_delta_pic50(
    smiles: str,
    predict_fn: Callable[[str], float],
    subs: Optional[Sequence[Substructure]] = None,
) -> Tuple[float, List[Tuple[str, float, Optional[str]]]]:
    """
    importance_i = pIC50_full - pIC50_without_i

    predict_fn: SMILES -> pIC50 (same scale as training).
    Returns (pIC50_full, list of (sub_name, delta, smiles_after_remove)).
    """
    if subs is None:
        subs = decompose_smiles(smiles)
    full_pred = float(predict_fn(smiles))
    rows: List[Tuple[str, float, Optional[str]]] = []
    for s in subs:
        smi2 = smiles_after_removing_atoms(smiles, s.atom_indices)
        if smi2 is None or smi2 == "":
            rows.append((s.name, float("nan"), None))
            continue
        try:
            pred_i = float(predict_fn(smi2))
        except Exception:
            rows.append((s.name, float("nan"), smi2))
            continue
        rows.append((s.name, full_pred - pred_i, smi2))
    return full_pred, rows


def linear_per_fragment_importance(
    coef: np.ndarray,
    scaler_mean: Optional[np.ndarray],
    scaler_scale: Optional[np.ndarray],
    fragment_fps: Sequence[np.ndarray],
    pooling: str = "sum",
) -> np.ndarray:
    """
    Method 1: estimate importance_i from linear weights and aggregation-pooled embeddings.
    coef: (D,) fitted on scaled pooled features.
    scaler_mean, scaler_scale: from StandardScaler (None => identity).
    fragment_fps: list of (D,) per substructure.
    """
    coef = np.asarray(coef, dtype=np.float64).ravel()
    if not fragment_fps:
        return np.array([], dtype=np.float64)
    imp = np.zeros(len(fragment_fps), dtype=np.float64)
    if scaler_mean is not None and scaler_scale is not None:
        sc = np.asarray(scaler_scale, dtype=np.float64).ravel()
        sc = np.where(sc < 1e-12, 1.0, sc)
        w_eff = coef / sc
    else:
        w_eff = coef

    fps = np.stack([np.asarray(fp, dtype=np.float64).ravel() for fp in fragment_fps], axis=0)
    n_frag, fp_dim = fps.shape

    def add_sum_weights(weights):
        return fps @ weights

    def add_mean_weights(weights):
        return (fps @ weights) / max(n_frag, 1)

    def add_max_weights(weights):
        max_values = fps.max(axis=0)
        is_max = fps == max_values[None, :]
        counts = np.maximum(is_max.sum(axis=0), 1)
        assigned = is_max * (weights / counts)[None, :]
        return assigned.sum(axis=1)

    if pooling == "sum":
        imp += add_sum_weights(w_eff[:fp_dim])
    elif pooling == "mean":
        imp += add_mean_weights(w_eff[:fp_dim])
    elif pooling == "max":
        imp += add_max_weights(w_eff[:fp_dim])
    elif pooling == "sum_mean":
        imp += add_sum_weights(w_eff[:fp_dim])
        imp += add_mean_weights(w_eff[fp_dim : 2 * fp_dim])
    elif pooling == "sum_mean_max":
        imp += add_sum_weights(w_eff[:fp_dim])
        imp += add_mean_weights(w_eff[fp_dim : 2 * fp_dim])
        imp += add_max_weights(w_eff[2 * fp_dim : 3 * fp_dim])
    else:
        raise ValueError("Unknown pooling: " + str(pooling))
    return imp


def attn_times_embedding_norm(
    alpha: np.ndarray,
    embeddings: np.ndarray,
) -> np.ndarray:
    """Method 2 proxy: alpha_i * ||embedding_i||_2."""
    alpha = np.asarray(alpha, dtype=np.float64).ravel()
    emb = np.asarray(embeddings, dtype=np.float64)
    norms = np.linalg.norm(emb, axis=-1)
    return alpha[: norms.shape[0]] * norms
