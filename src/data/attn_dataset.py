"""
Build fragment-level features for attention model: fingerprint or GNN per fragment.
"""
import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

# Suppress RDKit warnings (aromaticity, deprecation, etc.)
RDLogger.DisableLog("rdApp.*")

from src.data.fragments import (
    fragment_paths_with_indices,
    get_fragment_smiles,
    get_fragment_3d_coords,
)
from src.data.substructure_decompose import decompose_smiles
from src.models.gnn_encoder import load_pretrained_gnn, build_fragment_gnn_input

# Default fingerprint size for fragment encoding
FP_RADIUS = 2
FP_NBITS = 2048


def smiles_to_fp(smiles, radius=FP_RADIUS, n_bits=FP_NBITS):
    """Morgan fingerprint from SMILES. Returns numpy array (n_bits,) or None."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp, dtype=np.float32)
    except Exception:
        return None


def get_substructure_features_fp(
    smiles,
    fp_radius=FP_RADIUS,
    fp_n_bits=FP_NBITS,
    include_scaffold=True,
    include_brics=True,
    include_fg=True,
):
    """
    ECFP per substructure from BRICS + Murcko scaffold + functional groups (deduped atom sets).
    Returns (features_list, labels_list, substructure_objects) where labels are substructure names.
    """
    subs = decompose_smiles(
        smiles,
        include_scaffold=include_scaffold,
        include_brics=include_brics,
        include_fg=include_fg,
    )
    if not subs:
        return [], [], []
    features = []
    labels = []
    kept_subs = []
    for s in subs:
        fp = smiles_to_fp(s.smiles, radius=fp_radius, n_bits=fp_n_bits)
        if fp is not None:
            features.append(fp)
            labels.append(s.name)
            kept_subs.append(s)
    return features, labels, kept_subs


def get_fragment_features_fp(smiles, min_len=2, max_len=6, fp_radius=FP_RADIUS, fp_n_bits=FP_NBITS):
    """
    For one molecule: list of fragment fingerprint vectors and fragment labels.
    Returns (features_list, labels_list). Each element of features_list is (fp_n_bits,) float32.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [], []
    features = []
    labels = []
    for path_key, path_indices in fragment_paths_with_indices(smiles, min_len, max_len):
        smi = get_fragment_smiles(mol, path_indices)
        if smi is None:
            continue
        fp = smiles_to_fp(smi, radius=fp_radius, n_bits=fp_n_bits)
        if fp is not None:
            features.append(fp)
            labels.append(path_key)
    return features, labels


def get_full_mol_atoms_coords(smiles):
    """3D conformer for full molecule; returns (atoms symbols, coords) or ([], [])."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [], []
    try:
        if mol.GetNumConformers() == 0:
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
        conf = mol.GetConformer()
    except Exception:
        return [], []
    atoms = []
    coords = []
    for i in range(mol.GetNumAtoms()):
        atoms.append(mol.GetAtomWithIdx(i).GetSymbol())
        pos = conf.GetAtomPosition(i)
        coords.append([pos.x, pos.y, pos.z])
    return atoms, coords


def get_substructure_features_gnn(
    smiles,
    gnn_model,
    atom_dict,
    device,
    include_scaffold=True,
    include_brics=True,
    include_fg=True,
):
    """
    Method 3: GNN atom embeddings, mean-pooled per BRICS/scaffold/FG substructure.
    Returns (features_list as numpy vectors, labels_list, subs).
    """
    subs = decompose_smiles(
        smiles,
        include_scaffold=include_scaffold,
        include_brics=include_brics,
        include_fg=include_fg,
    )
    if not subs:
        return [], [], []
    atoms, coords = get_full_mol_atoms_coords(smiles)
    if not atoms:
        return [], [], []
    atoms_idx, dist_mats, sizes = build_fragment_gnn_input([atoms], [coords], atom_dict, device)
    if not atoms_idx:
        return [], [], []
    with torch.no_grad():
        atom_emb = gnn_model.forward_atom_embeddings(atoms_idx, dist_mats, sizes)
    features = []
    labels = []
    for s in subs:
        idx = torch.tensor(s.atom_indices, device=device, dtype=torch.long)
        if idx.numel() == 0:
            continue
        vec = atom_emb.index_select(0, idx).mean(0)
        features.append(vec.cpu().numpy())
        labels.append(s.name)
    return features, labels, subs


def get_fragment_features_gnn(smiles, gnn_model, atom_dict, device, min_len=2, max_len=6):
    """
    For one molecule: list of fragment GNN embedding vectors and fragment labels.
    gnn_model: MolecularGNNEncoder in eval mode.
    Returns (features_list, labels_list). Each element is (dim,) tensor.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [], []
    atoms_list = []
    coords_list = []
    labels = []
    for path_key, path_indices in fragment_paths_with_indices(smiles, min_len, max_len):
        atoms, coords = get_fragment_3d_coords(mol, path_indices)
        if not atoms or not coords:
            continue
        atoms_list.append(atoms)
        coords_list.append(coords)
        labels.append(path_key)
    if not atoms_list:
        return [], []
    atoms_idx, dist_mats, sizes = build_fragment_gnn_input(atoms_list, coords_list, atom_dict, device)
    if not atoms_idx:
        return [], []
    with torch.no_grad():
        emb = gnn_model.forward_embed(atoms_idx, dist_mats, sizes)
        emb = F.normalize(emb, p=2, dim=1)
    features = [emb[i].cpu().numpy() for i in range(emb.size(0))]
    return features, labels


def build_padded_fragment_tensor(features_list, labels_list, device, max_fragments=None):
    """
    features_list: list of list of arrays (each inner list = one molecule's fragments).
    labels_list: list of list of str (fragment path keys).
    Returns: features (N, T, D), mask (N, T), list of list of labels per sample.
    """
    if max_fragments is None:
        max_fragments = max(len(f) for f in features_list)
    dim = features_list[0][0].shape[0] if features_list[0] else 0
    N = len(features_list)
    D = dim
    T = max_fragments
    features = np.zeros((N, T, D), dtype=np.float32)
    mask = np.zeros((N, T), dtype=np.bool_)
    all_labels = []
    for i, (feats, labels) in enumerate(zip(features_list, labels_list)):
        n = min(len(feats), T)
        if n > 0:
            features[i, :n] = np.stack(feats[:n])
            mask[i, :n] = True
        all_labels.append(labels[:T])
    features_t = torch.FloatTensor(features).to(device)
    mask_t = torch.BoolTensor(mask).to(device)
    return features_t, mask_t, all_labels
