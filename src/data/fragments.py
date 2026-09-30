"""
Load FAAH CSV and build ligand x fragment matrix from atom-path fragments (length 2-6).
Helpers for fragment subgraph (SMILES for fingerprint, 3D coords for GNN).
"""
import pandas as pd
from collections import Counter
from rdkit import Chem
from rdkit.Chem import AllChem


def load_faah_csv(csv_path):
    """
    Load FAAH_clean.csv. Expects columns 'Smiles' and 'pIC50'.
    Returns DataFrame with columns 'smiles', 'pic50'.
    """
    df = pd.read_csv(csv_path)
    # normalize column names for internal use
    col_map = {}
    if "Smiles" in df.columns:
        col_map["Smiles"] = "smiles"
    if "smiles" not in df.columns and "Smiles" not in df.columns:
        raise ValueError("CSV must contain 'Smiles' or 'smiles' column")
    if "pIC50" in df.columns:
        col_map["pIC50"] = "pic50"
    if "pic50" not in df.columns and "pIC50" not in df.columns:
        raise ValueError("CSV must contain 'pIC50' or 'pic50' column")
    df = df.rename(columns=col_map)
    if "smiles" not in df.columns:
        df["smiles"] = df["Smiles"]
    if "pic50" not in df.columns:
        df["pic50"] = df["pIC50"]
    return df[["smiles", "pic50"]].copy()


def fragment_paths(smiles, min_len=2, max_len=6):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return Counter()
    frags = Counter()
    num_atoms = mol.GetNumAtoms()
    for start in range(num_atoms):
        stack = [(start, [start])]
        while stack:
            idx, path = stack.pop()
            if len(path) >= min_len:
                atoms = [mol.GetAtomWithIdx(i).GetSymbol() for i in path]
                frags["-".join(atoms)] += 1
            if len(path) < max_len:
                for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors():
                    ni = neighbor.GetIdx()
                    if ni not in path:
                        stack.append((ni, path + [ni]))
    return frags


def fragment_paths_with_indices(smiles, min_len=2, max_len=6):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return
    num_atoms = mol.GetNumAtoms()
    for start in range(num_atoms):
        stack = [(start, [start])]
        while stack:
            idx, path = stack.pop()
            if len(path) >= min_len:
                atoms = [mol.GetAtomWithIdx(i).GetSymbol() for i in path]
                yield "-".join(atoms), list(path)
            if len(path) < max_len:
                for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors():
                    ni = neighbor.GetIdx()
                    if ni not in path:
                        stack.append((ni, path + [ni]))


def get_fragment_subgraph_mol(mol, path_indices):
    """
    Return the induced subgraph of mol on path_indices as an RWMol.
    path_indices: list of atom indices in mol.
    """
    if mol is None or not path_indices:
        return None
    rw = Chem.RWMol()
    idx_map = {}
    for idx in path_indices:
        atom = mol.GetAtomWithIdx(idx)
        new_atom = Chem.Atom(atom.GetSymbol())
        rw.AddAtom(new_atom)
        idx_map[idx] = rw.GetNumAtoms() - 1
    for i in path_indices:
        for j in path_indices:
            if i >= j:
                continue
            bond = mol.GetBondBetweenAtoms(i, j)
            if bond is not None:
                rw.AddBond(idx_map[i], idx_map[j], bond.GetBondType())
    sub = rw.GetMol()
    try:
        Chem.SanitizeMol(sub)
    except Exception:
        pass
    return sub


def get_fragment_smiles(mol, path_indices):
    """Get canonical SMILES of the fragment subgraph for fingerprinting."""
    sub = get_fragment_subgraph_mol(mol, path_indices)
    if sub is None:
        return None
    try:
        return Chem.MolToSmiles(sub)
    except Exception:
        return None


def get_fragment_3d_coords(mol, path_indices):
    """
    Get 3D coordinates for fragment atoms. Embeds mol in 3D if no conformer.
    Returns (atoms: list of str, coords: list of [x,y,z]).
    """
    if mol is None or not path_indices:
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
    for idx in path_indices:
        atom = mol.GetAtomWithIdx(idx)
        atoms.append(atom.GetSymbol())
        pos = conf.GetAtomPosition(idx)
        coords.append([pos.x, pos.y, pos.z])
    return atoms, coords


def build_fragment_matrix(df, min_occurrences=10):
    """
    Build ligand x fragment count matrix from DataFrame with 'smiles' and 'pic50'.
    Filters out fragments that appear in fewer than min_occurrences ligands.
    Returns:
        X: DataFrame, index = range(len(df)), columns = fragment names
        y: Series, pic50
        all_frags: list of fragment names (after filtering)
    """
    all_frags = set()
    frag_list = []
    for smi in df["smiles"]:
        fr = fragment_paths(smi)
        frag_list.append(fr)
        all_frags |= set(fr.keys())
    all_frags = sorted(all_frags)

    X = []
    for fr in frag_list:
        X.append([fr.get(f, 0) for f in all_frags])
    X = pd.DataFrame(X, columns=all_frags)
    y = df["pic50"].copy()

    # filter rare fragments
    X = X.loc[:, (X > 0).sum() >= min_occurrences]
    return X, y, list(X.columns)
