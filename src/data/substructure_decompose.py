"""
Decompose ligands into substructures: BRICS + Murcko scaffold + functional groups (SMARTS).
Each piece yields atom indices in the parent molecule for masking and induced subgraph SMILES for ECFP.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Sequence, Set, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem, BRICS
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from src.data.fragments import get_fragment_smiles

# Common functional groups as (label, SMARTS). Overlaps are deduped by atom set later.
DEFAULT_FG_PATTERNS: Sequence[Tuple[str, str]] = (
    ("amide", "[NX3][CX3](=[OX1])[#6]"),
    ("carboxylic_acid", "[CX3](=O)[OX2H1]"),
    ("ester", "[CX3](=O)[OX2H0]"),
    ("ketone", "[#6][CX3](=O)[#6]"),
    ("primary_amine", "[NX3;H2][!#1]"),
    ("secondary_amine", "[NX3;H1]([!#1])[!#1]"),
    ("tertiary_amine", "[NX3;H0]([!#1])([!#1])[!#1]"),
    ("hydroxyl_aromatic", "[OX2H1][c]"),
    ("hydroxyl_aliphatic", "[OX2H1][C]"),
    ("nitro", "[NX3+](=O)[O-]"),
    ("sulfonamide", "S(=O)(=O)N"),
    ("aryl_halide", "[c][F,Cl,Br,I]"),
    ("nitrile", "[CX2]#[NX1]"),
    ("ether", "[OX2]([#6])[#6]"),
    ("thioether", "[SX2]([#6])[#6]"),
)


@dataclass(frozen=True)
class Substructure:
    """One substructure in the parent ligand."""

    kind: str
    name: str
    atom_indices: Tuple[int, ...]
    smiles: str


def _ensure_3d(mol: Chem.Mol) -> None:
    if mol.GetNumConformers() == 0:
        try:
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
        except Exception:
            try:
                AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=42)
            except Exception:
                pass


def brics_atom_fragment_sets(mol: Chem.Mol) -> List[Tuple[str, Tuple[int, ...]]]:
    """
    Connected components after removing all BRICS-cleavable bonds.
    Returns list of ('brics', atom_indices) for each fragment.
    """
    bonds = list(BRICS.FindBRICSBonds(mol))
    if not bonds:
        return []
    cut = set()
    for (a, b), _ in bonds:
        cut.add((min(a, b), max(a, b)))
    n = mol.GetNumAtoms()
    adj: List[List[int]] = [[] for _ in range(n)]
    for bnd in mol.GetBonds():
        a1, a2 = bnd.GetBeginAtomIdx(), bnd.GetEndAtomIdx()
        p = (min(a1, a2), max(a1, a2))
        if p in cut:
            continue
        adj[a1].append(a2)
        adj[a2].append(a1)
    visited = [False] * n
    out: List[Tuple[str, Tuple[int, ...]]] = []
    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        comp: List[int] = []
        while stack:
            u = stack.pop()
            if visited[u]:
                continue
            visited[u] = True
            comp.append(u)
            for v in adj[u]:
                if not visited[v]:
                    stack.append(v)
        if comp:
            out.append(("brics", tuple(sorted(comp))))
    return out


def murcko_scaffold_atoms(mol: Chem.Mol) -> Optional[Tuple[int, ...]]:
    """Atom indices of the Murcko scaffold in the parent molecule, if matchable."""
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
    except Exception:
        return None
    if scaf is None or scaf.GetNumAtoms() == 0:
        return None
    try:
        m = Chem.MolFromSmiles(Chem.MolToSmiles(scaf))
    except Exception:
        return None
    if m is None:
        return None
    match = mol.GetSubstructMatch(m)
    if not match:
        return None
    return tuple(sorted(int(x) for x in match))


def functional_group_hits(
    mol: Chem.Mol,
    patterns: Sequence[Tuple[str, str]] = DEFAULT_FG_PATTERNS,
) -> List[Tuple[str, Tuple[int, ...]]]:
    """Each unique atom set per (label, SMARTS) match."""
    seen: Set[Tuple[int, ...]] = set()
    out: List[Tuple[str, Tuple[int, ...]]] = []
    for label, smarts in patterns:
        q = Chem.MolFromSmarts(smarts)
        if q is None:
            continue
        try:
            matches = mol.GetSubstructMatches(q, uniquify=True)
        except Exception:
            continue
        for m in matches:
            t = tuple(sorted(int(x) for x in m))
            if t in seen:
                continue
            seen.add(t)
            out.append((label, t))
    return out


def _dedupe_by_atom_set(
    items: List[Tuple[str, str, Tuple[int, ...]]],
) -> List[Tuple[str, str, Tuple[int, ...]]]:
    """Keep first occurrence of each unique atom set."""
    seen: Set[Tuple[int, ...]] = set()
    out: List[Tuple[str, str, Tuple[int, ...]]] = []
    for kind, name, atoms in items:
        if atoms in seen:
            continue
        seen.add(atoms)
        out.append((kind, name, atoms))
    return out


def decompose_smiles(
    smiles: str,
    include_scaffold: bool = True,
    include_brics: bool = True,
    include_fg: bool = True,
    fg_patterns: Sequence[Tuple[str, str]] = DEFAULT_FG_PATTERNS,
) -> List[Substructure]:
    """
    BRICS + Murcko scaffold + functional groups. Atom sets are deduplicated.
    Each Substructure has canonical subgraph SMILES for Morgan (ECFP) encoding.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    _ensure_3d(mol)

    raw: List[Tuple[str, str, Tuple[int, ...]]] = []

    if include_scaffold:
        sc = murcko_scaffold_atoms(mol)
        if sc:
            raw.append(("scaffold", "murcko", sc))

    if include_brics:
        for kind, atoms in brics_atom_fragment_sets(mol):
            raw.append((kind, "fragment", atoms))

    if include_fg:
        for label, atoms in functional_group_hits(mol, fg_patterns):
            raw.append(("fg", label, atoms))

    raw = _dedupe_by_atom_set(raw)

    substructs: List[Substructure] = []
    for i, (kind, name, atoms) in enumerate(raw):
        smi = get_fragment_smiles(mol, list(atoms))
        if smi is None:
            continue
        sub_name = f"{kind}:{name}" if kind != "brics" else f"brics:{i}"
        substructs.append(Substructure(kind=kind, name=sub_name, atom_indices=atoms, smiles=smi))
    return substructs


def iter_substructures_for_training(
    smiles_list: Sequence[str],
    **kwargs,
) -> Iterator[Tuple[str, List[Substructure]]]:
    for smi in smiles_list:
        subs = decompose_smiles(smi, **kwargs)
        yield smi, subs


def substructure_set_signature(subs: Sequence[Substructure]) -> str:
    """Stable string for caching (not SMILES-order dependent)."""
    parts = sorted(f"{s.kind}|{s.smiles}" for s in subs)
    return "||".join(parts)
