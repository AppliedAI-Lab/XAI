import argparse
import os
from typing import List

import pandas as pd
from rdkit import Chem


def randomize_smiles(smiles: str, num_aug: int) -> List[str]:
    """Return up to num_aug randomized SMILES strings different from the original."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    original = Chem.MolToSmiles(mol, canonical=True)
    augmented: List[str] = []

    # Try more times than requested to avoid duplicates identical to original
    max_trials = num_aug * 10 if num_aug > 0 else 0
    tried = 0
    while len(augmented) < num_aug and tried < max_trials:
        tried += 1
        rand_smiles = Chem.MolToSmiles(
            mol,
            canonical=False,
            doRandom=True,
        )
        if rand_smiles == smiles or rand_smiles == original:
            continue
        if rand_smiles in augmented:
            continue
        augmented.append(rand_smiles)

    return augmented


def augment_dataset(input_csv: str, output_csv: str, num_aug: int, smiles_col: str = "smiles") -> None:
    if num_aug <= 0:
        raise ValueError("num_aug must be > 0")

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    if smiles_col not in df.columns:
        raise KeyError(f"Column '{smiles_col}' not found in {input_csv}")

    rows = []
    for _, row in df.iterrows():
        base_smiles = row[smiles_col]
        rows.append(row.to_dict())

        if not isinstance(base_smiles, str) or not base_smiles.strip():
            continue

        aug_smiles_list = randomize_smiles(base_smiles, num_aug)
        for s in aug_smiles_list:
            new_row = row.to_dict()
            new_row[smiles_col] = s
            rows.append(new_row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_csv, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Augment SMILES in a CSV file by generating randomized, non-canonical SMILES.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/FAAH_clean.csv",
        help="Path to input CSV file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/FAAH_clean_augmented.csv",
        help="Path to output augmented CSV file.",
    )
    parser.add_argument(
        "--num_aug",
        type=int,
        default=50,
        help="Number of augmentations per original SMILES.",
    )
    parser.add_argument(
        "--smiles_col",
        type=str,
        default="Smiles",
        help="Name of the SMILES column in the CSV file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    augment_dataset(args.input, args.output, args.num_aug, args.smiles_col)


if __name__ == "__main__":
    main()

