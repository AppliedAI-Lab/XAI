import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SPLIT_SIZE = 0.9

# Project root: .../SubstructureAttn/data
data_dir = Path(__file__).resolve().parent.parent.parent / "data/raw_csv"

for file in os.listdir(data_dir):
    if not file.endswith(".csv"):
        continue

    if not file.startswith("FAAH"):
        continue

    stem = Path(file).stem
    prefix = stem.split("_", 1)[0]
    out_root = data_dir / prefix
    out_root.mkdir(parents=True, exist_ok=True)

    path = data_dir / file
    df = pd.read_csv(path)[["Smiles", "pIC50"]]

    train_df, test_df = train_test_split(
        df,
        train_size=SPLIT_SIZE,
        random_state=42,
        shuffle=True,
    )

    train_df.to_csv(out_root / f"{prefix}_train.csv", index=False)
    test_df.to_csv(out_root / f"{prefix}_test.csv", index=False)
