import os
import sys
import atexit

import pandas as pd


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASETS_ROOT = os.path.join(ROOT, "data", "datasets")


class TeeStream:
    """Write console output to multiple streams."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            try:
                stream.write(data)
                stream.flush()
            except ValueError:
                pass

    def flush(self):
        for stream in self.streams:
            try:
                stream.flush()
            except ValueError:
                pass

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def start_training_log(checkpoint_dir, filename="training.log"):
    """Mirror stdout/stderr to training.log in the checkpoint directory."""
    os.makedirs(checkpoint_dir or ".", exist_ok=True)
    log_path = os.path.join(checkpoint_dir or ".", filename)
    log_file = open(log_path, "a", buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeStream(original_stdout, log_file)
    sys.stderr = TeeStream(original_stderr, log_file)

    def close_log():
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()

    atexit.register(close_log)
    print("Training log:", log_path)
    return log_path


def resolve_split_paths(dataset="FAAH", train_csv=None, test_csv=None):
    """Resolve train/test CSV paths, defaulting to pre-created dataset splits."""
    if train_csv is not None:
        train_path = train_csv
    else:
        train_path = os.path.join(DATASETS_ROOT, dataset, f"{dataset}_train.csv")

    if test_csv is not None:
        test_path = test_csv
    else:
        test_path = os.path.join(DATASETS_ROOT, dataset, f"{dataset}_test.csv")

    return train_path, test_path


def load_split_frames(load_fn, dataset="FAAH", train_csv=None, test_csv=None):
    """Load normalized train/test DataFrames with columns smiles and pic50."""
    train_path, test_path = resolve_split_paths(dataset, train_csv, test_csv)
    if not os.path.isfile(train_path):
        raise FileNotFoundError(f"Train CSV not found: {train_path}")
    if not os.path.isfile(test_path):
        raise FileNotFoundError(f"Test CSV not found: {test_path}")
    return load_fn(train_path), load_fn(test_path), train_path, test_path


TARGETS_CSV = os.path.join(ROOT, "data", "targets.csv")


def target_validation_frame(target=None, csv_path=None):
    """Return target-validation SMILES from data/targets.csv as a DataFrame."""
    path = csv_path or TARGETS_CSV
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Targets CSV not found: {path}")
    df = pd.read_csv(path)
    required = {"target", "pdb_id", "smiles"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Targets CSV missing columns: {sorted(missing)}")
    if target is not None:
        df = df[df["target"] == target].reset_index(drop=True)
    return df[["target", "pdb_id", "smiles"]]
