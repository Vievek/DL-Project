"""
Shared data pipeline: loading, the ONE fixed train/val/test split, and tokenizer/vocab building.

Everyone imports from here instead of re-implementing their own split or tokenizer — that's what
keeps the 4-model comparison fair and leakage-free. See config.yaml for the shared settings.
"""

import yaml
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_raw(cfg: dict) -> pd.DataFrame:
    """Load the raw Jigsaw CSV. Expects a 'comment_text' column plus the label columns in
    cfg['data']['labels']."""
    return pd.read_csv(cfg["data"]["raw_csv"])


def make_or_load_split(
    input_path,
    output_dir,
    random_state=42
):
    """
    Create a fixed 70/15/15 multi-label stratified split.

    The same random seed should be used by the whole team
    so every model uses exactly the same train/validation/test data.
    """

    input_path = Path(input_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    test_path = output_dir / "test.csv"

    # Reuse existing split
    if (
        train_path.exists()
        and val_path.exists()
        and test_path.exists()
    ):
        print("Existing split found. Loading it.")

        return (
            pd.read_csv(train_path),
            pd.read_csv(val_path),
            pd.read_csv(test_path)
        )

    print("Creating new 70/15/15 split...")

    df = pd.read_csv(input_path)

    # First split:
    # 70% train
    # 30% temporary
    msss_1 = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=0.30,
        random_state=random_state
    )

    train_idx, temp_idx = next(
        msss_1.split(
            df,
            df[LABELS]
        )
    )

    train_df = df.iloc[train_idx].reset_index(drop=True)
    temp_df = df.iloc[temp_idx].reset_index(drop=True)

    # Second split:
    # Half of 30% = 15%
    # Half of 30% = 15%
    msss_2 = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=random_state
    )

    val_idx, test_idx = next(
        msss_2.split(
            temp_df,
            temp_df[LABELS]
        )
    )

    val_df = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)

    # Save
    train_df.to_csv(
        train_path,
        index=False
    )

    val_df.to_csv(
        val_path,
        index=False
    )

    test_df.to_csv(
        test_path,
        index=False
    )

    print("Train:", train_df.shape)
    print("Validation:", val_df.shape)
    print("Test:", test_df.shape)

    return train_df, val_df, test_df

def build_vocab(train_texts, min_freq: int = 2):
    """
    Build the vocabulary from the TRAINING SPLIT ONLY. Never call this on val/test/full data —
    that's leakage. Used by the BiLSTM / TextCNN / attention models (DistilBERT uses its own
    pretrained tokenizer instead).

    TODO: implement tokenization + frequency-based vocab building (e.g. via torchtext or a simple
    Counter-based approach).
    """
    raise NotImplementedError("Build vocab from train_texts only — see docstring.")


def compute_class_weights(train_df: pd.DataFrame, label_cols: list):
    """
    Compute per-class positive weights (pos_weight for BCEWithLogitsLoss) from the TRAINING split,
    to counter label imbalance (threat / identity_hate are rare).
    """
    weights = {}
    n = len(train_df)
    for col in label_cols:
        n_pos = train_df[col].sum()
        n_neg = n - n_pos
        weights[col] = (n_neg / max(n_pos, 1))
    return weights
