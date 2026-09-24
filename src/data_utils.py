"""
Shared data pipeline: loading, the ONE fixed train/val/test split, and tokenizer/vocab building.

Everyone imports from here instead of re-implementing their own split or tokenizer — that's what
keeps the 4-model comparison fair and leakage-free. See config.yaml for the shared settings.
"""

import yaml
import pandas as pd
from sklearn.model_selection import train_test_split


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_raw(cfg: dict) -> pd.DataFrame:
    """Load the raw Jigsaw CSV. Expects a 'comment_text' column plus the label columns in
    cfg['data']['labels']."""
    import os
    if not os.path.exists(cfg["data"]["raw_csv"]):
        print(f"Warning: {cfg['data']['raw_csv']} not found. Returning a mock dataset for testing.")
        # Create a mock dataframe
        data = {"comment_text": ["this is a test"] * 100}
        for label in cfg["data"]["labels"]:
            data[label] = [0, 1] * 50
        return pd.DataFrame(data)
    return pd.read_csv(cfg["data"]["raw_csv"])


def make_or_load_split(cfg: dict, df: pd.DataFrame):
    """
    Create the ONE canonical stratified train/val/test split (seeded), or load it if it already
    exists on disk. Run this ONCE as a team and commit the resulting split file references (not
    the data itself — see .gitignore) so everyone trains/evaluates on identical rows.

    TODO: implement multi-label stratification (e.g. iterative-stratification package) — plain
    train_test_split does not stratify properly on multi-label targets, which matters given how
    rare some classes are (threat, identity_hate).
    """
    seed = cfg["seed"]
    val_ratio = cfg["data"]["val_ratio"]
    test_ratio = cfg["data"]["test_ratio"]

    train_val, test = train_test_split(df, test_size=test_ratio, random_state=seed)
    train, val = train_test_split(
        train_val, test_size=val_ratio / (1 - test_ratio), random_state=seed
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


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
