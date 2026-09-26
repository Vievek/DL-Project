"""
Shared data pipeline: loading, the ONE fixed train/val/test split, and tokenizer/vocab building.

Everyone imports from here instead of re-implementing their own split or tokenizer — that's what
keeps the 4-model comparison fair and leakage-free. See config.yaml for the shared settings.
"""

import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import re
import json
from collections import Counter


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_raw(cfg: dict) -> pd.DataFrame:
    """Load the raw Jigsaw CSV. Expects a 'comment_text' column plus the label columns in
    cfg['data']['labels']."""
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


# ---------------------------------------------------------------------------
# M2 (Tharsiga) — text cleaning, tokenizer, vocab, encoding
# Used by BiLSTM / TextCNN / BiLSTM+Attention. DistilBERT uses its own tokenizer.
# ---------------------------------------------------------------------------
PAD_TOKEN, UNK_TOKEN = "<pad>", "<unk>"
PAD_IDX, UNK_IDX = 0, 1

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HTML_RE = re.compile(r"<[^>]+>")
_TOKEN_RE = re.compile(r"[a-z0-9']+|[^\sa-z0-9']")


def clean_text(text) -> str:
    """Lowercase (GloVe 6B is uncased), remove URLs and HTML tags, turn newlines/tabs into spaces.
    Stopwords are kept (negation matters) and there is no stemming."""
    text = str(text).lower()
    text = _URL_RE.sub(" ", text)
    text = _HTML_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text) -> list:
    """Clean, then split into word tokens + single punctuation tokens (e.g. '!' is its own token)."""
    return _TOKEN_RE.findall(clean_text(text))


def build_vocab(train_texts, min_freq: int = 2, verbose: bool = True) -> dict:
    """
    Build the vocabulary from the TRAINING SPLIT ONLY. Never call this on val/test/full data —
    that's leakage. Returns {token: index} with <pad>=0 and <unk>=1.
    """
    counter = Counter()
    for text in train_texts:
        counter.update(tokenize(text))

    vocab = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
    for token, freq in counter.most_common():
        if freq < min_freq:
            break
        vocab[token] = len(vocab)

    if verbose:
        total = sum(counter.values())
        covered = sum(f for t, f in counter.items() if t in vocab)
        print(f"Unique tokens in train: {len(counter):,}")
        print(f"Vocab size (min_freq={min_freq}, incl. <pad>/<unk>): {len(vocab):,}")
        print(f"Train token coverage: {covered / total:.2%}")
        print("Top-20 tokens:", counter.most_common(20))
    return vocab


def oov_rate(texts, vocab: dict) -> float:
    """Share of tokens in `texts` that are not in the vocab (become <unk>). Use on the val split."""
    total = unk = 0
    for text in texts:
        for tok in tokenize(text):
            total += 1
            unk += tok not in vocab
    return unk / max(total, 1)


def encode(text, vocab: dict, max_len: int = 128):
    """Text -> (input_ids, length). Truncate at the end, pad at the end with <pad>=0.
    length is at least 1 so an empty comment does not crash LSTM packing."""
    ids = [vocab.get(tok, UNK_IDX) for tok in tokenize(text)][:max_len]
    length = max(len(ids), 1)
    ids = ids + [PAD_IDX] * (max_len - len(ids))
    return ids, length


def save_vocab(vocab: dict, path: str = "data/splits/vocab.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)


def load_vocab(path: str = "data/splits/vocab.json") -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


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
