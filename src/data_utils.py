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
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


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

class ToxicDataset(Dataset):
    """Encodes every comment ONCE at start-up. Each item = (input_ids[max_len], length, labels[6])."""

    def __init__(self, df, vocab: dict, label_cols: list, max_len: int = 128, text_col: str = "comment_text"):
        encoded = [encode(t, vocab, max_len) for t in df[text_col]]
        self.input_ids = torch.tensor([e[0] for e in encoded], dtype=torch.long)
        self.lengths = torch.tensor([e[1] for e in encoded], dtype=torch.long)
        self.labels = torch.tensor(df[label_cols].values, dtype=torch.float)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.lengths[idx], self.labels[idx]


def collate_fn(batch):
    """Shared by BiLSTM / TextCNN / BiLSTM+Attention.
    Returns (input_ids [B, L] long, lengths [B] long, mask [B, L] bool, labels [B, 6] float).
    mask is True for real tokens. It is built from lengths, so an empty comment still has
    1 True position (avoids NaN in attention softmax)."""
    input_ids, lengths, labels = zip(*batch)
    input_ids = torch.stack(input_ids)
    lengths = torch.stack(lengths)
    labels = torch.stack(labels)
    mask = torch.arange(input_ids.size(1)).unsqueeze(0) < lengths.unsqueeze(1)
    return input_ids, lengths, mask, labels


def build_dataloaders(cfg: dict, train_df, val_df, test_df, vocab: dict, num_workers: int = 0):
    """Train loader is shuffled with a seeded generator (reproducible); val/test are not shuffled."""
    label_cols = cfg["data"]["labels"]
    max_len = cfg["preprocessing"]["max_length"]
    batch_size = cfg["training"]["batch_size"]
    g = torch.Generator()
    g.manual_seed(cfg["seed"])

    loaders = []
    for df, shuffle in [(train_df, True), (val_df, False), (test_df, False)]:
        ds = ToxicDataset(df, vocab, label_cols, max_len)
        loaders.append(DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=num_workers,
            generator=g if shuffle else None,
            pin_memory=torch.cuda.is_available(),
        ))
    return tuple(loaders)  # (train_loader, val_loader, test_loader)

def load_glove_embeddings(vocab: dict, glove_path: str, embed_dim: int = 100,
                          seed: int = 42, verbose: bool = True) -> torch.Tensor:
    """
    Build the [vocab_size, embed_dim] embedding matrix from GloVe 6B.
    - word found in GloVe  -> its GloVe vector
    - word not in GloVe (incl. <unk>) -> small random U(-0.25, 0.25) (Kim, 2014), seeded
    - <pad> (index 0) -> all zeros
    Use in a model: nn.Embedding.from_pretrained(emb, freeze=False, padding_idx=PAD_IDX)
    """
    rng = np.random.default_rng(seed)
    emb = rng.uniform(-0.25, 0.25, size=(len(vocab), embed_dim)).astype(np.float32)
    emb[PAD_IDX] = 0.0

    found = 0
    with open(glove_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            idx = vocab.get(parts[0])
            if idx is not None and len(parts) == embed_dim + 1:
                emb[idx] = np.asarray(parts[1:], dtype=np.float32)
                found += 1

    if verbose:
        print(f"GloVe: found {found:,} / {len(vocab):,} vocab words ({found / len(vocab):.2%}); "
              f"the rest are random-initialised")
    return torch.from_numpy(emb)


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
