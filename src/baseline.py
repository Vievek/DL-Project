"""
Baseline — TF-IDF + Logistic Regression (traditional ML, NOT one of the 4 required deep-learning
models). Useful as a reference point in the Results table to show what the deep models buy you
over a simple approach.

Usage (from repo root):
    python -m src.baseline --config config.yaml
"""

import argparse
import json
import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.multioutput import MultiOutputClassifier

from src.data_utils import load_config, load_raw, make_or_load_split


def build_baseline(max_features: int = 20000):
    """
    Fit TfidfVectorizer on TRAIN texts only (same leakage rule as everywhere else), then wrap
    LogisticRegression in MultiOutputClassifier for the 6 independent binary labels.
    """
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    classifier = MultiOutputClassifier(LogisticRegression(max_iter=1000, class_weight="balanced"))
    return vectorizer, classifier


def predict_proba(classifier, X) -> np.ndarray:
    """MultiOutputClassifier returns a list of (n, 2) arrays; stack the positive-class column."""
    return np.column_stack([p[:, 1] for p in classifier.predict_proba(X)])


def tune_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    """Per-class threshold that maximises F1 on the VALIDATION split (never on test)."""
    grid = np.linspace(0.05, 0.95, 91)
    thresholds = np.full(y_true.shape[1], 0.5)
    for j in range(y_true.shape[1]):
        scores = [f1_score(y_true[:, j], y_prob[:, j] >= t, zero_division=0) for t in grid]
        thresholds[j] = grid[int(np.argmax(scores))]
    return thresholds


def evaluate(y_true: np.ndarray, y_prob: np.ndarray, thresholds, labels: list) -> dict:
    """Same metric set as config.yaml's `evaluation.metrics`, so all models compare fairly."""
    y_pred = (y_prob >= np.asarray(thresholds)).astype(int)

    def safe(fn, j):
        try:
            return float(fn(y_true[:, j], y_prob[:, j]))
        except ValueError:  # class absent from this split
            return None

    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "precision_per_class": dict(
            zip(labels, map(float, precision_score(y_true, y_pred, average=None, zero_division=0)))
        ),
        "recall_per_class": dict(
            zip(labels, map(float, recall_score(y_true, y_pred, average=None, zero_division=0)))
        ),
        "roc_auc": {l: safe(roc_auc_score, j) for j, l in enumerate(labels)},
        "pr_auc": {l: safe(average_precision_score, j) for j, l in enumerate(labels)},
        "thresholds": dict(zip(labels, map(float, thresholds))),
    }


def run_baseline(cfg: dict, train_df, val_df, test_df) -> dict:
    labels = cfg["data"]["labels"]
    vectorizer, classifier = build_baseline()

    X_train = vectorizer.fit_transform(train_df["comment_text"])  # fit on TRAIN only
    X_val = vectorizer.transform(val_df["comment_text"])
    X_test = vectorizer.transform(test_df["comment_text"])

    classifier.fit(X_train, train_df[labels].values)

    thresholds = tune_thresholds(val_df[labels].values, predict_proba(classifier, X_val))
    results = {
        "model": "tfidf_logreg_baseline",
        "val": evaluate(val_df[labels].values, predict_proba(classifier, X_val), thresholds, labels),
        "test": evaluate(
            test_df[labels].values, predict_proba(classifier, X_test), thresholds, labels
        ),
    }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out", default="results/baseline_results.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_df, val_df, test_df = make_or_load_split(cfg, load_raw(cfg))
    results = run_baseline(cfg, train_df, val_df, test_df)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Test macro-F1: {results['test']['macro_f1']:.4f}  ->  saved to {args.out}")


if __name__ == "__main__":
    main()
