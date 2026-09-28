"""
M2 (Tharsiga) - ONE-TIME test evaluation of the final TextCNN (C4, GloVe).

Rules this script follows:
  * Model and thresholds are FROZEN: both are read from the checkpoint saved by
    textcnn_train.train_textcnn (thresholds were tuned on VALIDATION only).
  * Before touching test, it re-checks the checkpoint on VALIDATION and stops if the
    numbers don't match results/textcnn_results.json (wrong checkpoint / split / vocab).
  * Test is evaluated ONCE. If the output JSON already exists, it stops (use --force only
    if the first run crashed before saving).

Usage (Colab, from repo root):
    python -m src.textcnn_test \
        --raw-csv   /content/drive/MyDrive/DL-Project/data/train.csv \
        --split-dir /content/drive/MyDrive/DL-Project/data/splits \
        --ckpt      /content/drive/MyDrive/DL-Project/checkpoints/textcnn_best.pt
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from src import data_utils as du
from src.models.textcnn import TextCNNClassifier, count_parameters
from src.textcnn_train import compute_metrics, predict_logits, save_json, set_seed, sigmoid

EXPECTED_SIZES = (111699, 23936, 23936)
EXPECTED_VOCAB = 74916


def plot_confusion(y_true, y_prob, thresholds, labels, path):
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    t = np.array([thresholds[c] for c in labels])
    y_pred = (y_prob >= t).astype(int)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
    for j, (ax, c) in enumerate(zip(axes.ravel(), labels)):
        cm = confusion_matrix(y_true[:, j], y_pred[:, j], labels=[0, 1])
        ax.imshow(cm, cmap="Blues")
        for (r, k), v in np.ndenumerate(cm):
            ax.text(k, r, f"{v:,}", ha="center", va="center",
                    color="white" if v > cm.max() / 2 else "black", fontsize=11)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["pred 0", "pred 1"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["true 0", "true 1"])
        ax.set_title(f"{c} (thr {thresholds[c]:.3f})")
    fig.suptitle("TextCNN - test confusion matrices (validation-tuned thresholds)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")


def plot_roc_pr(y_true, y_prob, labels, roc_path, pr_path):
    import matplotlib.pyplot as plt
    from sklearn.metrics import (average_precision_score, precision_recall_curve,
                                 roc_auc_score, roc_curve)
    fig, ax = plt.subplots(figsize=(7, 6))
    for j, c in enumerate(labels):
        fpr, tpr, _ = roc_curve(y_true[:, j], y_prob[:, j])
        ax.plot(fpr, tpr, label=f"{c} (AUC {roc_auc_score(y_true[:, j], y_prob[:, j]):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("TextCNN - test ROC curves"); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(roc_path, dpi=150); print(f"Saved {roc_path}")

    fig, ax = plt.subplots(figsize=(7, 6))
    for j, c in enumerate(labels):
        p, r, _ = precision_recall_curve(y_true[:, j], y_prob[:, j])
        ax.plot(r, p, label=f"{c} (AP {average_precision_score(y_true[:, j], y_prob[:, j]):.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("TextCNN - test precision-recall curves"); ax.legend(loc="lower left")
    fig.tight_layout(); fig.savefig(pr_path, dpi=150); print(f"Saved {pr_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--raw-csv", default=None)
    ap.add_argument("--split-dir", default=None)
    ap.add_argument("--vocab", default="data/splits/vocab.json")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val-results", default="results/textcnn_results.json")
    ap.add_argument("--out", default="results/textcnn_test_results.json")
    ap.add_argument("--fig-prefix", default="results/textcnn")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-size-check", action="store_true", help="only for smoke tests")
    args = ap.parse_args()

    if os.path.exists(args.out) and not args.force:
        raise SystemExit(f"{args.out} already exists - test was already evaluated. Stopping.")

    cfg = du.load_config(args.config)
    labels = cfg["data"]["labels"]
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- 1. same split + same vocab as training ----
    train_df, val_df, test_df = du.make_or_load_split(
        args.raw_csv or cfg["data"]["raw_csv"], args.split_dir or cfg["data"]["split_dir"],
        random_state=cfg["seed"])
    sizes = (len(train_df), len(val_df), len(test_df))
    vocab = du.load_vocab(args.vocab)
    print(f"Split sizes {sizes} | vocab {len(vocab):,} | device {device}")
    if not args.skip_size_check:
        assert sizes == EXPECTED_SIZES, f"Split sizes {sizes} != {EXPECTED_SIZES}"
        assert len(vocab) == EXPECTED_VOCAB, f"Vocab {len(vocab)} != {EXPECTED_VOCAB}"
    _, val_loader, test_loader = du.build_dataloaders(cfg, train_df, val_df, test_df, vocab)

    # ---- 2. rebuild the frozen model from the checkpoint ----
    ckpt = torch.load(args.ckpt, map_location="cpu")
    hp, thresholds = ckpt["hparams"], ckpt["thresholds"]
    assert ckpt["vocab_size"] == len(vocab), "checkpoint vocab_size != vocab.json"
    model = TextCNNClassifier(
        ckpt["vocab_size"], len(labels), embed_dim=cfg["preprocessing"]["embed_dim"],
        num_filters=hp["num_filters"], filter_sizes=tuple(hp["filter_sizes"]),
        dropout=hp["dropout"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"Loaded {args.ckpt} | hparams {hp} | embeddings {ckpt.get('embeddings')}")

    # ---- 3. safety check on VALIDATION before touching test ----
    val_logits, val_true = predict_logits(model, val_loader, device)
    val_m = compute_metrics(val_true, sigmoid(val_logits), thresholds, labels)
    print(f"Validation re-check: macro-F1 {val_m['macro_f1']:.4f}")
    if os.path.exists(args.val_results) and not args.skip_size_check:
        ref = json.load(open(args.val_results))["val"]
        diff = abs(val_m["macro_f1"] - ref["macro_f1"])
        assert diff < 1e-3, (f"Val macro-F1 {val_m['macro_f1']:.4f} != saved {ref['macro_f1']:.4f}"
                             " - wrong checkpoint? Test NOT evaluated.")
        for c in labels:
            assert abs(thresholds[c] - ref["thresholds"][c]) < 1e-6, f"threshold mismatch: {c}"
        print(f"Matches {args.val_results} (diff {diff:.1e}). Thresholds match. OK.")

    # ---- 4. TEST - once ----
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    test_logits, test_true = predict_logits(model, test_loader, device)
    if device == "cuda":
        torch.cuda.synchronize()
    infer_sec = time.time() - t0
    test_prob = sigmoid(test_logits)
    test_m = compute_metrics(test_true, test_prob, thresholds, labels)

    result = {
        "model": "textcnn",
        "note": "Test evaluated ONCE with the frozen C4 checkpoint and validation-tuned thresholds.",
        "checkpoint": os.path.basename(args.ckpt),
        "hparams": hp,
        "embeddings": ckpt.get("embeddings"),
        "split_sizes": {"train": sizes[0], "val": sizes[1], "test": sizes[2]},
        "val": val_m,
        "test": test_m,
        "efficiency": {
            "params_total": count_parameters(model),
            "params_without_embedding": count_parameters(model, include_embedding=False),
            "inference_sec_per_1000": infer_sec / len(test_df) * 1000,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        },
    }
    save_json(result, args.out)
    plot_confusion(test_true, test_prob, thresholds, labels, f"{args.fig_prefix}_confusion.png")
    plot_roc_pr(test_true, test_prob, labels, f"{args.fig_prefix}_roc.png", f"{args.fig_prefix}_pr.png")

    print(f"\nTEST macro-F1 {test_m['macro_f1']:.4f} | micro-F1 {test_m['micro_f1']:.4f}")
    print("Per-class test F1:", {c: round(v, 3) for c, v in test_m["f1_per_class"].items()})


if __name__ == "__main__":
    main()
