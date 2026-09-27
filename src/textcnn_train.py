"""
M2 (Tharsiga) — training + validation utilities for TextCNN.

Uses the shared pipeline in src/data_utils.py (same split, vocab, dataloaders, config, seed).
VALIDATION ONLY: this file never touches the test split. The test set is evaluated once,
by the shared evaluation step, after all models are frozen.

Output metric keys match results/distilbert_results.json so the final tables line up.
"""

import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

from src.models.textcnn import TextCNNClassifier, count_parameters

THRESHOLD_GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)  # 0.05, 0.10, ..., 0.95


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict_logits(model, loader, device):
    """Returns (logits [N, 6], targets [N, 6]) as numpy arrays."""
    model.eval()
    all_logits, all_targets = [], []
    for input_ids, lengths, mask, labels in loader:
        logits = model(input_ids.to(device), lengths.to(device), mask.to(device))
        all_logits.append(logits.float().cpu().numpy())
        all_targets.append(labels.numpy())
    return np.concatenate(all_logits), np.concatenate(all_targets)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def tune_thresholds(y_true, y_prob, label_cols):
    """Per-class threshold that maximises F1 on VALIDATION (config: per_class_on_validation)."""
    best = {}
    for j, col in enumerate(label_cols):
        scores = [f1_score(y_true[:, j], (y_prob[:, j] >= t).astype(int), zero_division=0)
                  for t in THRESHOLD_GRID]
        best[col] = float(THRESHOLD_GRID[int(np.argmax(scores))])
    return best


def compute_metrics(y_true, y_prob, thresholds, label_cols):
    """All metrics listed in config.yaml -> evaluation.metrics."""
    t = np.array([thresholds[c] for c in label_cols])
    y_pred = (y_prob >= t).astype(int)
    per = lambda fn: {c: float(fn(y_true[:, j], y_pred[:, j], zero_division=0))
                      for j, c in enumerate(label_cols)}
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_per_class": per(f1_score),
        "precision_per_class": per(precision_score),
        "recall_per_class": per(recall_score),
        "roc_auc": {c: float(roc_auc_score(y_true[:, j], y_prob[:, j]))
                    for j, c in enumerate(label_cols)},
        "pr_auc": {c: float(average_precision_score(y_true[:, j], y_prob[:, j]))
                   for j, c in enumerate(label_cols)},
        "thresholds": {c: float(thresholds[c]) for c in label_cols},
    }


def train_textcnn(cfg, train_loader, val_loader, vocab_size, embeddings=None, hparams=None,
                  max_epochs=None, ckpt_path=None, use_class_weights=None, device=None):
    """
    Train TextCNN with early stopping on validation macro-F1 (per-class tuned thresholds).
    embeddings: GloVe matrix from data_utils.load_glove_embeddings, or None for random init.
    Returns (model, result_dict, best_val_probs, val_targets).
    """
    label_cols = cfg["data"]["labels"]
    hp = {"num_filters": 100, "filter_sizes": [3, 4, 5], "dropout": 0.5,
          "lr": cfg["training"]["learning_rate"]}
    hp.update(hparams or {})
    hp["filter_sizes"] = list(hp["filter_sizes"])
    max_epochs = max_epochs or cfg["training"]["epochs"]
    patience = cfg["training"]["early_stopping_patience"]
    if use_class_weights is None:
        use_class_weights = cfg["training"]["class_weighting"]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Same seed + same shuffle order for every run -> fair comparison between configs
    set_seed(cfg["seed"])
    gen = getattr(train_loader.sampler, "generator", None)
    if gen is not None:
        gen.manual_seed(cfg["seed"])

    model = TextCNNClassifier(
        vocab_size, len(label_cols),
        embed_dim=cfg["preprocessing"]["embed_dim"],
        num_filters=hp["num_filters"], filter_sizes=tuple(hp["filter_sizes"]),
        dropout=hp["dropout"], pretrained_embeddings=embeddings,
    ).to(device)

    # pos_weight = n_neg / n_pos per label, from the TRAIN split (same formula as
    # data_utils.compute_class_weights)
    y_train = train_loader.dataset.labels
    n_pos = y_train.sum(dim=0)
    pos_weight = ((len(y_train) - n_pos) / n_pos.clamp(min=1)).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight if use_class_weights else None)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp["lr"])

    history, best_f1, best_state, bad_epochs = [], -1.0, None, 0
    best_metrics, best_val_prob, best_epoch, val_true = None, None, 0, None

    for epoch in range(1, max_epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        for input_ids, lengths, mask, labels in train_loader:
            input_ids, lengths = input_ids.to(device), lengths.to(device)
            mask, labels = mask.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(input_ids, lengths, mask), labels)
            loss.backward()
            optimizer.step()
            running += loss.item() * labels.size(0)
        if device == "cuda":
            torch.cuda.synchronize()
        epoch_time = time.time() - t0
        train_loss = running / len(train_loader.dataset)

        val_logits, val_true = predict_logits(model, val_loader, device)
        with torch.no_grad():
            val_loss = criterion(torch.from_numpy(val_logits).to(device),
                                 torch.from_numpy(val_true).to(device)).item()
        val_prob = sigmoid(val_logits)
        thr = tune_thresholds(val_true, val_prob, label_cols)
        metrics = compute_metrics(val_true, val_prob, thr, label_cols)
        f1_at_05 = float(f1_score(val_true, (val_prob >= 0.5).astype(int),
                                  average="macro", zero_division=0))

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "val_macro_f1": metrics["macro_f1"], "val_macro_f1_at_0.5": f1_at_05,
                        "epoch_time_sec": epoch_time})
        print(f"Epoch {epoch:2d} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | "
              f"val macro-F1 {metrics['macro_f1']:.4f} (at 0.5: {f1_at_05:.4f}) | "
              f"{epoch_time:.0f}s")

        if metrics["macro_f1"] > best_f1:
            best_f1, best_metrics, best_val_prob, best_epoch = (
                metrics["macro_f1"], metrics, val_prob, epoch)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch} (best epoch {best_epoch})")
                break

    model.load_state_dict(best_state)
    if ckpt_path:
        os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
        torch.save({"state_dict": best_state, "hparams": hp, "vocab_size": vocab_size,
                    "thresholds": best_metrics["thresholds"],
                    "embeddings": "glove.6B.100d" if embeddings is not None else "random"},
                   ckpt_path)

    times = [h["epoch_time_sec"] for h in history]
    result = {
        "model": "textcnn",
        "hparams": hp,
        "embeddings": "glove.6B.100d" if embeddings is not None else "random",
        "class_weighting": bool(use_class_weights),
        "params_total": count_parameters(model),
        "params_without_embedding": count_parameters(model, include_embedding=False),
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "avg_epoch_time_sec": float(np.mean(times)),
        "total_train_time_sec": float(np.sum(times)),
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "val": best_metrics,
        "history": history,
    }
    return model, result, best_val_prob, val_true


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"Saved {path}")


def plot_curves(history, path, title="TextCNN learning curves"):
    import matplotlib.pyplot as plt
    ep = [h["epoch"] for h in history]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(ep, [h["train_loss"] for h in history], "o-", label="train loss")
    a1.plot(ep, [h["val_loss"] for h in history], "o-", label="val loss")
    a1.set_xlabel("epoch"); a1.set_ylabel("weighted BCE loss"); a1.legend(); a1.set_title("Loss")
    a2.plot(ep, [h["val_macro_f1"] for h in history], "o-", label="val macro-F1 (tuned thr.)")
    a2.plot(ep, [h["val_macro_f1_at_0.5"] for h in history], "o--", label="val macro-F1 (thr. 0.5)")
    a2.set_xlabel("epoch"); a2.set_ylabel("macro-F1"); a2.legend(); a2.set_title("Validation macro-F1")
    fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    return fig
