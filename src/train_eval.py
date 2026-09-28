"""
Shared training/evaluation entry point — ALL FOUR MODELS run through this script so results are
directly comparable (same metrics, same logging format, same config).

Usage:
    python -m src.train_eval --model bilstm --config config.yaml

TODO (team): flesh this out together on Day 2 (see the plan doc's timeline) before individual
model-building starts on Day 3 — it's the shared foundation everyone builds on.
"""

import argparse
import importlib
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import f1_score

from src.data_utils import (
    load_config,
    load_raw,
    make_or_load_split,
    compute_class_weights,
    build_vocab,
    build_dataloaders,
)

MODEL_REGISTRY = {
    "bilstm": "src.models.bilstm",
    "textcnn": "src.models.textcnn",
    "bilstm_attention": "src.models.bilstm_attention",
    "distilbert": "src.models.distilbert",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        choices=MODEL_REGISTRY.keys(),
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a small subset for a quick local pipeline check.",
    )
    parser.add_argument(
        "--full-train",
        action="store_true",
        help="Run the full training configuration.",
    )
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(args, cfg, vocab_size):
    module = importlib.import_module(MODEL_REGISTRY[args.model])

    if args.model == "bilstm":
        model = module.BiLSTMClassifier(
            vocab_size=vocab_size,
            num_labels=len(cfg["data"]["labels"]),
            embed_dim=100,
            hidden_dim=128,
            dropout=0.3,
            embedding_dropout=0.1,
            pad_idx=0,
        )
    else:
        raise NotImplementedError(f"Model '{args.model}' is not implemented yet.")

    return model


def run_one_epoch(model, loader, loss_fn, optimizer, device, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_samples = 0

    for input_ids, lengths, mask, labels in loader:
        input_ids = input_ids.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            logits = model(input_ids, lengths, mask)
            loss = loss_fn(logits, labels)

            if train:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


def evaluate(model, loader, loss_fn, device, threshold=0.5):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for input_ids, lengths, mask, labels in loader:
            input_ids = input_ids.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)

            logits = model(input_ids, lengths, mask)
            loss = loss_fn(logits, labels)

            probs = torch.sigmoid(logits)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / total_samples

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    predictions = (all_probs >= threshold).astype(int)

    macro_f1 = f1_score(
        all_labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    return avg_loss, macro_f1, all_probs, all_labels


def tune_thresholds(y_true, y_probs):
    """
    Find the best classification threshold independently for each label
    using the validation set only.
    """

    thresholds = np.arange(0.10, 0.91, 0.05)
    best_thresholds = []

    for class_idx in range(y_true.shape[1]):
        best_f1 = -1.0
        best_threshold = 0.5

        for threshold in thresholds:
            predictions = (y_probs[:, class_idx] >= threshold).astype(int)

            score = f1_score(
                y_true[:, class_idx],
                predictions,
                zero_division=0,
            )

            if score > best_f1:
                best_f1 = score
                best_threshold = threshold

        best_thresholds.append(best_threshold)

    return np.array(best_thresholds)


def macro_f1_with_thresholds(y_true, y_probs, thresholds):
    """
    Calculate multilabel Macro-F1 using one threshold per class.
    """

    predictions = (y_probs >= thresholds).astype(int)

    return f1_score(
        y_true,
        predictions,
        average="macro",
        zero_division=0,
    )


def main():
    args = parse_args()
    cfg = load_config(args.config)

    df = load_raw(cfg)

    train_df, val_df, test_df = make_or_load_split(
        cfg["data"]["raw_csv"],
        cfg["data"]["split_dir"],
        random_state=cfg["seed"],
    )
    if args.smoke_test:
        train_df = train_df.head(1000).copy()
        val_df = val_df.head(200).copy()
        test_df = test_df.head(200).copy()

        print("SMOKE TEST: using 1,000 train / 200 validation / 200 test samples")

    class_weights = compute_class_weights(
        train_df,
        cfg["data"]["labels"],
    )

    print(f"Model: {args.model}")
    print(f"Train/val/test sizes: {len(train_df)}/{len(val_df)}/{len(test_df)}")
    print(f"Class weights: {class_weights}")

    # # TODO: dynamically import the chosen model module, build DataLoaders (tokenizer fit on
    # # train_df only per data_utils.build_vocab), train with early stopping, log
    # # loss/accuracy/macro-F1 curves, save checkpoint to Drive (not to the repo), and — ONLY on
    # # the final run — evaluate once on test_df using the metrics listed in config.yaml.
    # raise NotImplementedError(
    #     "Wire up model import + train loop + eval here. See config.yaml for the shared "
    #     "hyperparameters every model must respect."
    # )
    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")

    # Build vocabulary using TRAINING DATA ONLY
    vocab = build_vocab(
        train_df["comment_text"],
        min_freq=cfg["preprocessing"]["min_vocab_freq"],
    )

    print(f"Vocabulary size: {len(vocab):,}")

    # Build shared DataLoaders
    train_loader, val_loader, test_loader = build_dataloaders(
        cfg,
        train_df,
        val_df,
        test_df,
        vocab,
    )

    # Build M1 BiLSTM
    model = build_model(
        args,
        cfg,
        vocab_size=len(vocab),
    ).to(device)

    print(f"Trainable parameters: {count_parameters(model):,}")

    # BCEWithLogitsLoss with train-only positive class weights
    pos_weight = torch.tensor(
        [class_weights[label] for label in cfg["data"]["labels"]],
        dtype=torch.float32,
        device=device,
    )

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
    )

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    max_epochs = 1 if args.smoke_test else cfg["training"]["epochs"]
    patience = cfg["training"]["early_stopping_patience"]

    history = []

    best_val_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    best_state = None
    best_thresholds = None

    training_start = time.time()

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()

        train_loss = run_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
            train=True,
        )

        val_loss, _, val_probs, val_labels = evaluate(
            model,
            val_loader,
            loss_fn,
            device,
            threshold=0.5,
        )

        val_thresholds = tune_thresholds(
            val_labels,
            val_probs,
        )

        val_macro_f1 = macro_f1_with_thresholds(
            val_labels,
            val_probs,
            val_thresholds,
        )

        epoch_time = time.time() - epoch_start

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_macro_f1": val_macro_f1,
                "epoch_time_seconds": epoch_time,
            }
        )

        print(
            f"Epoch {epoch}/{max_epochs} | "
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val Macro-F1: {val_macro_f1:.4f} | "
            f"Time: {epoch_time:.2f}s"
        )

        print(
            "  Validation thresholds:",
            dict(
                zip(
                    cfg["data"]["labels"],
                    val_thresholds.tolist(),
                )
            ),
        )

        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            best_epoch = epoch
            epochs_without_improvement = 0

            best_state = {
                key: value.cpu().clone() for key, value in model.state_dict().items()
            }

            best_thresholds = val_thresholds.copy()

            print("  New best validation Macro-F1.")

        else:
            epochs_without_improvement += 1

            print(f"  No improvement " f"({epochs_without_improvement}/{patience})")

            if epochs_without_improvement >= patience:
                print("Early stopping triggered.")
                break

    total_training_time = time.time() - training_start

    print()
    print("Training complete.")
    print(f"Best validation Macro-F1: {best_val_f1:.4f}")
    print(f"Best epoch: {best_epoch}")
    print(f"Total training time: {total_training_time:.2f}s")

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    thresholds_path = "results/bilstm_best_thresholds.json"

    with open(thresholds_path, "w", encoding="utf-8") as f:
        json.dump(
            dict(
                zip(
                    cfg["data"]["labels"],
                    best_thresholds.tolist(),
                )
            ),
            f,
            indent=2,
        )

    print(f"Best validation thresholds saved to {thresholds_path}")

    # Save training history
    results_path = (
        "results/bilstm_log.csv"
        if not args.smoke_test
        else "results/bilstm_smoke_log.csv"
    )

    import csv

    with open(results_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_loss",
                "val_macro_f1",
                "epoch_time_seconds",
            ],
        )

        writer.writeheader()
        writer.writerows(history)

    print(f"Training log saved to {results_path}")


if __name__ == "__main__":
    main()
