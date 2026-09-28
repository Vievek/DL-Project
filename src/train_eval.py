"""
Toxic Comment Classification - Shared Training and Evaluation Script
Owner: vievegan
Role: M3 (Shared training/evaluation script)

Usage:
    python -m src.train_eval --model bilstm --config config.yaml
    python -m src.train_eval --model distilbert --data-pipeline hf --ckpt-dir /content/drive/MyDrive/toxic/ckpt

--data-pipeline:
    vocab  (default) own vocabulary built from train split (BiLSTM / TextCNN / attention models)
    hf     pretrained Hugging Face tokenizer (DistilBERT) — do NOT use build_vocab for this one

TODO (team): the "vocab" pipeline (TextDataset/evaluate/plot_curves below) still uses dummy random
tokens and expects a generic module.Model() — needs wiring to data_utils.build_dataloaders /
build_vocab and each model's real classifier class (BiLSTMClassifier, etc).
"""
import argparse
import importlib
import json
import os
import tempfile

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.data_utils import load_config, load_raw, make_or_load_split, compute_class_weights

MODEL_REGISTRY = {
    "bilstm": "src.models.bilstm",
    "textcnn": "src.models.textcnn",
    "bilstm_attention": "src.models.bilstm_attention",
    "distilbert": "src.models.distilbert",
}

class TextDataset(Dataset):
    def __init__(self, df, label_cols):
        self.df = df
        self.label_cols = label_cols

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Dummy tokenization for now (returns random integers)
        tokens = torch.randint(0, 1000, (128,))
        labels = torch.tensor(self.df.iloc[idx][self.label_cols].values.astype(float), dtype=torch.float32)
        return tokens, labels

def evaluate(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)

            probs = torch.sigmoid(outputs)
            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    val_loss /= len(dataloader.dataset)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Calculate metrics
    preds_bin = (all_preds > 0.5).astype(int)

    macro_f1 = f1_score(all_targets, preds_bin, average='macro', zero_division=0)
    micro_f1 = f1_score(all_targets, preds_bin, average='micro', zero_division=0)

    metrics = {
        'val_loss': val_loss,
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'all_preds': all_preds,
        'all_targets': all_targets
    }

    return metrics

def plot_curves(targets, preds, label_cols, results_dir):
    os.makedirs(results_dir, exist_ok=True)

    from sklearn.metrics import roc_curve, precision_recall_curve
    plt.figure(figsize=(10, 8))
    for i, label in enumerate(label_cols):
        # Using a simple check to prevent errors when there is only one class in testing dummy split
        if len(np.unique(targets[:, i])) > 1:
            fpr, tpr, _ = roc_curve(targets[:, i], preds[:, i])
            auc = roc_auc_score(targets[:, i], preds[:, i])
            plt.plot(fpr, tpr, label=f"{label} (AUC = {auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.savefig(os.path.join(results_dir, "roc_curves.png"))
    plt.close()

    plt.figure(figsize=(10, 8))
    for i, label in enumerate(label_cols):
        if len(np.unique(targets[:, i])) > 1:
            precision, recall, _ = precision_recall_curve(targets[:, i], preds[:, i])
            pr_auc = average_precision_score(targets[:, i], preds[:, i])
            plt.plot(recall, precision, label=f"{label} (AUC = {pr_auc:.2f})")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR Curves')
    plt.legend()
    plt.savefig(os.path.join(results_dir, "pr_curves.png"))
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODEL_REGISTRY.keys())
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data-pipeline", default="vocab", choices=["vocab", "hf"])
    parser.add_argument("--ckpt-dir", default=None, help="checkpoint dir (Drive on Colab, never the repo)")
    parser.add_argument("--out", default=None, help="results JSON path (default results/<model>_results.json)")
    parser.add_argument("--head-epochs", type=int, default=1, help="distilbert: frozen-base epochs")
    parser.add_argument("--ft-epochs", type=int, default=2, help="distilbert: full fine-tune epochs")
    parser.add_argument("--subsample", type=int, default=None, help="use only N random rows (smoke tests)")
    parser.add_argument("--smoke", action="store_true",
                        help="distilbert: 1-epoch sanity check on a 10k subsample -> results/distilbert_smoke.json")
    parser.add_argument("--no-fp16", action="store_true", help="disable fp16 mixed precision")
    return parser.parse_args()


def get_splits(cfg: dict, subsample: int = None):
    """Resolve the ONE canonical split via data_utils.make_or_load_split (input_path, output_dir,
    random_state). A --subsample writes its own temp CSV + a separate output dir so smoke tests
    never touch or reuse the team's real cached split in data/splits/."""
    if subsample:
        df = load_raw(cfg).sample(subsample, random_state=cfg["seed"])
        tmp_dir = tempfile.mkdtemp(prefix="toxic_subsample_")
        tmp_csv = os.path.join(tmp_dir, "subsample.csv")
        df.to_csv(tmp_csv, index=False)
        return make_or_load_split(tmp_csv, os.path.join(tmp_dir, "splits"), random_state=cfg["seed"])
    return make_or_load_split(cfg["data"]["raw_csv"], cfg["data"]["split_dir"], random_state=cfg["seed"])


def main():
    args = parse_args()
    cfg = load_config(args.config)
    label_cols = cfg["data"]["labels"]
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    if args.smoke and not args.subsample:
        args.subsample = 10000
    full_train_size = int(len(load_raw(cfg)) * cfg["data"]["train_ratio"])
    train_df, val_df, test_df = get_splits(cfg, args.subsample)
    class_weights_dict = compute_class_weights(train_df, label_cols)
    class_weights = torch.tensor([class_weights_dict[l] for l in label_cols], dtype=torch.float32)

    print(f"Model: {args.model} | data pipeline: {args.data_pipeline}")
    print(f"Train/val/test sizes: {len(train_df)}/{len(val_df)}/{len(test_df)}")
    print(f"Class weights: {class_weights_dict}")

    if args.model == "distilbert":
        if args.data_pipeline != "hf":
            raise SystemExit("DistilBERT uses its pretrained tokenizer: pass --data-pipeline hf")
        module = importlib.import_module(MODEL_REGISTRY["distilbert"])
        if args.smoke:
            results = module.smoke_test(cfg, train_df, val_df, test_df, full_train_size,
                                        args.ckpt_dir, fp16=not args.no_fp16)
            out = args.out or "results/distilbert_smoke.json"
        else:
            results = module.train_distilbert(
                cfg, train_df, val_df, test_df, args.ckpt_dir, args.head_epochs, args.ft_epochs,
                fp16=not args.no_fp16,
            )
            out = args.out or "results/distilbert_results.json"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {out}")
        return

    # --- generic "vocab" pipeline for BiLSTM / TextCNN / BiLSTM+Attention (M3's loop, TODO above) ---
    train_dataset = TextDataset(train_df, label_cols)
    val_dataset = TextDataset(val_df, label_cols)
    test_dataset = TextDataset(test_df, label_cols)

    batch_size = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Dynamically load model
    module = importlib.import_module(MODEL_REGISTRY[args.model])
    model = module.Model().to("cpu")

    optimizer = optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)

    epochs = cfg["training"]["epochs"]
    patience = cfg["training"]["early_stopping_patience"]

    best_macro_f1 = -1
    patience_counter = 0
    best_model_path = os.path.join(results_dir, f"{args.model}_best.pt")
    log_data = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        val_metrics = evaluate(model, val_loader, criterion, "cpu")
        val_loss = val_metrics['val_loss']
        val_macro_f1 = val_metrics['macro_f1']

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Macro-F1: {val_macro_f1:.4f}")

        log_data.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_macro_f1": val_macro_f1
        })

        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Save logs
    pd.DataFrame(log_data).to_csv(os.path.join(results_dir, f"{args.model}_log.csv"), index=False)

    # Final evaluation on test set using best model
    model.load_state_dict(torch.load(best_model_path))
    test_metrics = evaluate(model, test_loader, criterion, "cpu")
    print(f"\nFinal Test Macro-F1: {test_metrics['macro_f1']:.4f}")

    # Plot curves
    plot_curves(test_metrics['all_targets'], test_metrics['all_preds'], label_cols, results_dir)

if __name__ == "__main__":
    main()
