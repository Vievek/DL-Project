"""
M4 — DistilBERT, fine-tuned (owner: DEEPDEV)

Pretrained Transformer encoder (Hugging Face `distilbert-base-uncased`) with a linear multi-label
head on top of the [CLS] token, fine-tuned end-to-end. Uses its OWN pretrained tokenizer — do not
run this through the shared vocab-building path in data_utils.build_vocab (that's for the
non-pretrained models only).
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from tqdm import tqdm
from transformers import DistilBertModel, DistilBertTokenizerFast

from src.baseline import evaluate, tune_thresholds
from src.data_utils import compute_class_weights, load_config, load_raw, make_or_load_split

MODEL_NAME = "distilbert-base-uncased"


def get_tokenizer():
    return DistilBertTokenizerFast.from_pretrained(MODEL_NAME)


class DistilBertClassifier(nn.Module):
    def __init__(self, num_labels: int, dropout: float = 0.1, freeze_base: bool = False):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(MODEL_NAME)
        if freeze_base:
            for p in self.bert.parameters():
                p.requires_grad = False
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]  # [CLS] representation
        return self.classifier(self.dropout(cls_token))  # raw logits, shape (batch, num_labels)



class ToxicDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length: int):
        self.enc = tokenizer(
            list(texts), max_length=max_length, padding="max_length", truncation=True,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.enc["input_ids"][i], self.enc["attention_mask"][i], self.labels[i]


@torch.no_grad()
def predict_probs(model, loader, device, use_amp: bool = False) -> np.ndarray:
    model.eval()
    out = []
    for ids, mask, _ in loader:
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(ids.to(device), mask.to(device))
        out.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(out)


def _run_epochs(model, loader, val_loader, val_y, optimizer, criterion, scaler, device, n_epochs,
                stage, state, ckpt_dir, patience, use_amp):
    for epoch in range(n_epochs):
        epoch_start = time.time()
        model.train()
        total = 0.0
        for ids, mask, y in tqdm(loader, desc=f"{stage} epoch {epoch + 1}/{n_epochs}"):
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, enabled=use_amp):  # fp16 forward
                loss = criterion(model(ids, mask).float(), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()

        epoch_time = time.time() - epoch_start
        val_f1 = f1_score(val_y, predict_probs(model, val_loader, device, use_amp) >= 0.5,
                          average="macro", zero_division=0)
        print(f"[{stage}] epoch {epoch + 1}: train_loss={total / len(loader):.4f} "
              f"val_macro_f1@0.5={val_f1:.4f} epoch_time={epoch_time:.0f}s")
        state["history"].append({"stage": stage, "epoch": epoch + 1,
                                 "train_loss": total / len(loader), "val_macro_f1": val_f1,
                                 "epoch_time_sec": epoch_time})
        if ckpt_dir:  # save to Drive EVERY epoch so a Colab disconnect loses nothing
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "last.pt"))
        if val_f1 > state["best_f1"]:
            state["best_f1"], state["bad"] = val_f1, 0
            if ckpt_dir:
                torch.save(model.state_dict(), os.path.join(ckpt_dir, "best.pt"))
            state["best_state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            state["bad"] += 1
            if state["bad"] >= patience:
                print("Early stopping.")
                return


def train_distilbert(cfg, train_df, val_df, test_df, ckpt_dir=None, head_epochs=1, ft_epochs=2,
                     head_lr=1e-3, ft_lr=2e-5, max_pos_weight=50.0, fp16=True):
    labels = cfg["data"]["labels"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = fp16 and device.type == "cuda"  # fp16 autocast + GradScaler only makes sense on GPU
    torch.manual_seed(cfg["seed"])
    bs, max_len = cfg["training"]["batch_size"], cfg["preprocessing"]["max_length"]

    tok = get_tokenizer()
    mk = lambda df, shuffle: torch.utils.data.DataLoader(
        ToxicDataset(df["comment_text"], df[labels].values, tok, max_len),
        batch_size=bs, shuffle=shuffle)
    train_loader, val_loader, test_loader = mk(train_df, True), mk(val_df, False), mk(test_df, False)

    weights = compute_class_weights(train_df, labels)
    pos_weight = torch.tensor([min(weights[l], max_pos_weight) for l in labels], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = DistilBertClassifier(len(labels), freeze_base=True).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    state = {"best_f1": -1.0, "bad": 0, "history": [], "best_state": None}
    patience = cfg["training"]["early_stopping_patience"]
    val_y = val_df[labels].values

    t0 = time.time()
    opt = torch.optim.AdamW(model.classifier.parameters(), lr=head_lr)
    _run_epochs(model, train_loader, val_loader, val_y, opt, criterion, scaler, device,
                head_epochs, "head-only", state, ckpt_dir, patience, use_amp)

    for p in model.bert.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW(model.parameters(), lr=ft_lr)
    state["bad"] = 0
    _run_epochs(model, train_loader, val_loader, val_y, opt, criterion, scaler, device,
                ft_epochs, "fine-tune", state, ckpt_dir, patience, use_amp)
    train_time = time.time() - t0

    model.load_state_dict(state["best_state"])
    val_prob = predict_probs(model, val_loader, device, use_amp)
    thresholds = tune_thresholds(val_y, val_prob)  # tuned on VAL only

    t1 = time.time()
    test_prob = predict_probs(model, test_loader, device, use_amp)
    infer_time = time.time() - t1

    return {
        "model": "distilbert-base-uncased",
        "val": evaluate(val_y, val_prob, thresholds, labels),
        "test": evaluate(test_df[labels].values, test_prob, thresholds, labels),
        "efficiency": {
            "params_total": sum(p.numel() for p in model.parameters()),
            "train_time_sec": train_time,
            "inference_sec_per_1000": infer_time / len(test_df) * 1000,
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "fp16": use_amp,
        },
        "history": state["history"],
    }


def smoke_test(cfg, train_df, val_df, test_df, full_train_size, ckpt_dir=None, fp16=True):
    """1-epoch sanity check on a subsample: verifies [CLS] extraction, output shape and checkpoint
    saving, then extrapolates time/epoch to the full training set."""
    labels = cfg["data"]["labels"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = get_tokenizer()
    texts = list(train_df["comment_text"].head(4))
    enc = tok(texts, max_length=cfg["preprocessing"]["max_length"], padding="max_length",
              truncation=True, return_tensors="pt").to(device)

    model = DistilBertClassifier(len(labels)).to(device).eval()
    with torch.no_grad():
        logits = model(enc["input_ids"], enc["attention_mask"])
        hidden = model.bert(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        manual = model.classifier(hidden.last_hidden_state[:, 0, :])  # token 0 = [CLS]
    checks = {
        "output_shape": list(logits.shape),
        "output_shape_ok": tuple(logits.shape) == (4, len(labels)),
        "cls_extraction_ok": bool(torch.allclose(logits, manual, atol=1e-5)),
        "first_token_is_cls_id": bool((enc["input_ids"][:, 0] == tok.cls_token_id).all()),
    }
    assert checks["output_shape_ok"] and checks["cls_extraction_ok"], checks
    del model

    results = train_distilbert(cfg, train_df, val_df, test_df, ckpt_dir, head_epochs=0,
                               ft_epochs=1, fp16=fp16)
    ckpt_file = os.path.join(ckpt_dir, "last.pt") if ckpt_dir else None
    checks["checkpoint_saved"] = bool(ckpt_file and os.path.getsize(ckpt_file) > 0)
    checks["checkpoint_path"] = ckpt_file

    epoch_time = results["history"][0]["epoch_time_sec"]
    n = len(train_df)
    est_epoch = epoch_time / n * full_train_size
    return {
        "model": "distilbert-base-uncased",
        "subsample_train_rows": n,
        "full_train_rows": full_train_size,
        "params_total": results["efficiency"]["params_total"],
        "gpu": results["efficiency"]["gpu"],
        "fp16": results["efficiency"]["fp16"],
        "epoch_time_sec_subsample": epoch_time,
        "est_epoch_time_sec_full": est_epoch,
        "est_time_3_epochs_min_full": est_epoch * 3 / 60,
        "inference_sec_per_1000": results["efficiency"]["inference_sec_per_1000"],
        "checks": checks,
        "val_macro_f1": results["val"]["macro_f1"],
        "test_macro_f1": results["test"]["macro_f1"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--ckpt_dir", default=None, help="e.g. /content/drive/MyDrive/toxic/ckpt")
    parser.add_argument("--out", default="results/distilbert_results.json")
    parser.add_argument("--head_epochs", type=int, default=1)
    parser.add_argument("--ft_epochs", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None, help="subsample rows (smoke test only)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = load_raw(cfg)
    if args.limit:
        df = df.sample(args.limit, random_state=cfg["seed"])
    train_df, val_df, test_df = make_or_load_split(cfg, df)

    results = train_distilbert(cfg, train_df, val_df, test_df, args.ckpt_dir,
                               args.head_epochs, args.ft_epochs)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Test macro-F1: {results['test']['macro_f1']:.4f} -> saved to {args.out}")


if __name__ == "__main__":
    main()
