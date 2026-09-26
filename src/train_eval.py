"""
Shared training/evaluation entry point — ALL FOUR MODELS run through this script so results are
directly comparable (same metrics, same logging format, same config).

Usage:
    python -m src.train_eval --model bilstm --config config.yaml
    python -m src.train_eval --model distilbert --data-pipeline hf --ckpt-dir /content/drive/MyDrive/toxic/ckpt

--data-pipeline:
    vocab  (default) own vocabulary built from train split (BiLSTM / TextCNN / attention models)
    hf     pretrained Hugging Face tokenizer (DistilBERT) — do NOT use build_vocab for this one

TODO (team): the "vocab" pipeline for the other 3 models is still to be wired by their owners.
"""

import argparse
import importlib
import json
import os

from src.data_utils import load_config, load_raw, make_or_load_split, compute_class_weights

MODEL_REGISTRY = {
    "bilstm": "src.models.bilstm",
    "textcnn": "src.models.textcnn",
    "bilstm_attention": "src.models.bilstm_attention",
    "distilbert": "src.models.distilbert",
}


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


def main():
    args = parse_args()
    cfg = load_config(args.config)

    df = load_raw(cfg)
    full_train_size = int(len(df) * cfg["data"]["train_ratio"])
    if args.smoke and not args.subsample:
        args.subsample = 10000
    if args.subsample:
        df = df.sample(args.subsample, random_state=cfg["seed"])
    train_df, val_df, test_df = make_or_load_split(cfg, df)
    class_weights = compute_class_weights(train_df, cfg["data"]["labels"])

    print(f"Model: {args.model} | data pipeline: {args.data_pipeline}")
    print(f"Train/val/test sizes: {len(train_df)}/{len(val_df)}/{len(test_df)}")
    print(f"Class weights: {class_weights}")

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

    # TODO (other model owners): build vocab from train_df only (data_utils.build_vocab), make
    # DataLoaders, train with early stopping, save checkpoint to Drive, evaluate once on test_df.
    raise NotImplementedError(
        f"'{args.model}' with the vocab pipeline is not wired yet. See config.yaml for the shared "
        "hyperparameters every model must respect."
    )


if __name__ == "__main__":
    main()
