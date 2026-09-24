"""
Shared training/evaluation entry point — ALL FOUR MODELS run through this script so results are
directly comparable (same metrics, same logging format, same config).
Owner: vievegan
Role: M3 (Shared training/evaluation script)

Usage:
    python -m src.train_eval --model bilstm --config config.yaml

TODO (team): flesh this out together on Day 2 (see the plan doc's timeline) before individual
model-building starts on Day 3 — it's the shared foundation everyone builds on.
"""

import argparse

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
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    df = load_raw(cfg)
    train_df, val_df, test_df = make_or_load_split(cfg, df)
    class_weights = compute_class_weights(train_df, cfg["data"]["labels"])

    print(f"Model: {args.model}")
    print(f"Train/val/test sizes: {len(train_df)}/{len(val_df)}/{len(test_df)}")
    print(f"Class weights: {class_weights}")

    # TODO: dynamically import the chosen model module, build DataLoaders (tokenizer fit on
    # train_df only per data_utils.build_vocab), train with early stopping, log
    # loss/accuracy/macro-F1 curves, save checkpoint to Drive (not to the repo), and — ONLY on
    # the final run — evaluate once on test_df using the metrics listed in config.yaml.
    raise NotImplementedError(
        "Wire up model import + train loop + eval here. See config.yaml for the shared "
        "hyperparameters every model must respect."
    )


if __name__ == "__main__":
    main()
