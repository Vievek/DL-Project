# Data

Raw data is **not committed** to this repo (Kaggle's terms and file size). To reproduce:

1. Download from https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data
2. Place `train.csv` in `data/raw/`.
3. Run the split step in `src/data_utils.py` (`make_or_load_split`) to generate the team's fixed
   train/val/test split — do this ONCE and share the resulting split (row ids or CSVs) with the
   team via Drive so everyone trains on identical data.

Fallback dataset if Jigsaw access becomes a blocker: [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions)
(Google, 27 emotion categories, also multi-label).
