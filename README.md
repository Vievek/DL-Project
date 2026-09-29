# Toxic Comment Classification — SE4050 Deep Learning Group Assignment

Multi-label toxic comment classification on the [Jigsaw Toxic Comment Classification
Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) dataset. Four
distinct deep learning architectures are trained and compared, per the SE4050 assignment brief
(Supervised Deep Learning category — at least 4 distinct models).

## Team

| Member | Model | Notes |
|---|---|---|
| ShevoniR | M1 — BiLSTM | |
| Tharsiga Ranganathan | M2 — TextCNN | |
| Vievegan | M3 — BiLSTM + attention | Also owns the shared `src/train_eval.py` |
| Pancharatnam Deepatharshan (DEEPDEV) | M4 — DistilBERT (fine-tuned) | |

A TF-IDF + Logistic Regression baseline is also included for reference; it does **not** count
toward the 4 required deep-learning models.

## Project layout

```text
data/         # NOT committed (see below) — dataset lives here locally / on Drive
notebooks/    # EDA and exploratory notebooks
src/          # Shared pipeline: data loading, preprocessing, training/eval, models/
results/      # Metrics, figures, logs (small files only — no large checkpoints)
config.yaml   # Shared seeds, split ratios, max length, epochs, etc.
```

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` is pinned to the versions used on the team's Colab T4 runs (see
`requirements_colab.txt` for the full `pip freeze`). It includes `iterative-stratification`, which
`src/data_utils.py`'s split function needs — without it, any training/evaluation command below will
fail to import.

## Getting the data

1. Create a free Kaggle account and verify your phone number (Settings → Phone Verification) —
   required before the competition download button appears.
2. Join the competition and accept its rules:
   https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data
3. Download `train.csv` (~70 MB) — the only file needed — and place it at `data/raw/train.csv`.
   (Alternative if Kaggle access is blocked: the Hugging Face mirror
   `datasets.load_dataset("Heliosoph/Jigsaw-Toxic-Comments")`, though Kaggle is preferred for
   citing the authentic source in the report.)

There is no separate split script to run. The **first** time any training/evaluation command below
runs, `src/data_utils.make_or_load_split` builds the canonical 70/15/15 multi-label stratified
split (seed 42, via `iterstrat`) from `data/raw/train.csv` and caches it to `data/splits/` as
`train.csv`/`val.csv`/`test.csv`. Every later run — by you or any teammate, on any model — reuses
that exact same cached split, so results are always comparable.

Raw data and generated split files are **not committed** — see `.gitignore`. Everyone must train
and evaluate on the exact same split: upload your `data/splits/*.csv` to the team Drive and check
its checksum against `results/split_checksums.txt` (`md5sum data/splits/*.csv`, or on Windows
`certutil -hashfile data/splits/train.csv MD5`) to confirm it matches your teammates' before you
rely on it for a final result.

## Branches

One branch per model owner, created off `main` from the start (`scripts/setup_branches.sh`):

| Branch | Owner | Model |
|---|---|---|
| `model/bilstm` | ShevoniR | M1 — BiLSTM |
| `model/textcnn` | Tharsiga Ranganathan | M2 — TextCNN |
| `model/bilstm-attention` | Vievegan | M3 — BiLSTM + attention |
| `model/distilbert` | DEEPDEV | M4 — DistilBERT |

Work on your own branch, commit and push there regularly (this is what the rubric's "regular,
meaningful, traceable contributions" checks), then open a pull request into `main` once your model
is trained and evaluated. `main` should always be in a working, mergeable state — don't push
half-finished experiments straight to it. Shared files (`src/data_utils.py`, `src/train_eval.py`,
`config.yaml`) live on `main`; if you need to change one of these shared files, do it in a small
PR of its own so everyone sees the diff, rather than burying it inside your model branch.

```bash
git clone <repo-url>
cd toxic-comment-classification
git checkout model/textcnn          # swap for your own branch
# ...work, commit, push to your branch as you go...
git push origin model/textcnn
# open a PR into main on GitHub when your model is ready
```

## Running a model

Each model lives in `src/models/`. Train/evaluate through the shared script:

```bash
python -m src.train_eval --model bilstm --config config.yaml
python -m src.train_eval --model textcnn --config config.yaml
python -m src.train_eval --model bilstm_attention --config config.yaml
python -m src.train_eval --model distilbert --data-pipeline hf --config config.yaml
```

DistilBERT needs `--data-pipeline hf`: it uses its own pretrained Hugging Face tokenizer instead of
the shared vocabulary the other three models build from the training split. Useful DistilBERT-only
flags (see `python -m src.train_eval --help`): `--smoke` (fast 1-epoch sanity check on a 10k-row
subsample before committing to a full run), `--ckpt-dir` (save checkpoints here every epoch — point
this at Google Drive on Colab, never the repo), `--head-epochs`/`--ft-epochs` (frozen-base vs.
full-fine-tune epoch counts). A ready-to-run Colab notebook for DistilBERT is at
`notebooks/distilbert_colab.ipynb`.

All four runs read the same `config.yaml` (seed, split, max length, epochs, optimizer family) so
results are directly comparable. Do not change these shared settings per-model without updating
this file and telling the team. (DistilBERT's fine-tune learning rate is a documented exception —
see `report/sections/m4_distilbert_architecture.md` — since pretrained Transformers need a much
smaller rate than the shared `0.001`.)

## Results

| Model | Test macro-F1 | Params (total) | Train time | Inference (s / 1000 comments) |
|---|---:|---:|---:|---:|
| Baseline (TF-IDF + LogReg) | 0.6247 | — | — | — |
| BiLSTM (M1) | 0.6331 | 7.73M | 523 s | 0.352 |
| TextCNN (M2) | 0.6393 | 7.58M | 136 s | 0.045 |
| BiLSTM + Attention (M3) | 0.6481 | 7.73M | 254 s | 0.056 |
| DistilBERT (M4) | 0.6803 | 66.37M | 917 s | 0.941 |

All four deep models (plus the baseline) were trained and evaluated on the identical canonical
split (see "Getting the data" above), so this comparison is fair. Full breakdown — per-class
precision/recall/ROC-AUC/PR-AUC, parameter counts including trainable-only, epoch counts, GPU —
in [`results/efficiency_table.csv`](results/efficiency_table.csv) and each model's
`results/<model>_results.json` / `report/sections/m*_*.md`.

DistilBERT has the best macro-F1, especially on the rare classes (`threat`, `identity_hate`), at
the cost of far higher training time, inference latency and model size — see
`report/sections/m4_distilbert_results.md` for the trade-off discussion.

## Reproducibility

- Random seed fixed in `config.yaml` (default `42`) — set for Python, NumPy and the DL framework in
  every training script.
- Tokenizer/vocabulary are fit on the **training split only** (see `src/data_utils.py`) to avoid
  leakage.
- Log library versions in `requirements.txt` (pin versions once your environment is finalised).

## Acknowledgements

- Dataset: Jigsaw / Conversation AI, via Kaggle — cite in the report per Kaggle's licence terms.
- Pretrained weights: DistilBERT (Hugging Face `distilbert-base-uncased`), GloVe embeddings (if used
  for the ablation).
- Note any AI-assisted content (code, drafting) per the assignment's acknowledgement requirement.
