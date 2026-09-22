# Toxic Comment Classification — SE4050 Deep Learning Group Assignment

Multi-label toxic comment classification on the [Jigsaw Toxic Comment Classification
Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) dataset. Four
distinct deep learning architectures are trained and compared, per the SE4050 assignment brief
(Supervised Deep Learning category — at least 4 distinct models).

## Team

| Member | Model | Notes |
|---|---|---|
| TODO name | M1 — BiLSTM | |
| TODO name | M2 — TextCNN | |
| TODO name | M3 — BiLSTM + attention (or small Transformer encoder) | |
| DEEPDEV | M4 — DistilBERT (fine-tuned) | |

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

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Getting the data

1. Download the Jigsaw Toxic Comment Classification Challenge dataset from Kaggle:
   https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data
2. Place `train.csv` (and `test.csv`/`test_labels.csv` if used) into `data/raw/`.
3. Run the split script to generate the shared, fixed train/val/test split (see `src/data_utils.py`).

Raw data and any generated split files are **not committed** — see `.gitignore`. Everyone must use
the exact same split, so once it's generated, share the CSV via the team Drive and note its checksum
in `results/`.

## Branches

One branch per model owner, created off `main` from the start (`scripts/setup_branches.sh`):

| Branch | Owner | Model |
|---|---|---|
| `model/bilstm` | Teammate A | M1 — BiLSTM |
| `model/textcnn` | Teammate B | M2 — TextCNN |
| `model/bilstm-attention` | Teammate C | M3 — BiLSTM + attention |
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
python -m src.train_eval --model distilbert --config config.yaml
```

All four runs read the same `config.yaml` (seed, split, max length, epochs, optimizer family) so
results are directly comparable. Do not change these shared settings per-model without updating
this file and telling the team.

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
