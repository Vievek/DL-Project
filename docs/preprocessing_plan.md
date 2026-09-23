# Preprocessing Plan (M2 — Tharsiga Ranganathan)

Applies to the three word-level models: **BiLSTM (M1), TextCNN (M2), BiLSTM + Attention (M3)**.
DistilBERT (M4) uses its own WordPiece tokenizer and does not follow steps 2–5.

All numbers below come from `notebooks/eda_preprocessing_m2.ipynb` (full `train.csv`, 159,571 comments).

---

## 0. Order of operations (to prevent data leakage)

1. M1 removes duplicates and creates the fixed 70/15/15 split (train / val / test).
2. Cleaning (step 1) is applied to all three splits. It uses fixed rules, so it learns nothing from the data.
3. The vocabulary (step 3), class weights (step 6) and anything else that is *learned* are built from the **train split only**.
4. Val and test are encoded with the train vocabulary. Words not in it become `<unk>`.

## 1. Text cleaning

| Decision | Choice | Reason (evidence) |
|---|---|---|
| Lowercase | **Yes** | GloVe 6B is uncased. `IDIOT` would not match `idiot` and would become `<unk>`. Trade-off: we lose "shouting" (all caps) as a signal. We note this as a limitation. |
| Punctuation | **Keep**, as separate tokens | `!` appears 105,576 times. Repeated `!!!` is a sign of anger. GloVe has vectors for punctuation. |
| Newlines `\n`, tabs | **Replace with a space** | GloVe has no vector for `\n`, so it would carry no meaning. The first comment in the data already shows `\n` inside the text. |
| URLs | **Remove** | Word clouds showed URLs glued to words (e.g. `...http`), creating junk tokens. |
| HTML tags | **Remove** | Formatting only, no meaning. |
| Extra spaces | **Collapse to one space** | Tidy tokenization. |
| Stopwords | **Keep** | `not` is in the top 20 tokens. "not stupid" and "stupid" mean different things (negation). |
| Stemming / lemmatization | **No** | GloVe stores full words, and stems like `stupid → stupid`, `stupidly → stupid` would lose matches. |
| Spelling correction | **No** | Out of scope. Misspellings (e.g. `fggt`, `bastered`) become `<unk>`. We note this as a limitation. |

## 2. Tokenizer

Simple regex word tokenizer, the same one used in the EDA:

```python
re.findall(r"[a-z0-9']+|[^\sa-z0-9']", text)
```

Example: `"You are SO dumb!!\nGo away."` → `['you', 'are', 'so', 'dumb', '!', '!', 'go', 'away', '.']`

All three word-level models use the exact same tokenizer so the comparison is fair.

## 3. Vocabulary

- Built from the **train split only**.
- `min_freq = 2`: a word must appear at least twice to get its own ID.
- Special tokens: `<pad>` = 0, `<unk>` = 1.
- Saved to a file (`vocab.json`) so every model uses the identical vocabulary.

Evidence (full dataset, an estimate only):

| Measure | Value |
|---|---|
| Total tokens | 13,250,066 |
| Unique tokens | 201,353 |
| Vocab with min_freq = 2 | 94,883 |
| Text still covered with min_freq = 2 | 99.20% |

`min_freq = 2` removes more than half the vocabulary (mostly typos, spam and usernames) but keeps 99.2% of the text. The final train-only vocabulary will be somewhat smaller.

## 4. Sequence length

- `max_len = 128` (shared across all 4 models, from `config.yaml`).
- Shorter comments: **pad at the end** with `<pad>` (0).
- Longer comments: **cut the end** and keep the first 128 tokens.

Evidence:

| Measure (tokens) | Value |
|---|---|
| Median | 44 |
| 95th percentile | 279 |
| 99th percentile | 707 |
| Comments longer than 128 | 16.5% |
| Median, clean comments | 46 |
| Median, toxic comments | 29 |

83.5% of comments fit completely. Toxic comments are shorter than clean ones, so truncation affects them less. Future work: keep the start and the end of long comments (head + tail truncation).

## 5. Embeddings

- **GloVe 6B, 100 dimensions** (`glove.6B.100d.txt`) → set `embed_dim: 100` in `config.yaml`.
  The current value (128) does not match any GloVe file and must change.
- Why 100d instead of 200d: faster to load and train on a free Colab T4, and half the embedding size.
  With about 95k words, the embedding table is about 95,000 × 100 ≈ 9.5M parameters, which is most of TextCNN's size.
- Words in our vocab but not in GloVe: small random vectors.
- `<pad>` row: all zeros.
- Embeddings are fine-tuned during training.
- We report GloVe coverage (% of vocab found in GloVe) when loading.
- Optional ablation (Day 6): GloVe vs random embeddings, same everything else.

## 6. Class weights

- `pos_weight` for each label = (negative count) / (positive count), computed on the **train split only**.
- Passed to `BCEWithLogitsLoss` (used by M3's shared training script).
- Needed because the classes are very unbalanced: `threat` has 478 positive comments and `identity_hate` 1,405, out of 159,571.

## 7. Functions to provide in `src/data_utils.py` (agree with M3)

| Function | Input | Output |
|---|---|---|
| `clean_text(text)` | raw string | cleaned string |
| `tokenize(text)` | cleaned string | list of tokens |
| `build_vocab(train_texts, min_freq=2)` | train texts only | dict word → id |
| `encode(texts, vocab, max_len=128)` | texts + vocab | LongTensor of shape (N, 128) |
| `load_glove(path, vocab, dim=100)` | GloVe file + vocab | FloatTensor of shape (vocab_size, 100) |
| `compute_pos_weight(train_labels)` | train label matrix | FloatTensor of shape (6,) |

## 8. Decisions the team must confirm

1. GloVe **100d** and `embed_dim: 100` in `config.yaml`.
2. **Lowercase = True** (the report must explain the lost "caps" signal).
3. Who downloads GloVe (`glove.6B.zip`, about 820 MB) and puts `glove.6B.100d.txt` in the shared Drive folder.
4. Function names above, so M3's `train_eval.py` can call them.
