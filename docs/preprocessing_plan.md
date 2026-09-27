# Preprocessing Plan (M2 — Tharsiga Ranganathan)

Applies to the three word-level models: **BiLSTM (M1), TextCNN (M2), BiLSTM + Attention (M3)**.
DistilBERT (M4) uses its own WordPiece tokenizer and does not follow steps 2–5.

Numbers in sections 1–6 come from `notebooks/eda_preprocessing_m2.ipynb` (full `train.csv`, 159,571 comments) and were used to make the decisions.
**Section 9 has the final numbers from the implemented pipeline** (train split only). Use those in the report.

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

`min_freq = 2` removes more than half the vocabulary (mostly typos, spam and usernames) but keeps 99.2% of the text. The final train-only vocabulary is 74,916 (section 9).

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

- **GloVe 6B, 100 dimensions** (`glove.6B.100d.txt`), with `embed_dim: 100` in `config.yaml` (done).
- Why 100d instead of 200d: faster to load and train on a free Colab T4, and half the embedding size.
  With the final vocabulary, the embedding table is 74,916 × 100 = 7,491,600 parameters, which is about 99% of TextCNN's size.
- Words in our vocab but not in GloVe: small random vectors, U(−0.25, 0.25) (Kim, 2014), seeded with 42.
- `<pad>` row: all zeros.
- Embeddings are fine-tuned during training.
- GloVe coverage is printed when loading (section 9).
- Ablation (done): GloVe vs random embeddings, with everything else the same. See section 9.

## 6. Class weights

- `pos_weight` for each label = (negative count) / (positive count), computed on the **train split only**.
- Passed to `BCEWithLogitsLoss`.
- Needed because the classes are very unbalanced: `threat` has 478 positive comments and `identity_hate` 1,405, out of 159,571.

## 7. Functions in `src/data_utils.py` (implemented)

| Function | Input | Output |
|---|---|---|
| `clean_text(text)` | raw string | cleaned string |
| `tokenize(text)` | raw string (cleans it first) | list of tokens |
| `build_vocab(train_texts, min_freq=2)` | train texts only | dict token → id (`<pad>`=0, `<unk>`=1) |
| `save_vocab(vocab, path)` / `load_vocab(path)` | vocab / path | `vocab.json` / dict |
| `oov_rate(texts, vocab)` | texts + vocab | fraction of tokens that are `<unk>` |
| `encode(text, vocab, max_len=128)` | one text + vocab | (list of 128 ids, real length ≥ 1) |
| `ToxicDataset(df, vocab, label_cols, max_len)` | split DataFrame | items (input_ids, length, labels[6]), encoded once |
| `collate_fn(batch)` | list of items | (input_ids [B,128], lengths [B], mask [B,128] bool, labels [B,6]) |
| `build_dataloaders(cfg, train_df, val_df, test_df, vocab)` | config + splits | train / val / test DataLoaders (only train shuffled, seeded) |
| `load_glove_embeddings(vocab, glove_path, embed_dim=100, seed=42)` | GloVe file + vocab | FloatTensor (vocab_size, 100) |
| `compute_class_weights(train_df, label_cols)` | train split | `pos_weight` per label |

`mask` is `True` for real tokens. It is used by TextCNN (masked max pooling) and by the attention model (M3). `lengths` is used by BiLSTM packing.

## 8. Team decisions (status)

1. GloVe **100d** and `embed_dim: 100` in `config.yaml`: **done**.
2. **Lowercase = True**: **done**. The report explains the lost "caps" signal.
3. GloVe download: **done**. `glove.6B.100d.txt` is in the shared Drive folder (`DL-Project/embeddings/`).
4. Function names above: **implemented**. The shared `train_eval.py` does not use them yet (reported to the team lead).
5. Split counts (111,699 / 23,936 / 23,936): **waiting for M1 to confirm** that her split gives the same counts.

## 9. Final numbers (implemented pipeline)

| Measure | Value |
|---|---|
| Split (train / val / test), stratified, seed 42 | 111,699 / 23,936 / 23,936 |
| Batches of 32 (train / val / test) | 3,491 / 748 / 748 |
| Vocabulary (train only, min_freq 2, incl. `<pad>`/`<unk>`) | 74,916 |
| Vocabulary if built on full data (would leak) | 94,883 |
| Token coverage on train | 99.12% |
| OOV rate on validation | 1.74% |
| Vocab words found in GloVe | 54,950 / 74,916 (73.35%) |
| Comments longer than 128 tokens (truncated) | 16.5% |

**GloVe ablation (TextCNN C4, same seed, validation):** GloVe macro-F1 0.6486 vs random 0.6259 (+0.023). The biggest gain was `identity_hate` (+0.071), and GloVe converged at epoch 3 vs 9. Details: `results/textcnn_ablation.csv`, report Section 7.6.

**Known limitations:**
- GloVe splits "don't" into "do" + "n't", but our tokenizer keeps "don't", so some common contractions have no GloVe vector.
- Text in other scripts (e.g. Cyrillic) becomes single meaningless characters.
- 16.5% of comments lose their end to truncation. Some missed threats in the error analysis may be in the cut part.
- All-caps "shouting" is lost by lowercasing.
