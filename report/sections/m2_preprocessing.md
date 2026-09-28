# 4. Data Preprocessing

*Owner: M2 (Tharsiga Ranganathan). All models share this pipeline, so they use the same split, seed (42) and `max_length` (128). Code: `src/data_utils.py`.*

## 4.1 Dataset and exploratory analysis

We use the Jigsaw Toxic Comment Classification dataset (Kaggle). It has 159,571 Wikipedia talk-page comments. Each comment has six binary labels: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult` and `identity_hate`. A comment can have several labels at once, so this is a **multi-label** problem.

**Class imbalance.** The labels are very imbalanced. The table shows the positive counts in our validation split (23,936 comments):

| Label | Positives | % of split |
|---|---:|---:|
| toxic | 2,294 | 9.58% |
| severe_toxic | 240 | 1.00% |
| obscene | 1,267 | 5.29% |
| threat | 71 | 0.30% |
| insult | 1,182 | 4.94% |
| identity_hate | 211 | 0.88% |

`threat` has only 71 positives, so one extra correct or wrong example changes its F1 by about 0.01. This is why we (a) use class weights (`pos_weight`, computed from the training split) in the loss, (b) tune a separate decision threshold for each class, and (c) report **macro-F1** as the main metric, because it gives each class equal weight.

**Comment length.** After tokenization, the median comment has 44 tokens, the mean is 83, and the 95th percentile is 279. Toxic comments are shorter than clean ones (median 29 vs 46 tokens).

<!-- TODO: add EDA figure(s) from the EDA notebook, e.g. token-length histogram and label counts. -->

## 4.2 Train / validation / test split

We use one stratified split for all models (`make_or_load_split`, seed 42):

| Split | Comments | Batches (size 32) |
|---|---:|---:|
| Train | 111,699 | 3,491 |
| Validation | 23,936 | 748 |
| Test | 23,936 | 748 |

Stratification keeps the rare labels (for example `threat`) in similar proportions in every split. We saved the split to CSV so every model reads the same rows.

**Test set rule.** We choose all model settings, epochs and thresholds on **validation only**. The test set is evaluated once, at the end, for all models together.

<!-- TODO: confirm with Shevoni that her split gives the same counts, and what column(s) the stratification uses. -->

## 4.3 Cleaning and tokenization

`clean_text` does four simple things:

1. **Lowercase** the text, because GloVe 6B is uncased.
2. **Remove URLs** (anything starting with `http://`, `https://` or `www.`).
3. **Remove HTML tags.**
4. **Collapse whitespace**, so newlines, tabs and repeated spaces become one space.

We deliberately **keep stopwords** and do **no stemming**. Small words carry meaning here: "you are **not** stupid" and "you are stupid" differ only by a stopword.

`tokenize` cleans the text and then splits it with one regular expression, `[a-z0-9']+|[^\sa-z0-9']`:

- A **word token** is a run of letters `a–z`, digits and apostrophes, so contractions such as "don't" stay as one token.
- **Every other non-space character is its own one-character token.** This includes punctuation (so "!!!" becomes three `!` tokens the model can use as a signal), emoji, and letters from other scripts.

We use the same functions for every split. The cleaning does not learn anything from the data, so it cannot leak information.

## 4.4 Vocabulary (built on train only)

`build_vocab` counts tokens in the **training split only** and keeps tokens that appear at least twice (`min_vocab_freq = 2`). Index 0 is `<pad>` and index 1 is `<unk>`. The final vocabulary has **74,916** entries, including these two.

**Why train only?** If we built the vocabulary on the full dataset, it would have 94,883 entries. That means about 20,000 tokens would come from the validation and test comments. The model would then "know" words it should not see during training, which is data leakage. With a train-only vocabulary, unseen words map to `<unk>`, like in real use.

| Measure | Value |
|---|---:|
| Vocabulary size (train only, min freq 2) | 74,916 |
| Vocabulary size if built on full data | 94,883 |
| Token coverage on train | 99.12% |
| OOV rate on validation | 1.74% |

## 4.5 Encoding, padding and truncation

`encode` turns each token into its vocabulary index (unknown tokens become `<unk>` = 1). Every comment is then fixed to **128 tokens**:

- **Longer comments are truncated at the end** (we keep the first 128 tokens).
- **Shorter comments are padded at the end** with `<pad>` = 0.

`encode` also returns the real length of each comment, and this length is at least 1. That way an empty comment does not crash LSTM packing.

We chose 128 because it covers most comments (median 44 tokens) and keeps training fast. The cost is that **16.5%** of comments are longer than 128 tokens and lose their end.

## 4.6 Dataset, DataLoader and padding mask

`ToxicDataset` encodes every comment **once**, when it is created. It stores `input_ids`, `length` and the six labels as tensors, so training does not tokenize again every epoch.

`collate_fn` is shared by BiLSTM, TextCNN and BiLSTM + Attention. For each batch it returns:

| Output | Shape | Meaning |
|---|---|---|
| `input_ids` | [B, 128] | token indices |
| `lengths` | [B] | real length of each comment |
| `mask` | [B, 128] | `True` for real tokens, `False` for padding |
| `labels` | [B, 6] | the six 0/1 labels (float) |

The mask is built from `lengths`, so an empty comment still has one `True` position. This prevents a NaN in the attention softmax (M3). TextCNN uses the same mask to ignore padding positions in max-over-time pooling. BiLSTM uses `lengths` for sequence packing.

`build_dataloaders` creates the three loaders with batch size 32. Only the training loader is shuffled, and it uses a generator seeded with 42, so the batch order is reproducible. Validation and test are not shuffled.

## 4.7 Pre-trained word embeddings (GloVe)

`load_glove_embeddings` builds a [74,916 × 100] embedding matrix from **GloVe 6B, 100 dimensions**:

- A word **found in GloVe** gets its GloVe vector. This covers 54,950 of 74,916 words (73.35%).
- A word **not in GloVe** (including `<unk>`) gets a small random vector from U(−0.25, 0.25), following Kim (2014). The random generator is seeded with 42.
- **`<pad>`** gets an all-zero vector.

The embedding layer is **fine-tuned**, not frozen (`freeze=False`), so the model can adapt word meanings to toxic language. The matrix is copied (`.clone()`) before training, so one run never changes the GloVe matrix used by another run.

The ablation in Section 7 shows that GloVe helps TextCNN: validation macro-F1 is 0.6486 with GloVe and 0.6259 with random initialisation, and training converges about three times faster.

## 4.8 Known limitations

- **Contraction mismatch.** GloVe splits "don't" into "do" + "n't", but our tokenizer keeps "don't". Some common words therefore get no GloVe vector.
- **Non-Latin text.** The tokenizer is designed for English. Text in other scripts (for example Cyrillic) breaks into single meaningless characters. Some `identity_hate` examples in the error analysis are partly Cyrillic.
- **Truncation.** 16.5% of comments are cut at 128 tokens, and we keep only the start. In the error analysis, some missed threats may be in the removed part.
- **Label noise.** The error analysis found clean-labelled personal attacks and standard warnings labelled as threats. Preprocessing cannot fix this, but it limits the best score any model can reach.
