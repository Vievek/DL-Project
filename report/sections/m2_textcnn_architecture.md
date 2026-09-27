# 6. Model 2: TextCNN

*Owner: M2 (Tharsiga Ranganathan). Code: `src/models/textcnn.py`, training script `src/textcnn_train.py`.*

## 6.1 Why TextCNN?

TextCNN (Kim, 2014) is a convolutional network for text. A convolution filter slides over the sentence and looks at a few words at a time, like a small window. Each filter learns to fire on one kind of short phrase (an n-gram), for example an insult word followed by "you". Max-over-time pooling then keeps only the strongest match for that filter, **wherever it appears** in the comment.

This fits toxic comment detection well. Much toxicity is carried by short, local phrases (swear words, slurs, insults), and a comment is toxic if such a phrase appears anywhere. TextCNN is also fast, because all windows are processed in parallel. It is not step-by-step like an LSTM. This makes it a useful contrast to the recurrent models (M1 BiLSTM, M3 BiLSTM + Attention), which read the comment in order.

## 6.2 Architecture

For one batch of B comments (final configuration, filter widths 2, 3 and 4):

| Step | Layer | Output shape |
|---|---|---|
| 1 | Input token ids (from `collate_fn`) | [B, 128] |
| 2 | Embedding (GloVe 6B 100d, fine-tuned) | [B, 128, 100] → permuted to [B, 100, 128] |
| 3 | Three parallel convolutions, 100 filters each, widths 2 / 3 / 4, + ReLU | [B, 100, 127], [B, 100, 126], [B, 100, 125] |
| 4 | Masked max-over-time pooling (one value per filter) | 3 × [B, 100] |
| 5 | Concatenate | [B, 300] |
| 6 | Dropout (p = 0.5) | [B, 300] |
| 7 | Linear layer | [B, 6] raw logits |

<!-- TODO (optional): add a simple diagram of this pipeline. -->

**Different filter widths.** A width-2 filter sees pairs of words ("shut up"), a width-3 filter sees three words ("go to hell"), and a width-4 filter sees four. Using the three widths together lets the model catch phrases of different lengths.

**Masked max pooling.** Comments are padded to 128 tokens. Without a mask, a filter could get its maximum from padding, which carries no meaning. We use the padding mask from `collate_fn`: a window is kept only if it **starts on a real token**, and all other windows are set to −∞ before the max. Every comment has at least one real token (Section 4.5), so each filter always has at least one valid window.

**Multi-label output.** The model outputs 6 raw scores (logits), one per label, and uses **no softmax**. The labels are not mutually exclusive: a comment can be both `obscene` and `insult`. We train with `BCEWithLogitsLoss` with `pos_weight` (one independent sigmoid per label). At prediction time we apply a sigmoid to each logit and compare it with that label's tuned threshold (Section 7).

## 6.3 Final configuration and size

| Setting | Value |
|---|---|
| Embedding | GloVe 6B, 100d, fine-tuned (`freeze=False`) |
| Filters per width | 100 |
| Filter widths | 2, 3, 4 |
| Dropout | 0.5 |
| Max sequence length | 128 |
| Optimizer | Adam, learning rate 0.001 |
| Batch size | 32 |
| Loss | BCEWithLogitsLoss with per-class `pos_weight` |
| Epochs | up to 10, early stopping (patience 3) on validation macro-F1 |
| Seed | 42 |

This configuration (C4) was chosen from five candidates (Section 7).

| Part | Trainable parameters |
|---|---:|
| Embedding (74,916 × 100) | 7,491,600 |
| Convolutions (20,100 + 30,100 + 40,100) | 90,300 |
| Linear layer (300 × 6 + 6) | 1,806 |
| **Total** | **7,583,706** |
| Total without embedding | 92,106 |

Almost all parameters (98.8%) are in the embedding table. The part that actually "detects" toxic patterns is very small, only 92,106 parameters.

## 6.4 Implementation note: faster convolution with unfold + matmul

With PyTorch's standard `nn.Conv1d` on the Colab T4 GPU, one training step took about **85 ms**, so one epoch took about **296 s**. Profiling showed the problem was in the backward pass: cuDNN automatically chose a slow FFT-based algorithm for these convolution shapes.

We therefore compute the same convolution in a different way (`_conv_unfold`):

1. `unfold` cuts the input into all windows of k consecutive tokens, giving a matrix with one row per window position.
2. One matrix multiplication with the Conv1d's own weights (reshaped) gives the filter responses for all windows at once.

This is mathematically the same operation as the convolution, and it uses the same weights. The maximum difference from `nn.Conv1d` was **7 × 10⁻⁶**, which is floating-point rounding. Matrix multiplication is very fast on GPUs, so a step dropped from **85 ms to 4.5 ms**, and an epoch from **~296 s to ~23 s** (about 13× faster). This made the hyperparameter search and the ablation study possible within our time limit.

*Real-world picture:* the math is the same, like taking a direct road instead of a detour. We arrive at the same place, only much faster.

## 6.5 Limitations of the architecture

- **Short context.** The widest filter sees only 4 tokens. TextCNN cannot understand meaning spread over a whole sentence. The error analysis (Section 7) shows this clearly: implicit threats without threat keywords, such as "ban me and die", are missed.
- **Word order is lost after pooling.** Max pooling only records *that* a pattern appeared, not *where*, or in what order relative to other patterns.
- **Keyword triggering.** Because single strong words dominate the pooled features, the model also fires on profanity in quotes, song lyrics or neutral discussion (false positives in Section 7).

These weaknesses are expected for TextCNN. They are the reason we compare it with the recurrent models, the attention model and DistilBERT, which can use longer context.

## References

Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification. *Proceedings of EMNLP 2014*, 1746–1751.
