# 6. Model Architecture: DistilBERT, fine-tuned (M4)

*Owner: M4 (Pancharatnam Deepatharshan / DEEPDEV). Code: `src/models/distilbert.py`, training
entry point `src/train_eval.py --model distilbert --data-pipeline hf`.*

## Motivation

The other three models (BiLSTM, TextCNN, BiLSTM+Attention) all learn word representations from
scratch, on only the ~112k training comments available. DistilBERT instead starts from a
Transformer encoder already pretrained on a large general-English corpus, so it brings in
knowledge of grammar, word sense, and context that our small dataset alone cannot teach. The
comparison this gives us for the report is: how much does transfer learning from a pretrained
language model buy over training an architecture from scratch on the same data and the same
evaluation protocol?

Unlike M1–M3, DistilBERT does **not** use the shared vocabulary/GloVe pipeline in
`data_utils.py`. It uses its own pretrained WordPiece tokenizer, so the vocabulary, the notion of
an "unknown" token, and even how a word is split into subword pieces are all inherited from
pretraining rather than built from our training split. This is why `train_eval.py` has a
`--data-pipeline hf` flag: it skips `build_vocab`/`build_dataloaders` entirely for this model.

## Architecture

Input comments are tokenized with the Hugging Face `distilbert-base-uncased` tokenizer
(WordPiece), truncated/padded to `max_length=128` (`config.yaml`), same as the other models so
truncation is not a confound in the comparison.

- **Encoder**: pretrained `distilbert-base-uncased` — 6 Transformer layers, hidden size 768, 12
  attention heads. This is the distilled version of BERT-base: roughly half the layers, trained to
  match BERT's behaviour via knowledge distillation, at ~60% of the inference time.
- **Pooling**: the hidden state at the `[CLS]` token position (position 0) is taken as the
  sequence representation — the standard approach for classification with BERT-family models.
- **Dropout**: 0.1 on the pooled `[CLS]` vector.
- **Classifier head**: a single linear layer, 768 → 6, producing one raw logit per label (no
  softmax — multi-label, so each label is an independent sigmoid decision, same as the other 3
  models).

## Training regimen

Fine-tuning a pretrained encoder needs care: if the whole network is trained end-to-end from step
one, the randomly-initialised classifier head sends large, noisy gradients back into the carefully
pretrained weights and can partially destroy what pretraining learned. We train in two stages:

1. **Head-only (1 epoch).** The encoder is frozen (`requires_grad=False`); only the 768→6 linear
   head trains, at `lr=1e-3` (the shared config's learning rate). This lets the head reach a
   reasonable starting point before touching the pretrained weights.
2. **Full fine-tune (2 epochs).** The encoder is unfrozen and the whole model trains end-to-end at
   `lr=2e-5` — two orders of magnitude below the shared config's `0.001`. This is a deliberate,
   documented **per-model override**: pretrained Transformers are known to diverge or forget their
   pretraining at the learning rates that work for from-scratch models like the other 3.

Other settings follow the shared config: `BCEWithLogitsLoss` with per-class `pos_weight` computed
from the training split (capped at 50× so the rarest class, `threat`, does not dominate the loss),
batch size 32, and early stopping on validation macro-F1 (patience 3). Training uses fp16 mixed
precision (`torch.autocast` + `GradScaler`) on GPU, which roughly halves training time on the
Colab T4 with no measurable accuracy cost. A checkpoint is saved every epoch to Google Drive, since
Colab's local disk and runtime are wiped on disconnect.

## Parameter count

- **Total parameters**: 66,367,494 (dominated by the pretrained encoder).
- **Classifier head only**: 4,614 (768 × 6 weights + 6 biases) — the only part trained during
  stage 1.

This is roughly 9× the parameter count of BiLSTM+Attention (7.7M) and far larger than TextCNN, but
almost all of those parameters are *pretrained*, not learned from our ~112k comments — the fair
comparison is not "more parameters" but "more parameters that already encode general language
knowledge before training even starts."
