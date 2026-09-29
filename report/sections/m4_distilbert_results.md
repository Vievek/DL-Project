# 7. Results and Comparison: DistilBERT (M4)

*Owner: M4 (Pancharatnam Deepatharshan / DEEPDEV). Result file: `results/distilbert_results.json`.
Trained on Google Colab, Tesla T4 GPU, fp16 mixed precision.*

## Overall performance

DistilBERT achieved a **test macro-F1 of 0.6803** and **test micro-F1 of 0.7809** — the highest
macro-F1 of all four models on the shared test split, ahead of BiLSTM+Attention's 0.6481 and the
TF-IDF + Logistic Regression baseline's 0.6247 on the same canonical split. Validation macro-F1
was 0.6705, close to the test score, meaning no meaningful overfitting to the validation split from
threshold tuning.

All numbers below are computed on the **canonical 70/15/15 multi-label stratified split (seed
42)** shared by the whole team (`results/split_checksums.txt`) — the same rows every other model
was trained and evaluated on, so this comparison is fair.

## Per-class results (test split)

| Class | Precision | Recall | ROC-AUC | PR-AUC | Threshold |
|---|---:|---:|---:|---:|---:|
| toxic | 0.846 | 0.814 | 0.986 | 0.914 | 0.86 |
| severe_toxic | 0.427 | 0.699 | 0.992 | 0.469 | 0.95 |
| obscene | 0.822 | 0.862 | 0.994 | 0.919 | 0.92 |
| threat | 0.451 | 0.708 | 0.995 | 0.484 | 0.95 |
| insult | 0.698 | 0.820 | 0.988 | 0.825 | 0.92 |
| identity_hate | 0.482 | 0.711 | 0.990 | 0.603 | 0.95 |

ROC-AUC is above 0.98 for every class — DistilBERT ranks toxic comments above non-toxic ones very
reliably, including for the rarest classes. The gap between ROC-AUC and PR-AUC on `severe_toxic`
and `threat` (both above 0.99 ROC-AUC but under 0.5 PR-AUC) is the class-imbalance signature we see
across all four models: when positives are very rare, even a model that ranks them well produces
many false positives at any threshold that also catches most true positives. Recall on the rare
classes (0.70–0.71) is noticeably higher than on the from-scratch models (Section 7, M2/M3), likely
because the pretrained encoder already has some notion of these words' meaning before seeing a
single labelled example.

## Training curves and the threshold effect

| Stage | Epoch | Train loss | Val macro-F1 @ 0.5 | Epoch time |
|---|---:|---:|---:|---:|
| head-only | 1 | 0.429 | 0.417 | 92 s |
| fine-tune | 1 | 0.230 | 0.501 | 359 s |
| fine-tune | 2 | 0.157 | 0.537 | 360 s |

Note that the val macro-F1 logged *during* training (0.537 at the end of epoch 2) is measured at a
flat 0.5 decision threshold, while the final reported validation score (0.6705) uses per-class
thresholds tuned on validation via a precision-recall sweep (same protocol as Section 5/shared
evaluation). The 0.13 gap between these two numbers is entirely the effect of threshold tuning, not
extra training — it illustrates why a flat 0.5 cutoff is a poor choice under heavy class imbalance,
consistent with what M2 found for TextCNN (Section 7.3).

Training loss drops steadily and validation F1 improves monotonically across all 3 epochs with no
sign of overfitting yet — a longer fine-tune (more than 2 epochs) may improve results further,
which is a natural next step if compute budget allows.

## Efficiency

| Metric | Value |
|---|---|
| Total parameters | 66,367,494 |
| Total train time (3 epochs, ~112k rows) | 917 s (~15.3 min) |
| Inference time | 0.94 s per 1000 comments |
| Hardware | Tesla T4 (Colab), fp16 |

DistilBERT is by far the most expensive model to train and run of the four: roughly 15 minutes of
GPU time to train versus TextCNN's ~2.3 minutes for its best config (Section 7.2), and about 19×
slower at inference than BiLSTM+Attention's ~0.05 s per 1000 comments (Section 7, M3). This is the
central trade-off for the report's discussion: DistilBERT buys the best macro-F1 and the best
recall on rare classes, at a real cost in training time, inference latency, and model size — a
lightweight model like TextCNN or BiLSTM+Attention would be the practical choice for a low-latency
moderation pipeline, while DistilBERT would suit an offline or batch setting where accuracy on rare,
high-severity classes (`threat`, `identity_hate`) matters more than speed.
