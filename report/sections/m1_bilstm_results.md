# M1 — BiLSTM Results

## Training Performance

The BiLSTM was trained using the shared 70/15/15 multilabel-stratified dataset split. Early stopping was applied based on validation Macro-F1.

The model reached its highest validation Macro-F1 at **Epoch 5**, with a validation Macro-F1 of **0.6605**.

| Epoch | Training Loss | Validation Loss | Validation Macro-F1 | Validation Micro-F1 |
|---:|---:|---:|---:|---:|
| 1 | 0.5012 | 0.3386 | 0.5565 | 0.7265 |
| 2 | 0.2902 | 0.3248 | 0.6295 | 0.7544 |
| 3 | 0.2115 | 0.3716 | 0.6425 | 0.7558 |
| 4 | 0.1654 | 0.4214 | 0.6501 | 0.7543 |
| 5 | 0.1390 | 0.4796 | **0.6605** | **0.7561** |
| 6 | 0.1169 | 0.5960 | 0.6574 | 0.7524 |
| 7 | 0.1027 | 0.6850 | 0.6530 | 0.7515 |
| 8 | 0.0899 | 0.8514 | 0.6502 | 0.7457 |

The training loss decreased continuously during training. However, validation loss started increasing after the early epochs, indicating increasing overfitting. Validation Macro-F1 reached its highest value at Epoch 5 and did not improve afterwards.

The training process stopped after Epoch 8 because validation Macro-F1 did not improve for the configured early-stopping patience.

## Validation Results

At the selected Epoch 5 checkpoint, the validation performance was:

| Metric | Value |
|---|---:|
| Macro-F1 | **0.6605** |
| Micro-F1 | **0.7561** |

### Validation Per-Class Performance

| Label | Precision | Recall | F1 |
|---|---:|---:|---:|
| toxic | 0.8134 | 0.7733 | — |
| severe_toxic | 0.4982 | 0.5750 | — |
| obscene | 0.8390 | 0.8224 | — |
| threat | 0.6290 | 0.5493 | — |
| insult | 0.6834 | 0.7614 | — |
| identity_hate | 0.4416 | 0.5735 | — |

The model achieved stronger performance on the more frequently represented labels such as `toxic` and `obscene`, while the rarer categories such as `severe_toxic`, `threat`, and `identity_hate` remained more difficult to classify.

## Final Test Results

After selecting the best checkpoint and validation thresholds, the frozen model was evaluated on the test set.

| Metric | Test Result |
|---|---:|
| Macro-F1 | **0.6331** |
| Micro-F1 | **0.7453** |
| ROC-AUC | **0.9807** |
| PR-AUC | **0.6547** |

The test Macro-F1 was lower than the validation Macro-F1, decreasing from 0.6605 to 0.6331.

### Test Per-Class Results

| Label | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| toxic | 0.7945 | 0.7720 | 0.7831 | 0.9707 | 0.8676 |
| severe_toxic | 0.4536 | 0.5314 | 0.4894 | 0.9901 | 0.4542 |
| obscene | 0.8189 | 0.8131 | 0.8160 | 0.9876 | 0.8908 |
| threat | 0.4857 | 0.4722 | 0.4789 | 0.9793 | 0.4681 |
| insult | 0.6793 | 0.7604 | 0.7175 | 0.9800 | 0.7674 |
| identity_hate | 0.4654 | 0.5735 | 0.5138 | 0.9763 | 0.4802 |

The per-class results show that the model achieved higher F1 values for `toxic`, `obscene`, and `insult`. The rarer categories had lower F1 values, despite the use of class weighting.

## Efficiency

The BiLSTM contains **7,728,662 trainable parameters**.

The recorded epoch times for the final training run total approximately **523.39 seconds (8.72 minutes)** on a Google Colab Tesla T4 GPU.

The final test inference time was **8.43 seconds** for 23,936 test samples, corresponding to approximately **0.352 seconds per 1,000 samples**.

| Efficiency Measure | Value |
|---|---:|
| Trainable parameters | 7,728,662 |
| Training time | 523.39 seconds |
| Inference time / 1,000 samples | 0.352 seconds |
| GPU | Tesla T4 |

## Figures

The following figures were generated for the BiLSTM model:

- `bilstm_loss_curve.png` — training loss across epochs
- `bilstm_macro_f1_curve.png` — validation Macro-F1 across epochs
- `bilstm_roc_curves.png` — ROC curves for all six labels
- `bilstm_pr_curves.png` — Precision-Recall curves for all six labels
- `bilstm_confusion_matrix_toxic.png`
- `bilstm_confusion_matrix_severe_toxic.png`
- `bilstm_confusion_matrix_obscene.png`
- `bilstm_confusion_matrix_threat.png`
- `bilstm_confusion_matrix_insult.png`
- `bilstm_confusion_matrix_identity_hate.png`

## Discussion

The BiLSTM successfully learned useful sequential representations of toxic comments and achieved a test Macro-F1 of 0.6331. The high ROC-AUC values across the six classes indicate that the model generally separated positive and negative examples effectively when considering the full probability ranking.

However, the lower F1 and PR-AUC values for the minority toxicity categories show the difficulty of classification under severe class imbalance. Class weighting and validation-based threshold tuning were used to improve the handling of these minority classes.

The difference between validation and test Macro-F1 also shows that validation performance did not completely represent the final test performance. Therefore, the frozen test results should be used as the final performance figures when comparing M1 with the other models.
