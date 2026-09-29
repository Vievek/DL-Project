# M1 — BiLSTM Results

## Training Performance

The BiLSTM was initially trained using the shared 70/15/15 multilabel-stratified dataset split. Early stopping was applied based on validation Macro-F1.

For the original final training run, the model used an embedding dimension of 100, hidden dimension of 128, one BiLSTM layer, dropout of 0.3, and embedding dropout of 0.1.

The model reached its highest validation Macro-F1 at **Epoch 5**, with a validation Macro-F1 of **0.6605**.

| Epoch | Training Loss | Validation Loss | Validation Macro-F1 | Validation Micro-F1 |
| ----: | ------------: | --------------: | ------------------: | ------------------: |
|     1 |        0.5012 |          0.3386 |              0.5565 |              0.7265 |
|     2 |        0.2902 |          0.3248 |              0.6295 |              0.7544 |
|     3 |        0.2115 |          0.3716 |              0.6425 |              0.7558 |
|     4 |        0.1654 |          0.4214 |              0.6501 |              0.7543 |
|     5 |        0.1390 |          0.4796 |          **0.6605** |          **0.7561** |
|     6 |        0.1169 |          0.5960 |              0.6574 |              0.7524 |
|     7 |        0.1027 |          0.6850 |              0.6530 |              0.7515 |
|     8 |        0.0899 |          0.8514 |              0.6502 |              0.7457 |

The training loss decreased continuously during training, while the validation loss increased after the early epochs. Validation Macro-F1 reached its highest value at Epoch 5 and declined afterwards, indicating increasing overfitting.

The training process stopped after Epoch 8 because validation Macro-F1 did not improve for the configured early-stopping patience. The best checkpoint from this run was therefore Epoch 5.

## Hyperparameter Search

A five-configuration hyperparameter search was conducted to investigate the effect of LSTM hidden dimension and dropout.

| Configuration | Hidden Dimension | Dropout | Best Validation Macro-F1 |
| ------------- | ---------------: | ------: | -----------------------: |
| 1             |               64 |     0.3 |                   0.5606 |
| 2             |               64 |     0.5 |                   0.5463 |
| 3             |              128 |     0.3 |                   0.5682 |
| 4             |              128 |     0.5 |                   0.5687 |
| 5             |              256 |     0.3 |               **0.5861** |

The configuration with **256 hidden units and 0.3 dropout** produced the highest validation Macro-F1 during the hyperparameter search and was selected for retraining.

The hyperparameter-search Macro-F1 values were calculated using a fixed **0.5 decision threshold**. They therefore represent HPO comparison scores rather than the final threshold-tuned evaluation.

## Selected Configuration Retraining

The selected 256-hidden-unit configuration was retrained using the same dataset split, GloVe embeddings, class weights, optimizer, learning rate, and batch size.

The retrained model achieved its highest validation Macro-F1 of **0.5701 at Epoch 10**.

| Metric                            |      Value |
| --------------------------------- | ---------: |
| Selected hidden dimension         |        256 |
| Dropout                           |        0.3 |
| Best epoch                        |         10 |
| Best validation Macro-F1          | **0.5701** |
| Validation Micro-F1 at best epoch |     0.6859 |

The difference between the HPO result of 0.5861 and the retraining result of 0.5701 reflects variation between separate training runs. The HPO result was used for configuration selection, while the retraining result represents the subsequently trained selected configuration.

The retraining validation scores were also calculated using a fixed **0.5 decision threshold**.

## Class-Weight Ablation

An ablation experiment was performed using the selected BiLSTM architecture to examine the effect of class weighting.

| Configuration         | Best Validation Macro-F1 | Best Epoch |
| --------------------- | -----------------------: | ---------: |
| With class weights    |                   0.5701 |         10 |
| Without class weights |               **0.6319** |          4 |

For this ablation, both configurations were evaluated using the same fixed **0.5 decision threshold**.

The no-class-weight configuration achieved a higher validation Macro-F1 in this experiment. This result is reported as an experimental observation and does not imply that class weighting is generally ineffective; it reflects the outcome obtained on the specified validation split and experimental configuration.

## Validation Results

The original final validation evaluation used the Epoch 5 checkpoint from the 128-hidden-unit training run. For the final evaluation, per-class decision thresholds were tuned using the validation set and then frozen before test evaluation.

| Metric   |      Value |
| -------- | ---------: |
| Macro-F1 | **0.6605** |
| Micro-F1 | **0.7561** |

The model achieved stronger performance on the more frequently represented labels such as `toxic` and `obscene`, while minority categories such as `severe_toxic`, `threat`, and `identity_hate` remained more difficult to classify.

The final threshold-tuning procedure was performed independently for each class using the validation predictions. No test-set information was used to select the thresholds.

## Final Test Results

The final test results below correspond to the **original 128-hidden-unit BiLSTM checkpoint**, selected at Epoch 5, with per-class thresholds tuned on the validation set and then frozen for the test evaluation.

| Metric   | Test Result |
| -------- | ----------: |
| Macro-F1 |  **0.6331** |
| Micro-F1 |  **0.7453** |
| ROC-AUC  |  **0.9807** |
| PR-AUC   |  **0.6547** |

The test Macro-F1 was lower than the corresponding validation Macro-F1, decreasing from 0.6605 to 0.6331.

These test results should therefore be identified specifically as the results of the **128-hidden-unit evaluation run**. They should not be presented as the test performance of the subsequently retrained 256-hidden-unit configuration.

### Test Per-Class Results

| Label         | Precision | Recall |     F1 | ROC-AUC | PR-AUC |
| ------------- | --------: | -----: | -----: | ------: | -----: |
| toxic         |    0.7945 | 0.7720 | 0.7831 |  0.9707 | 0.8676 |
| severe_toxic  |    0.4536 | 0.5314 | 0.4894 |  0.9901 | 0.4542 |
| obscene       |    0.8189 | 0.8131 | 0.8160 |  0.9876 | 0.8908 |
| threat        |    0.4857 | 0.4722 | 0.4789 |  0.9793 | 0.4681 |
| insult        |    0.6793 | 0.7604 | 0.7175 |  0.9800 | 0.7674 |
| identity_hate |    0.4654 | 0.5735 | 0.5138 |  0.9763 | 0.4802 |

The per-class results show higher F1 values for `toxic`, `obscene`, and `insult`, while the minority categories had lower F1 values. The relatively high ROC-AUC values indicate that the model generally separated positive and negative examples well in terms of probability ranking, while the lower PR-AUC values for some minority classes reflect the difficulty introduced by class imbalance.

## Efficiency

The original 128-hidden-unit BiLSTM contained **7,728,662 trainable parameters**.

The recorded epoch times for the original final training run totalled approximately **523.39 seconds (8.72 minutes)** on a Google Colab Tesla T4 GPU.

The final test inference time was **8.43 seconds** for 23,936 test samples, corresponding to approximately **0.352 seconds per 1,000 samples**.

| Efficiency Measure             |          Value |
| ------------------------------ | -------------: |
| Trainable parameters           |      7,728,662 |
| Training time                  | 523.39 seconds |
| Inference time / 1,000 samples |  0.352 seconds |
| GPU                            |       Tesla T4 |

The subsequently retrained 256-hidden-unit configuration contains **8,227,862 parameters**. Its training history and checkpoint were stored separately from the original final evaluation.

## Figures

The following figures were generated for the BiLSTM model:

- `bilstm_loss_curve.png` — training and validation loss across epochs
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

The BiLSTM learned useful sequential representations for toxic comment classification. In the original final evaluation, the 128-hidden-unit configuration achieved a validation Macro-F1 of 0.6605 and a test Macro-F1 of 0.6331 after validation-based per-class threshold tuning.

The training history showed increasing validation loss and a decline in validation Macro-F1 after Epoch 5, providing evidence of overfitting in the original training run.

The hyperparameter search identified a 256-hidden-unit, 0.3-dropout configuration with a validation Macro-F1 of 0.5861 during the search. When this configuration was retrained, its best validation Macro-F1 was 0.5701. A separate class-weight ablation using the selected architecture produced a validation Macro-F1 of 0.6319 without class weights, compared with 0.5701 with class weights. These experimental scores used a fixed 0.5 threshold.

For the final evaluation methodology, class-specific thresholds were tuned only on the validation set and then frozen before evaluating the test set. This prevents test-set information from being used for threshold selection.

The lower test Macro-F1 compared with the validation Macro-F1 indicates that the validation result did not completely represent performance on unseen test data. The final test figures reported above therefore remain associated specifically with the original 128-hidden-unit evaluation run and should not be interpreted as test results for the subsequently retrained 256-hidden-unit model.
