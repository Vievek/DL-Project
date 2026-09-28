# 7. Results and comparison: BiLSTM + Attention (M3)

## Overall Performance
The BiLSTM with Attention achieved a **macro-F1 score of 0.6481** and a **micro-F1 score of 0.7435** on the final test set. The model performed exceptionally well on the `obscene` (F1 = 0.80) and `toxic` (F1 = 0.78) classes, while performance naturally dipped on the severely imbalanced classes like `threat` (F1 = 0.55) and `identity_hate` (F1 = 0.51). 

## Learning Curves and Overfitting
Looking at the learning curves, the model steadily improved until epoch 7, where it achieved its best validation macro-F1 of 0.6596. After epoch 7, the validation loss began to increase sharply while the training loss continued dropping toward zero—a clear sign of overfitting to the training data. Thanks to early stopping with a patience of 3, the training correctly halted at epoch 10, and the weights from epoch 7 were restored for final testing.

## Classification Metrics
The model maintained high discrimination power overall, shown by excellent ROC-AUC scores (all classes above 0.97). However, the Precision-Recall (PR) curves highlight the challenge of the rare classes. For example, `severe_toxic` and `threat` had PR-AUCs of 0.43 and 0.46, respectively. This shows that while the model can rank positive examples higher than negatives (good ROC), maintaining high precision without sacrificing recall remains difficult when positives are very rare.

## Inference Efficiency
In terms of efficiency, the BiLSTM-Attention model requires about 237k trainable parameters (excluding the static GloVe embeddings) and takes roughly 25 seconds per training epoch on a Tesla T4 GPU. Inference is extremely fast, taking just ~0.05 seconds per 1000 samples, making it highly practical for real-time moderation environments compared to heavier models like DistilBERT.
