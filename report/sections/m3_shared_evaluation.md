# 5. Experimental Design (Shared Pipeline)

To ensure a completely fair and leakage-free comparison between all four models, we enforced a strict shared data pipeline across the entire team.

## Dataset Split and Leakage Prevention
We established ONE canonical 70/15/15 stratified split (seed 42) for train, validation, and test sets. All models used this exact same split. To prevent data leakage, the tokenizer vocabulary, GloVe embedding matrix mapping, and the positive class weights (pos_weight for BCE loss) were computed **exclusively** on the 70% training split. The validation and test sets were never exposed to these fitting steps.

## Evaluation Metrics
Given the extreme class imbalance in the Jigsaw dataset (where over 90% of comments are non-toxic), standard accuracy is highly misleading. We selected **macro-F1** as our primary metric because it treats all 6 classes equally, forcing the models to perform well on the rare classes (like `threat`) to achieve a high score. We also tracked micro-F1, per-class Precision and Recall, and threshold-independent metrics like ROC-AUC and PR-AUC. 

## Threshold Tuning
For multi-label classification, applying a flat 0.5 threshold to the sigmoid probabilities often results in poor F1 scores due to class imbalance. Instead, we tuned the decision threshold independently for each of the 6 classes. We calculated the precision-recall curve on the **validation set** and selected the exact probability threshold that maximized the F1 score for that specific class. These validated thresholds were then frozen and applied to the test set exactly once.
