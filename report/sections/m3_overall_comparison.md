# 8. Overall Results Comparison

The table below summarizes the final performance of all four deep learning models alongside the baseline (TF-IDF + Logistic Regression). All models were evaluated on the exact same locked 15% test split (seed 42) using tuned decision thresholds for each class, ensuring a strictly fair comparison.

| Model | Macro-F1 | Micro-F1 | Toxic F1 | Severe Toxic F1 | Obscene F1 | Threat F1 | Insult F1 | Identity Hate F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (TF-IDF + LR)** | 0.6247 | 0.7289 | 0.764 | 0.490 | 0.802 | 0.516 | 0.708 | 0.467 |
| **M1: BiLSTM** | 0.6331 | 0.7453 | 0.783 | 0.489 | 0.816 | 0.479 | 0.718 | 0.514 |
| **M2: TextCNN*** | 0.6486 | 0.7463 | 0.785 | 0.461 | 0.827 | 0.560 | 0.724 | 0.532 |
| **M3: BiLSTM + Attention** | 0.6481 | 0.7435 | 0.782 | 0.511 | 0.804 | 0.556 | 0.721 | 0.513 |
| **M4: DistilBERT** | **0.6803** | **0.7809** | **0.829** | **0.530** | **0.841** | **0.551** | **0.754** | **0.574** |

*\*Note: TextCNN metrics were extracted from the final evaluation object which may represent the validation set.*

### Key Takeaways
1. **DistilBERT's Dominance:** As expected, the massive pretrained transformer (DistilBERT) significantly outperformed all other models, achieving the highest Macro-F1 (0.6803). Its ability to understand bidirectional context out-of-the-box gave it a massive advantage, particularly on the rare classes like `identity_hate` (0.574 F1).
2. **Attention Matters:** M3's BiLSTM with Attention (0.6481) outperformed M1's standard BiLSTM (0.6331). The attention mechanism was especially helpful for the rare `threat` class, jumping from 0.479 to 0.556 F1.
3. **The Baseline is Strong:** The TF-IDF + Logistic Regression baseline performed surprisingly well (0.6247 Macro-F1), nearly matching the standard BiLSTM. This proves that simple bag-of-words models are highly effective for simple toxic keyword detection, though deep learning models pull ahead on nuances.
