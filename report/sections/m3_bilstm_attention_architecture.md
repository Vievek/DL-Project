# 6. Model architectures: BiLSTM + Attention (M3)

## Motivation
While a standard BiLSTM compresses the entire sequence into a single hidden state, adding an attention mechanism allows the model to dynamically focus on specific tokens that drive toxicity (e.g., slurs or threats) regardless of their position in the text. This not only improves representation but also provides interpretability, allowing us to visualize exactly which words the model attended to for its predictions.

## Architecture
The text is tokenized and padded to a maximum length of 128. The architecture consists of the following layers:
- **Embedding Layer**: A 100-dimensional embedding initialized with pretrained GloVe 6B 100d vectors. Out-of-vocabulary words are randomly initialized.
- **BiLSTM Layer**: A single-layer bidirectional LSTM with a hidden dimension of 128. Because it is bidirectional, the output at each timestep is a 256-dimensional concatenated hidden state.
- **Additive Attention Layer**: A linear layer projects the 256-dimensional hidden state to a single score. These scores are masked to ignore padding tokens and normalized via softmax to produce attention weights. The final sequence representation is the weighted sum of the LSTM hidden states.
- **Dropout**: A dropout rate of 0.3 is applied to prevent overfitting.
- **Classifier**: A linear layer maps the 256-dimensional attended context vector to 6 raw logits (one per label).

## Parameter Count
- **Total parameters**: 7,728,919
- **Trainable parameters (excluding embeddings)**: 237,319

## Loss and Optimizer
The model was trained using `BCEWithLogitsLoss` to handle the multi-label nature of the task. We applied positive class weighting (pos_weight) computed directly from the training split to heavily penalize misses on rare classes like `threat` and `identity_hate`. The optimizer used was Adam with a learning rate of 0.001, and early stopping was applied with a patience of 3 epochs based on validation macro-F1.
