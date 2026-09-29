# M1 — BiLSTM Architecture

## Model Overview

The M1 model is a Bidirectional Long Short-Term Memory (BiLSTM) neural network designed for multi-label toxic comment classification. The model receives tokenized comments and predicts six toxicity labels: toxic, severe_toxic, obscene, threat, insult, and identity_hate.

The model uses the shared vocabulary and preprocessing pipeline defined for the project. The maximum input sequence length is 128 tokens, and the embedding dimension is 100.

## Architecture

The BiLSTM model consists of the following components:

1. **Embedding Layer**
   - Vocabulary size: 74,916
   - Embedding dimension: 100
   - Padding index: 0
   - GloVe 6B 100-dimensional embeddings were used for initialization.

2. **Embedding Dropout**
   - Dropout rate: 0.1

3. **Bidirectional LSTM**
   - Input dimension: 100
   - Hidden dimension: 128
   - Number of layers: 1
   - Bidirectional: Yes
   - The forward and backward hidden states are concatenated, producing a 256-dimensional representation.

4. **Dropout Layer**
   - Dropout rate: 0.3

5. **Classification Layer**
   - Input dimension: 256
   - Output dimension: 6
   - Produces one logit for each toxicity label.

The model uses packed sequences based on the actual sequence lengths. This prevents padding tokens from being processed unnecessarily by the LSTM.

## Mathematical Representation

For an input sequence of tokens, the embedding layer converts each token into a 100-dimensional vector:

\[
E = Embedding(X)
\]

The embedded sequence is then processed in both forward and backward directions by the BiLSTM:

\[
H = BiLSTM(E)
\]

The final forward and backward hidden states are concatenated:

\[
h = [h_{forward}; h_{backward}]
\]

This produces a 256-dimensional representation because the hidden dimension is 128 for each direction.

Dropout is then applied before the final classification layer:

\[
z = W \cdot Dropout(h) + b
\]

where \(z\) contains six output logits.

The logits are converted to independent probabilities using the sigmoid function:

\[
p_i = \frac{1}{1 + e^{-z_i}}
\]

Since a comment can belong to multiple toxicity categories, the model uses independent probabilities rather than a softmax distribution.

## Training Configuration

The shared training configuration was used:

| Configuration | Value |
|---|---|
| Maximum sequence length | 128 |
| Embedding dimension | 100 |
| Hidden dimension | 128 |
| LSTM layers | 1 |
| Bidirectional | Yes |
| Embedding dropout | 0.1 |
| Model dropout | 0.3 |
| Batch size | 32 |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Loss function | BCEWithLogitsLoss |
| Class weighting | Yes |
| Maximum epochs | 10 |
| Early stopping patience | 3 |
| Random seed | 42 |
| Hardware | Google Colab Tesla T4 |

Class-specific positive weights were calculated using only the training set to address the strong class imbalance.

## Number of Parameters

The final BiLSTM model contains **7,728,662 trainable parameters**.

## Threshold Selection

The model outputs probabilities for each of the six labels. Because the dataset is multi-label and highly imbalanced, a separate classification threshold was selected for each class.

The thresholds were optimized using the validation set's precision-recall curves. The validation set was used only for threshold selection and model selection. The selected thresholds were then frozen before evaluating the test set.

The final thresholds were:

| Label | Threshold |
|---|---:|
| toxic | 0.866426 |
| severe_toxic | 0.991600 |
| obscene | 0.963665 |
| threat | 0.998464 |
| insult | 0.933720 |
| identity_hate | 0.987822 |
