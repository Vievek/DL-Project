"""
M1 — BiLSTM (owner: Teammate A)

A bidirectional LSTM over word embeddings, pooled and passed through a linear head producing one
logit per label (multi-label, so no softmax — use BCEWithLogitsLoss with config.yaml's
class-weighting on top).
"""

import torch
import torch.nn as nn


class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.3,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        # *2 because bidirectional concatenates forward + backward final hidden states
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        # concat last layer's forward and backward hidden states
        final = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.classifier(self.dropout(final))  # raw logits, shape (batch, num_labels)
