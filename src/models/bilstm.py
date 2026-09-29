"""
M1 — BiLSTM (owner: Teammate A)

A bidirectional LSTM over word embeddings, pooled and passed through a linear head
producing one logit per label (multi-label classification).
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
        embedding_dropout: float = 0.1,
        pad_idx: int = 0,
        return_attention: bool = False,
    ):
        super().__init__()

        self.return_attention = return_attention

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=pad_idx,
        )

        # Dropout applied after the embedding layer
        self.embedding_dropout = nn.Dropout(embedding_dropout)

        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)

        # *2 because the LSTM is bidirectional
        self.classifier = nn.Linear(
            hidden_dim * 2,
            num_labels,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        embedded = self.embedding(input_ids)

        # Apply embedding dropout
        embedded = self.embedding_dropout(embedded)

        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        _, (hidden, _) = self.lstm(packed)

        # Last layer's forward and backward hidden states
        final = torch.cat(
            (hidden[-2], hidden[-1]),
            dim=1,
        )

        logits = self.classifier(self.dropout(final))

        # Attention is handled by the separate BiLSTM-Attention model.
        return logits