"""
M3 — BiLSTM + Attention (owner: Teammate C)

A BiLSTM encodes the sequence, then an additive attention layer learns to weight each timestep's
hidden state before pooling — lets the model focus on the specific words driving toxicity, and
gives you attention-weight visualisations for the Critical Analysis section (which words did the
model attend to for a false positive/negative?).

Alternative: swap this for a small Transformer encoder built from scratch (multi-head
self-attention + position embeddings, no pretraining) if the team prefers a starker contrast
against M4's pretrained DistilBERT — note this choice and its rationale in the report either way.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # lstm_out: (batch, seq_len, hidden_dim), mask: (batch, seq_len) 1=real token, 0=pad
        scores = self.attn(lstm_out).squeeze(-1)              # (batch, seq_len)
        scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = F.softmax(scores, dim=1)                     # (batch, seq_len)
        context = torch.bmm(weights.unsqueeze(1), lstm_out).squeeze(1)  # (batch, hidden_dim)
        return context, weights  # keep weights around for error-analysis visualisations


class BiLSTMAttentionClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embedded)              # (batch, seq_len, hidden_dim*2)
        context, attn_weights = self.attention(lstm_out, mask)
        logits = self.classifier(self.dropout(context))
        return logits, attn_weights  # keep both — weights are useful for the report's error analysis
