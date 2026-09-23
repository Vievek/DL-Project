"""
M2 — TextCNN 
Owner: Tharsiga Ranganathan (M2) - Preprocessing & TextCNN

Convolutional filters of several widths slide over the word-embedding sequence, each width
capturing a different n-gram-like pattern; max-pool each, concatenate, classify. Kim (2014),
"Convolutional Neural Networks for Sentence Classification" — cite in References.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNNClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        embed_dim: int = 128,
        num_filters: int = 100,
        filter_sizes=(3, 4, 5),
        dropout: float = 0.3,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, num_filters, kernel_size=fs) for fs in filter_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_filters * len(filter_sizes), num_labels)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids).permute(0, 2, 1)  # (batch, embed_dim, seq_len)
        conv_outs = [F.relu(conv(embedded)) for conv in self.convs]
        pooled = [F.max_pool1d(c, c.shape[2]).squeeze(2) for c in conv_outs]
        concatenated = torch.cat(pooled, dim=1)
        return self.classifier(self.dropout(concatenated))  # raw logits, shape (batch, num_labels)
