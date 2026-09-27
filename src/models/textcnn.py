"""
M2 — TextCNN
Owner: Tharsiga Ranganathan (M2) - Preprocessing & TextCNN

Kim (2014), "Convolutional Neural Networks for Sentence Classification" — cite in References.

Embedding (GloVe 6B 100d, fine-tuned) -> parallel Conv1d filters of widths (3, 4, 5), each width
capturing a different n-gram-like pattern -> ReLU -> max-over-time pooling (padding positions
masked out) -> concatenate -> dropout -> Linear(num_labels).
Outputs raw logits (multi-label, no softmax) -> train with BCEWithLogitsLoss(pos_weight=...).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNNClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        embed_dim: int = 100,
        num_filters: int = 100,
        filter_sizes=(3, 4, 5),
        dropout: float = 0.5,
        pad_idx: int = 0,
        pretrained_embeddings: torch.Tensor = None,
        freeze_embeddings: bool = False,
    ):
        super().__init__()
        if pretrained_embeddings is not None:
            # .clone() so training never changes the shared GloVe matrix used by other runs
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings.clone(), freeze=freeze_embeddings, padding_idx=pad_idx
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        embed_dim = self.embedding.embedding_dim

        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, num_filters, kernel_size=fs) for fs in filter_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_filters * len(filter_sizes), num_labels)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor = None,
                mask: torch.Tensor = None) -> torch.Tensor:
        # (batch, seq_len) -> (batch, seq_len, embed_dim) -> (batch, embed_dim, seq_len)
        # Conv1d slides over the LAST dimension, so the permute is required.
        x = self.embedding(input_ids).permute(0, 2, 1)

        pooled = []
        for conv in self.convs:
            c = F.relu(conv(x))  # (batch, num_filters, seq_len - k + 1)
            if mask is not None:
                # A window is kept only if it starts on a real token, so padding never wins the max.
                valid = mask[:, : c.size(2)].unsqueeze(1)
                c = c.masked_fill(~valid, float("-inf"))
            pooled.append(c.max(dim=2).values)  # max-over-time -> (batch, num_filters)

        features = torch.cat(pooled, dim=1)  # (batch, num_filters * len(filter_sizes))
        return self.classifier(self.dropout(features))  # raw logits, (batch, num_labels)


def count_parameters(model: nn.Module, include_embedding: bool = True) -> int:
    """Trainable parameters, optionally excluding the embedding table."""
    return sum(
        p.numel() for name, p in model.named_parameters()
        if p.requires_grad and (include_embedding or not name.startswith("embedding."))
    )