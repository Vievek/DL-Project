"""
M4 — DistilBERT, fine-tuned (owner: DEEPDEV)

Pretrained Transformer encoder (Hugging Face `distilbert-base-uncased`) with a linear multi-label
head on top of the [CLS] token, fine-tuned end-to-end. Uses its OWN pretrained tokenizer — do not
run this through the shared vocab-building path in data_utils.build_vocab (that's for the
non-pretrained models only).
"""

import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizerFast

MODEL_NAME = "distilbert-base-uncased"


def get_tokenizer():
    return DistilBertTokenizerFast.from_pretrained(MODEL_NAME)


class DistilBertClassifier(nn.Module):
    def __init__(self, num_labels: int, dropout: float = 0.1, freeze_base: bool = False):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(MODEL_NAME)
        if freeze_base:
            for p in self.bert.parameters():
                p.requires_grad = False
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]  # [CLS] representation
        return self.classifier(self.dropout(cls_token))  # raw logits, shape (batch, num_labels)


# TODO (DEEPDEV): tokenize with max_length=128 (config.yaml) using get_tokenizer(), fp16 mixed
# precision on Colab if training is slow, 1-3 epochs is usually enough for fine-tuning — log
# params/train-time/inference-time on the SAME GPU type as the other 3 models for a fair
# efficiency comparison (Section 6 of the plan doc).
