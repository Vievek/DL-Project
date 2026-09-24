import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, vocab_size=1000, embed_dim=16, num_classes=6):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.linear = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        # x is [batch_size, seq_len]
        x = self.embed(x)
        # pool over sequence
        x = x.mean(dim=1)
        return self.linear(x)
