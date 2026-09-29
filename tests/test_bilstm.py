import torch
import torch.nn as nn

from src.models.bilstm import BiLSTMClassifier


def test_bilstm_forward_and_backward():
    # Batch of 4 sequences, each padded/truncated to length 20
    input_ids = torch.randint(1, 100, (4, 20))

    # Random valid sequence lengths
    lengths = torch.tensor([20, 15, 10, 5])

    model = BiLSTMClassifier(
        vocab_size=100,
        num_labels=6,
        embed_dim=100,
        hidden_dim=64,
    )

    # Forward pass
    logits = model(input_ids, lengths)

    # Expected output: one logit for each of 6 labels
    assert logits.shape == (4, 6)

    # Check that loss + backward pass works
    targets = torch.randint(0, 2, (4, 6)).float()

    loss_fn = nn.BCEWithLogitsLoss()
    loss = loss_fn(logits, targets)

    loss.backward()