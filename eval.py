"""
Training and validation loops for the Show, Attend and Tell image captioning model.

Provides epoch-level training with teacher forcing and validation with no gradient
computation. Both functions return the average loss over the dataset.

The training loop includes doubly-stochastic attention regularization from the
original SAT paper: an extra penalty that encourages the attention weights for
each spatial region to sum to approximately 1 over the full caption.
"""

import torch
from tqdm import tqdm


def train_epoch(model, dataloader, criterion, optimizer, vocab_size, epoch, device,
                alpha_c=1.0):
    """
    Run one training epoch with teacher forcing and attention regularization.

    Args:
        model: The AttentionDecoder model to train.
        dataloader: DataLoader yielding (features, captions, lengths) batches.
        criterion: Loss function (e.g., CrossEntropyLoss).
        optimizer: Optimizer instance (e.g., Adam).
        vocab_size: Size of the vocabulary (used to reshape output logits).
        epoch: Current epoch index (0-based), used for progress bar display.
        device: torch.device to move tensors to (e.g., 'cuda' or 'cpu').
        alpha_c: Weight for the doubly-stochastic attention regularization.
            Set to 0 to disable. Default: 1.0.

    Returns:
        float: Average training loss over all batches in this epoch.
    """
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}", unit="batch")

    for features, captions, lengths in progress_bar:
        features = features.to(device)
        captions = captions.to(device)

        optimizer.zero_grad()
        predictions, alphas = model(features, captions, lengths)

        # predictions: (B, max_decode_len, vocab_size)
        # targets: captions shifted left by one (skip SOS, predict next token)
        max_decode_len = predictions.size(1)
        targets = captions[:, 1:max_decode_len + 1]

        loss = criterion(
            predictions.reshape(-1, vocab_size),
            targets.reshape(-1),
        )

        # Doubly-stochastic attention regularization (Eq. 14 in the SAT paper):
        # encourages sum of attention weights over all time steps ≈ 1 for each pixel
        if alpha_c > 0:
            loss += alpha_c * ((1.0 - alphas.sum(dim=1)) ** 2).mean()

        loss.backward()
        # Gradient clipping to stabilise training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def validate(model, dataloader, criterion, vocab_size, device):
    """
    Run one validation pass (no gradient computation).

    Args:
        model: The AttentionDecoder model to evaluate.
        dataloader: DataLoader yielding (features, captions, lengths) batches.
        criterion: Loss function (e.g., CrossEntropyLoss).
        vocab_size: Size of the vocabulary (used to reshape output logits).
        device: torch.device to move tensors to.

    Returns:
        float: Average validation loss over all batches.
    """
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for features, captions, lengths in dataloader:
            features = features.to(device)
            captions = captions.to(device)

            predictions, alphas = model(features, captions, lengths)

            max_decode_len = predictions.size(1)
            targets = captions[:, 1:max_decode_len + 1]

            loss = criterion(
                predictions.reshape(-1, vocab_size),
                targets.reshape(-1),
            )
            total_loss += loss.item()

    avg_test_loss = total_loss / len(dataloader)
    return avg_test_loss
