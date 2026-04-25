"""
Main training script for the Show, Attend and Tell image captioning model.

Orchestrates the full pipeline:
    1. Builds vocabulary from Flickr8k captions.
    2. Extracts VGG16 features and stores them in an HDF5 file (if not already done).
    3. Creates train/test data loaders.
    4. Trains the AttentionDecoder, saving checkpoints for the best model.

Usage:
    python main.py
"""

import os
import torch
from torch import nn, optim
from torchvision import transforms

from models import MyVGG16, AttentionDecoder
from get_data import build_vocab, get_loaders, Flickr8kDataset
from eval import train_epoch, validate
from create_h5 import save_to_h5

# ── Hyperparameters ──────────────────────────────────────────────────────────
EMBED_DIM = 256
HIDDEN_DIM = 512
ATTENTION_DIM = 256
ENCODER_DIM = 512       # VGG16 feature channels
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
EPOCHS = 10
MIN_WORD_FREQ = 2
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4

# ── Paths ────────────────────────────────────────────────────────────────────
IMAGES_DIR = "data/Images"
TOKENS_FILE = "data/captions.txt"
HDF5_FILE = "data/features.h5"
BEST_CHECKPOINT_PATH = "best_checkpoint.pth"
FINAL_MODEL_PATH = "final_model.pth"
VOCAB_PATH = "vocab.pkl"

# ── Image transforms (ImageNet normalization) ───────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def main():
    """Run the full training pipeline."""
    torch.manual_seed(SEED)

    # Step 1: Build vocabulary and split data
    train_dict, test_dict, vocab = build_vocab(TOKENS_FILE, MIN_WORD_FREQ, VOCAB_PATH)
    vocab_size = len(vocab)

    # Step 2: Extract features with VGG16 encoder (skip if already cached)
    if not os.path.isfile(HDF5_FILE):
        print("Extracting image features with VGG16...")
        encoder = MyVGG16()
        encoder = encoder.to(DEVICE)
        save_to_h5(encoder, IMAGES_DIR, DEVICE, HDF5_FILE)
        print(f"Features saved to {HDF5_FILE}")

    # Step 3: Create data loaders
    train_loader, test_loader = get_loaders(
        train_dict,
        test_dict,
        vocab,
        h5_path=HDF5_FILE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        collate_fn=Flickr8kDataset.collate_fn,
    )

    # Step 4: Initialize attention decoder and training components
    decoder = AttentionDecoder(
        embedding_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        vocab_size=vocab_size,
        encoder_dim=ENCODER_DIM,
        attention_dim=ATTENTION_DIM,
        dropout=0.5,
    ).to(DEVICE)

    optimizer = optim.Adam(decoder.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore PAD token

    # Step 5: Training loop
    best_val_loss = float("inf")
    for epoch in range(EPOCHS):
        train_loss = train_epoch(
            model=decoder,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            vocab_size=vocab_size,
            epoch=epoch,
            device=DEVICE,
        )

        val_loss = validate(
            model=decoder,
            dataloader=test_loader,
            criterion=criterion,
            vocab_size=vocab_size,
            device=DEVICE,
        )

        print(f"Epoch {epoch+1}/{EPOCHS} — Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": decoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "vocab_size": vocab_size,
            }, BEST_CHECKPOINT_PATH)
            print(f"  -> New best model saved (val_loss={val_loss:.4f})")

    # Save final model
    torch.save(decoder.state_dict(), FINAL_MODEL_PATH)
    print(f"Training complete. Final model saved to {FINAL_MODEL_PATH}")


if __name__ == "__main__":
    main()
