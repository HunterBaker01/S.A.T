"""
Feature extraction pipeline for precomputing VGG16 image features.

Processes all images in a directory through the VGG16 convolutional layers,
reshapes the spatial feature maps into (num_regions, 512) vectors, and stores
them in an HDF5 file keyed by image ID (filename without extension).

The HDF5 file is later consumed by Flickr8kDataset during training, avoiding
redundant forward passes through the encoder at every epoch.
"""

import os

import h5py
import torch
from torchvision import transforms
from PIL import Image

# ImageNet normalization applied before feeding images to VGG16
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def extract_features(image_path, encoder, device):
    """
    Load an image and extract VGG16 convolutional feature maps.

    Args:
        image_path: Path to the image file.
        encoder: MyVGG16 encoder model.
        device: torch.device to run inference on.

    Returns:
        Feature map tensor of shape (1, 512, H, W).
    """
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = encoder(image)

    return features


def reshape_features(features):
    """
    Reshape VGG16 feature maps into a 2D spatial feature matrix.

    Converts (1, C, H, W) -> (H*W, 512), where each row is a spatial
    region's feature vector. For 224x224 input, VGG16 produces 7x7=49 regions.

    Args:
        features: Tensor of shape (1, 512, H, W).

    Returns:
        Tensor of shape (H*W, 512).
    """
    features = features.squeeze(0)       # (512, H, W)
    features = features.permute(1, 2, 0)  # (H, W, 512)
    features = features.view(-1, 512)     # (H*W, 512)
    return features


def save_to_h5(encoder, image_dir, device, save_path="features.h5"):
    """
    Extract and save VGG16 features for all JPEG images in a directory.

    Iterates over all .jpg files, extracts and reshapes features, and
    stores them in an HDF5 file with the image ID (filename without
    extension) as the dataset key.

    Args:
        encoder: MyVGG16 encoder model (will be set to eval mode).
        image_dir: Path to directory containing .jpg image files.
        device: torch.device for running the encoder.
        save_path: Output path for the HDF5 file.
    """
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    image_id_to_path = {}
    for fname in os.listdir(image_dir):
        if fname.lower().endswith(".jpg"):
            image_id = os.path.splitext(fname)[0]
            image_path = os.path.join(image_dir, fname)
            image_id_to_path[image_id] = image_path

    print(f"Extracting features for {len(image_id_to_path)} images...")

    with h5py.File(save_path, "w") as h5f:
        for i, (image_id, image_path) in enumerate(image_id_to_path.items()):
            raw_features = extract_features(image_path, encoder, device)
            features = reshape_features(raw_features)

            h5f.create_dataset(
                name=image_id,
                data=features.cpu().numpy(),
                dtype="float32",
            )

            if (i + 1) % 500 == 0:
                print(f"  Processed {i+1}/{len(image_id_to_path)} images")

    print(f"All features saved to {save_path}")
