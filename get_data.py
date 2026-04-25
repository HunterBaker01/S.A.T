"""
Data loading and vocabulary management for the image captioning model.

Handles:
    - Parsing Flickr8k caption files into image-to-caption dictionaries.
    - Building a word-level vocabulary with frequency thresholding.
    - Loading precomputed VGG16 features from HDF5 and pairing them with
      tokenized captions in a PyTorch Dataset.
    - Creating train/test DataLoaders with variable-length caption padding.
"""

import os
import re
import csv
import random
import pickle
from collections import Counter

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image


class ImageCaptionDataset(Dataset):
    """
    Basic image-caption dataset that loads raw images from disk.

    This is a simpler dataset class intended for prototyping or when
    precomputed features are not available. Each sample returns a
    (transformed) image tensor and a randomly chosen caption string.

    Note:
        The main training pipeline uses Flickr8kDataset (with precomputed
        HDF5 features) instead of this class.

    Args:
        image_dir: Path to directory containing image files.
        captions_file: Path to CSV file with columns [image_id, caption].
        transform: Optional torchvision transform to apply to images.
    """

    def __init__(self, image_dir, captions_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.captions = {}
        with open(captions_file, "r") as f:
            for line in f:
                parts = line.strip().split(",", 1)
                if len(parts) < 2:
                    continue
                image_id = parts[0]
                caption = parts[1]
                if image_id not in self.captions:
                    self.captions[image_id] = []
                self.captions[image_id].append(caption)
        self.image_ids = list(self.captions.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.image_dir, image_id)
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        caption = random.choice(self.captions[image_id])
        return image, caption


class Vocabulary:
    """
    Word-level vocabulary with frequency-based filtering.

    Maintains bidirectional mappings between words and integer indices.
    Includes four special tokens: PAD (0), SOS (1), EOS (2), UNK (3).

    Args:
        freq_threshold: Minimum number of occurrences for a word to be
            included in the vocabulary. Words below this threshold are
            mapped to UNK at encoding time.
    """

    def __init__(self, freq_threshold=5):
        self.freq_threshold = freq_threshold
        self.toks = {0: "pad", 1: "sos", 2: "eos", 3: "unk"}
        self.ind = {v: k for k, v in self.toks.items()}
        self.index = 4  # Next available index for new words

    def __len__(self):
        return len(self.toks)

    def tokenizer(self, text):
        """Split text into lowercase word tokens using regex."""
        text = text.lower()
        tokens = re.findall(r"\w+", text)
        return tokens

    def build_vocab(self, sentence_list):
        """
        Build vocabulary from a list of sentences.

        Counts word frequencies across all sentences, then adds words
        that meet the frequency threshold to the vocabulary.

        Args:
            sentence_list: List of caption strings to build vocab from.
        """
        frequencies = Counter()
        for sentence in sentence_list:
            tokens = self.tokenizer(sentence)
            frequencies.update(tokens)

        for word, freq in frequencies.items():
            if freq >= self.freq_threshold:
                self.ind[word] = self.index
                self.toks[self.index] = word
                self.index += 1

    def numerical(self, text):
        """
        Convert a text string to a list of token indices.

        Words not in the vocabulary are mapped to UNK.

        Args:
            text: Input string to encode.

        Returns:
            List of integer indices corresponding to each word.
        """
        tokens = self.tokenizer(text)
        numerical = []
        for token in tokens:
            if token in self.ind:
                numerical.append(self.ind[token])
            else:
                numerical.append(self.ind["unk"])
        return numerical


def parse_tokens(file):
    """
    Parse a Flickr8k-format caption file into a dictionary.

    Reads a CSV file with columns [image_filename, caption] and groups
    captions by image ID (filename without the .jpg extension, to match
    HDF5 feature keys).

    Args:
        file: Path to the captions CSV file.

    Returns:
        Dict mapping image_id (str, no extension) to a list of caption strings.
    """
    imgid2captions = {}
    with open(file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # Skip header row
        for row in reader:
            if len(row) < 2:
                continue
            raw_id, caption = row[0], row[1]
            # Strip .jpg extension to match HDF5 feature keys
            img_id = os.path.splitext(raw_id)[0]
            if img_id not in imgid2captions:
                imgid2captions[img_id] = []
            imgid2captions[img_id].append(caption)
    return imgid2captions


class Flickr8kDataset(Dataset):
    """
    Dataset that pairs precomputed HDF5 image features with tokenized captions.

    Each sample returns:
        - features: Tensor of shape (num_regions, 512) from VGG16.
        - caption: Tensor of token indices wrapped with SOS and EOS.

    The HDF5 file is opened lazily on first access to support
    multi-worker DataLoader (each worker gets its own file handle).

    Args:
        imgid2captions: Dict mapping image_id to list of caption strings.
        vocab: Vocabulary instance for tokenizing captions.
        h5_path: Path to HDF5 file containing precomputed features.
    """

    def __init__(self, imgid2captions, vocab, h5_path):
        self.samples = []
        self.vocab = vocab
        self.h5_path = h5_path
        self.h5f = None

        for img_id, caps in imgid2captions.items():
            for c in caps:
                self.samples.append((img_id, c))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Lazy-open HDF5 file (required for multi-worker DataLoader)
        if self.h5f is None:
            self.h5f = h5py.File(self.h5_path, "r")

        img_id, caption = self.samples[idx]

        # Return full spatial features (num_pixels, 512) for attention
        features = torch.tensor(
            self.h5f[img_id][:],
            dtype=torch.float32,
        )

        numerical_caption = [self.vocab.ind["sos"]]
        numerical_caption += self.vocab.numerical(caption)
        numerical_caption.append(self.vocab.ind["eos"])

        return features, torch.tensor(numerical_caption, dtype=torch.long)

    @staticmethod
    def collate_fn(batch):
        """
        Collate variable-length captions by padding to the longest in the batch.

        Sorts the batch by caption length (descending) for efficient packing,
        pads shorter captions with zeros (PAD token), and stacks features.

        Args:
            batch: List of (features, caption_tensor) tuples.

        Returns:
            Tuple of (features, padded_captions, lengths):
                - features: Tensor of shape (batch, feature_dim).
                - padded_captions: Tensor of shape (batch, max_len).
                - lengths: List of original caption lengths.
        """
        batch.sort(key=lambda x: len(x[1]), reverse=True)
        features = [item[0] for item in batch]
        captions = [item[1] for item in batch]
        lengths = [len(cap) for cap in captions]
        max_len = max(lengths)

        padded_captions = torch.zeros(len(captions), max_len, dtype=torch.long)
        for i, cap in enumerate(captions):
            end = lengths[i]
            padded_captions[i, :end] = cap[:end]

        features = torch.stack(features, dim=0)
        return features, padded_captions, lengths


def build_vocab(tokens_file, min_word_freq, vocab_path):
    """
    Build vocabulary and split data into train/test sets.

    Parses the caption file, builds a frequency-filtered vocabulary,
    saves it to disk via pickle, and performs an 80/20 train/test split
    at the image level (all captions for an image go to the same split).

    Args:
        tokens_file: Path to the captions CSV file.
        min_word_freq: Minimum word frequency for vocabulary inclusion.
        vocab_path: Path to save the pickled Vocabulary object.

    Returns:
        Tuple of (train_dict, test_dict, vocab):
            - train_dict: Dict mapping image_id to captions (80% of images).
            - test_dict: Dict mapping image_id to captions (20% of images).
            - vocab: The built Vocabulary instance.
    """
    imgid2captions = parse_tokens(tokens_file)
    all_captions = []
    for caps in imgid2captions.values():
        all_captions.extend(caps)

    vocab = Vocabulary(freq_threshold=min_word_freq)
    vocab.build_vocab(all_captions)

    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)

    print(f"Vocabulary saved to: {vocab_path}")

    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")

    img_ids = list(imgid2captions.keys())
    random.shuffle(img_ids)
    split_idx = int(0.8 * len(img_ids))
    train_ids = img_ids[:split_idx]
    test_ids = img_ids[split_idx:]

    train_dict = {iid: imgid2captions[iid] for iid in train_ids}
    test_dict = {iid: imgid2captions[iid] for iid in test_ids}
    return train_dict, test_dict, vocab


def get_loaders(train_dict, test_dict, vocab, h5_path="features.h5",
                batch_size=16, num_workers=4, collate_fn=None):
    """
    Create train and test DataLoaders.

    Args:
        train_dict: Dict mapping image_id to captions for training.
        test_dict: Dict mapping image_id to captions for testing.
        vocab: Vocabulary instance.
        h5_path: Path to HDF5 features file.
        batch_size: Number of samples per batch.
        num_workers: Number of worker processes for data loading.
        collate_fn: Custom collate function for variable-length padding.

    Returns:
        Tuple of (train_loader, test_loader).
    """
    if collate_fn is None:
        collate_fn = Flickr8kDataset.collate_fn

    train_dataset = Flickr8kDataset(train_dict, vocab, h5_path=h5_path)
    test_dataset = Flickr8kDataset(test_dict, vocab, h5_path=h5_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=num_workers,
    )

    return train_loader, test_loader
