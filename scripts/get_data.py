from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np
import re
from collections import Counter
import csv
import torchvision.transforms as transforms
import os
import torch
import pickle
import random

class ImageCaptionDataset(Dataset):
    def __init__(self, image_dir, captions_file, transform=None):
        """
        :param image_dir: directory containing the images
        :param captions_file: file containing the captions
        :param transform: transformation to be applied to the images
        """
        self.image_dir = image_dir
        self.transform = transform
        self.captions = {}
        with open (captions_file, 'r') as f:
            # This block is used for Flickr8k format, may need changing depending on how
            # the captions file is formatted
            for line in f:
                parts = line.strip().split(',')
                image_id = parts[0]
                caption = parts[1]
                if image_id not in self.captions:
                    self.captions[image_id] = [] # dictionary to map image_id to captions
                self.captions[image_id].append(caption) # each id can have more than one caption
        self.image_ids = list(self.captions.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        # again this works for the directory setup with an images dir and a captions file
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.image_dir, image_id)
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        caption = np.random.choice(self.captions[image_id])
        return image, caption

class Vocabulary:
    def __init__(self, freq_threshold=5):
        """
        :param freq_threshold: number of times a word must appear to be added to the vocab
        """
        self.freq_threshold = freq_threshold
        self.toks = {0: "pad", 1: "sos", 2: "eos", 3: "unk"} # Predefined tokens
        self.ind = {v: k for k, v in self.toks.items()} # reversed positions of the above
        self.index = 4 # starting point for the new vocab

    def __len__(self):
        return len(self.toks)

    def tockenizer(self, text):
        text = text.lower()
        tockens = re.findall(r"\w+", text) # make each word a token
        return tockens

    def build_vocab(self, sentence_list):
        frequencies = Counter()
        for sentence in sentence_list:
            tokens = self.tockenizer(sentence)
            frequencies.update(tokens) # update how often a word is used

        for word, freq in frequencies.items():
            if freq >= self.freq_threshold:
                self.ind[word] = self.index # update the dict that holds the tokens
                self.toks[self.index] = word
                self.index += 1

    def numerical(self, text):
        tokens = self.tockenizer(text)
        numerical = []
        for token in tokens:
            if token in self.ind:
                numerical.append(self.ind[token])
            else:
                numerical.append(self.ind["unk"])
        return numerical

def parse_tokens(file):
    imgid2captions = {}
    with open(file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            img_id, caption = row[0], row[1]
            if img_id not in imgid2captions:
                imgid2captions[img_id] = []
            imgid2captions[img_id].append(caption)
    return imgid2captions

class Flickr8kDataset(Dataset):
    def __init__(self, imgid2captions, vocab, transform=None, images_dir="data/Images"):
        """
        :param imgid2captions: dict mapping image id to the caption
        :param vocab: the vocabulary to use for training the model
        :param transform: transformation for the images
        :param images_dir: directory containing the images
        """

        self.imgid2captions = []
        self.transform = transform
        self.vocab = vocab

        for img_id, caps in imgid2captions.items():
            for c in caps:
                self.imgid2captions.append((img_id, c))

    def __len__(self):
        return len(self.imgid2captions)

    def __getitem__(self, idx):
        img_id, caption = self.imgid2captions[idx]
        img_path = os.path.join(images_dir, img_id)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        numerical_caption = [self.vocab.ind["sos"]]
        numerical_caption += self.vocab.numerical(caption)
        numerical_caption.append(self.vocab.ind["eos"])

        return image, torch.tensor(numerical_caption, dtype=torch.long)

    def collate_fn(batch):
        """
        Used to allow for different sized captions by padding each caption in
        the batch with 0's to the length of the longest caption.
        """
        batch.sort(key=lambda x: len(x[1]), reverse=True)
        images = [item[0] for item in batch]
        captions = [item[1] for item in batch]
        lengths = [len(cap) for cap in captions]
        max_len = max(lengths)

        padded_captions = torch.zeros(len(captions), max_len, dtype=torch.long)
        for i, cap in enumerate(captions):
            end = lengths[i]
            padded_captions[i, :end] = cap[:end]

        images = torch.stack(images, dim=0)
        return images, padded_captions, lengths

def build_vocab(tokens_file, min_word_freq, vocab_path):
    imgid2captions = parse_tokens(tokens_file)
    all_captions = []
    for caps in imgid2captions.values():
        all_captions.extend(caps)

    vocab = Vocabulary(freq_threshold=min_word_freq)
    vocab.build_vocab(all_captions)

    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)

    print("Vocabulary saved to: ", vocab_path)

    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")

    img_ids = list(imgid2captions.keys())
    random.shuffle(img_ids)
    split_idx = int(0.8 * len(img_ids))
    train_ids = img_ids[:split_idx]
    test_ids = img_ids[split_idx:]

    train_dict = {iid: imgid2captions[iid] for iid in train_ids}
    test_dict= {iid: imgid2captions[iid] for iid in test_ids}
    return train_dict, test_dict, vocab

def get_loaders(train_dict, test_dict, vocab, transform):
    train_dataset = Flickr8kDataset(train_dict, vocab, transform=None)
    test_dataset = Flickr8kDataset(test_dict, vocab, transform=None)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=NUM_WORKERS,
    )

    test_loader= DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=NUM_WORKERS,
    )
    return train_loader, test_loader
