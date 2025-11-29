import torchvision.transforms as transforms
import os
import torch
from PIL import Image

class Flickr8kDataset(Dataset):
    def __init__(self, imgid2captions, vocab, transform=transform):
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
        img_path = os.path.join(IMGAGES_DIR, img_id)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        numerical_caption = [self.vocab.ind["sos"]]
        numerical_caption += self.vocab.numerical(caption)
        numerical_caption.append(self.vocab.ind["eos"])

        return image, torch.tensor(numerical_caption, dtype=torch.long)

    def collate_fn(batch):
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
