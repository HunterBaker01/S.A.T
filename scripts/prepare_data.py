import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

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
