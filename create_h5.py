import os
import h5py
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

transform = transforms.Compose([
transforms.Resize(256),
transforms.CenterCrop(224),
transforms.ToTensor(),
transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
    ),
])

def extract_features(image_path):
  image = Image.open(image_path).convert("RGB")
  image = transform(image).unsqueeze(0).to(device)

  with torch.no_grad():
    features = encoder(image)
  
  return features

def reshape_features(features):
  features = features.squeeze(0)
  features = features.permute(1, 2, 0)
  features = features.view(-1, 512) # Works for our model, needs changing if you change the output size of the encoder
  return features

def save_to_h5(model, image_dir, device, save_path="features.h5"):
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    image_id_to_path = {}

    for fname in os.listdir(image_dir):
      if fname.endswith(".jpg"):
        image_id = fname.split(".")[0]
        image_path = os.path.join(image_dir, fname)
        image_id_to_path[image_id] = image_path


    with h5py.File(save_path, "w") as h5f:
      for image_id, image_path in image_id_to_path.items():
        raw_features = extract_features(image_path)
        features = reshape_features(raw_features)
        
        h5f.create_dataset(
            name=image_id,
            data=features.cpu().numpy(),
            dtype="float32"
        )
