import torchvision.transforms as transforms
from torchvision.datasets import Flickr8k
from torch.utils.data import random_split

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

def get_data(root, annotations_file, transform=transform, train_size=0.8):
    dataset = Flickr8k(
        root=root,
        ann_file=annotations_file,
        transform=transform
    )
    
    train_size = int(train_size * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size]
    )

    return train_dataset, test_dataset
