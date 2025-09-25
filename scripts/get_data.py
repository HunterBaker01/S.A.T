import torchvision.transforms as transforms
import torchvision.datasets import Flickr8k

transform = transforms.Compose([
    transforms.Resize(224, 224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

def get_data(root=, annotations_file, transform=transform):
    dataset = Flickr8k(root=root, annotations_file=annotations_file, transform=transform)
    return dataset
