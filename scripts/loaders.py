import torchvision.transforms as transforms
from get_data import Flickr8kDataset

def get_loaders(train_dict, test_dict, vocab, transform):
    train_dataset = Flickr8kDataset(train_dict, vocab, transform=transform)
    test_dataset = Flickr8kDataset(test_dict, vocab, transform=transform)

    train_loader = Dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=num_workers,
    )

    test_loader= Dataloader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=num_workers,
    )
    return train_loader, test_loader
