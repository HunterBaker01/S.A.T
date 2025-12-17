import torch
from torch import nn, optim
from torchvision import transforms
from models import MyVGG16, MyLSTM
from get_data import build_vocab, get_loaders, Flickr8kDataset
from eval import train_epoch

EMBED_DIM = 256
HIDDEN_DIM = 512
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 5
MIN_WORD_FREQ = 2
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4
 
IMAGES_DIR = "data/Images"
TOKENS_FILE = "data/captions.txt"
HDF5_FILE = "data/features.h5"
 
BEST_CHECKPOINT_PATH = "best_checkpoint.pth"
FINAL_MODEL_PATH = "final_model.pth"
VOCAB_PATH = "vocab.pkl"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

def main():
    train_dict, test_dict, vocab = build_vocab(TOKENS_FILE, MIN_WORD_FREQ, VOCAB_PATH)
    vocab_size = len(vocab)

    train_loader, test_loader = get_loaders(train_dict,
                                            test_dict,
                                            vocab,
                                            HDF5_FILE)

    encoder = MyVGG16()
    decoder = MyLSTM(EMBED_DIM, HIDDEN_DIM, vocab_size)

    optimizer = optim.Adam(decoder.parameters(), LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    min_loss = 10000
    for epoch in range(EPOCHS):
        loss = train_epoch(model=decoder,
                    dataloader=train_loader,
                    criterion=criterion,
                    optimizer=optimizer,
                    vocab_size=vocab_size,
                    epoch=epoch,
                    device=DEVICE)
        if loss < min_loss:
            min_loss = loss

    print(min_loss)
