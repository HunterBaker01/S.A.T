from torch import nn
from torch import optim
from tqdm import tqdm

def train_epoch(model, dataloader, criterion, optimizer, vocab_size, epoch, device):
    """
    :param model: model to train, if the encodere is already pretrained just the decoder goes here
    :param dataloader: training dataset
    :param criterion: the loss function
    :param optimizer: optimizer to use
    :param vocab_size: size of the vocabulary that has been built
    :param epoch: current epoch
    :param device: the device to train on (GPU in this case)
    """
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}", unit="batch")
    for features, captions, _lengths in progress_bar:
        features = images.to(device)
        captions = captions.to(device)
        
        optimizer.zero_grad()
        outputs = model(features, captions)
        outputs = outputs[:, 1:, :].contiguous().view(-1, vocab_size)
        targets = captions[:, 1:].contiguous().view(-1)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def validate(model, dataloader, criterion, vocab_size):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, captions, _lengths in dataloader:
            images = images.to(device)
            captions = captions.to(device)
            outputs = model(images, captions)
            outputs = outputs[:, 1:, :].contiguous().view(-1, vocab_size)
            targets = captions[:, 1:].contiguous().view(-1)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

    avg_test_loss = total_loss / len(dataloader)
    return avg_test_loss
