from tqdm import tqdm

DEVICE = "cuda"

def train_epoch(model, dataloader, criterion, optimizer, vocab_size, epoch):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}", unit="batch")
    for images, captions, _lengths in progress_bar:
        images = images.to(DEVICE)
        captions = captions.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images, captions)
        outputs = outputs[:, 1:, :].contiguous().view(-1, vocab_size)
        targets = captions[:, 1:].contiguous().view(-1)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(dataloader)
    return avg_loss
