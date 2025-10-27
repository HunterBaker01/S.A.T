def test(model, dataloader, criterion, vocab_size):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, captions, _lengths in dataloader:
            images = images.to(DEVICE)
            captions = captions.to(DEVICE)
            outputs = model(images, captions)
            outputs = outputs[:, 1:, :].contiguous().view(-1, vocab_size)
            targets = captions[:, 1:].contiguous().view(-1)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

    avg_test_loss = total_loss / len(dataloader)
    return avg_test_loss
