import torch

def evaluate_model(model, val_loader, criterion, device):
    val_loss = 0
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch['x'].to(device), batch['y'].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return val_loss / len(val_loader), 100. * correct / total