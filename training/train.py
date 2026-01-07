import torch
import copy
from val import evaluate_model  

def train_and_validate(model, train_loader, val_loader, criterion, optimizer, scheduler, device, train_steps):
    model.to(device)
    best_val_acc = 0.0
    best_model_state = None
    train_iter = iter(train_loader)

    for step in range(train_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        model.train()
        inputs, labels = batch['x'].to(device), batch['y'].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Validation each 50 step, saving the best model.
        if (step + 1) % 50 == 0:
            val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = copy.deepcopy(model.state_dict())
                print(f" New Best @ Step {step+1}: {val_acc:.2f}% (Loss: {val_loss:.4f})")
            elif (step + 1) % 500 == 0:
                print(f" Step {step+1}: Acc={val_acc:.2f}% (Best: {best_val_acc:.2f}%)")

    if best_model_state is None:
        best_model_state = copy.deepcopy(model.state_dict())
        _, best_val_acc = evaluate_model(model, val_loader, criterion, device)

    return best_model_state, best_val_acc