import torch
import torch.nn as nn
from copy import deepcopy
from model import MLP
 
 
device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")

def get_model(lr: float = 5e-4):
    model = MLP()
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    return model, loss_fn, optimizer


def train_epoch(model, optimizer, loss_fn, loader):
    model.train()
    losses = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        output = model(x)
        loss = loss_fn(output, y.flatten())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses += loss.cpu().item() 
    return losses/len(loader)


def validate(model, loss_fn, loader):
    model.eval()
    losses = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = loss_fn(output, y.flatten())
            losses += loss.cpu().item()
    return losses/len(loader)


def run_epoch(epochs, train_loader, valid_loader, history_dict, fold_num, lr, verbose=False) -> dict:
    model, loss_fn, optimizer = get_model(lr)
    train_history, valid_history = [], []
    best_model, best_loss = None, float('inf')
        
    for i in range(epochs):
        train_loss = train_epoch(model, optimizer, loss_fn, train_loader)
        valid_loss = validate(model, loss_fn, valid_loader)

        if best_loss > valid_loss:
            best_model, best_loss = deepcopy(model.state_dict()), valid_loss

        train_history.append(train_loss)
        valid_history.append(valid_loss)

        learning_rate = optimizer.param_groups[0]['lr']
        if verbose: 
            print(f'epoch {i+1}/{epochs} train {train_loss :.4f} valid {valid_loss :.4f} lr {learning_rate :.4f} ')
    
    history_dict[fold_num] = [best_model, best_loss, train_history, valid_history]
    return history_dict