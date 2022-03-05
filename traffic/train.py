import time
import torch
import torch.nn as nn
import torch.nn.init as init
from copy import deepcopy
from model import MLP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))

def get_model(in_f, out_f, lr=3e-4):
    model = MLP(in_features=in_f, out_features=out_f)
    model = model.to(device)
    loss_fn = RMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return model, loss_fn, optimizer

def train_epoch(model, optimizer, loss_fn, loader):
    model.train()
    losses = 0.
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        outputs = model(x)

        loss = loss_fn(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses += loss
    return losses/len(loader)

def validate(model, loss_fn, loader, scaler):
    model.eval()
    losses = scale_losses = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)

            loss = loss_fn(outputs, y)

            scale_outputs = torch.FloatTensor(scaler.inverse_transform(outputs.cpu()))
            scale_y = torch.FloatTensor(scaler.inverse_transform(y.cpu()))
            scale_loss = loss_fn(scale_outputs, scale_y)

            losses += loss
            scale_losses += scale_loss
    return loss/len(loader), scale_losses/len(loader)

def run_epoch(epochs, train_loader, valid_loader, in_f, out_f, y_scaler, init=None, lr=3e-4, wandb=None, log=False):
    start = time.time()    
    model, loss_fn, optimizer = get_model(in_f, out_f, lr=lr)
    best_model, best_loss, best_scale, best_epoch = None, float('inf'), float('inf'), 0
    
    if init is not None:
        init_weight(model, kind=init)
        
    for i in range(epochs):
        train_loss = train_epoch(model, optimizer, loss_fn, train_loader)
        valid_loss, scale_loss = validate(model, loss_fn, valid_loader, y_scaler)
        
        if best_loss > valid_loss:
            best_model, best_scale, best_epoch, best_loss = deepcopy(model.state_dict()), scale_loss, i, valid_loss
            
        learning_rate = optimizer.param_groups[0]['lr']
        if log:
            wandb_dict = {
                'train loss': train_loss,
                'valid loss': valid_loss,
                'RMSE': scale_loss,
                'learning rate': learning_rate
            }
            wandb.log(wandb_dict)
        
        print(f'epoch {i+1 :2d} train {train_loss :.4f} valid {valid_loss :.4f} {scale_loss :.0f} LR {learning_rate :.5f} TIME {time.time() - start :.2f}s')
        
    history_dict = {
        'model': best_model,
        'rmse': best_scale,
        'epoch': best_epoch,
    }
    return history_dict

def init_weight(model, kind='xavier'):
    for name, i in model.named_parameters():
        if kind == 'xavier':
            if i.dim() < 2:
                continue
            if 'weight' in name:
                init.xavier_normal_(i, gain=1.0)
            elif 'bias' in name:
                init.xavier_uniform_(i, gain=1.0)
            else:
                pass
        elif kind == 'kaiming':
            if i.dim() < 2:
                continue
            if 'weight' in name:
                init.kaiming_normal_(i)
            elif 'bias' in name:
                init.kaiming_uniform_(i)
            else:
                pass