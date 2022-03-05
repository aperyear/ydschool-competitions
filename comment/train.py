import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.nn.modules.loss import _Loss
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro")
    rec = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")
    print(f"acc {acc :.2f} prec {prec :.2f} rec {rec :.2f} f1 {f1 :.2f}")
    return f1

def train_epoch(model, optimizer, loss_fn, loader):
    model.train()
    losses = 0
    y_true, y_pred = [], []
    for x, y in tqdm(loader):
        mask = x['attention_mask'].squeeze(1).to(device)
        input_ids = x['input_ids'].squeeze(1).to(device)
        segment_ids = x['token_type_ids'].squeeze(1).to(device)
        y = y.to(device)

        output = model(input_ids, mask, segment_ids) 

        loss = loss_fn(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = list(output.argmax(dim=-1).cpu().numpy())
        trues = list(y.cpu().numpy())
        y_true += trues
        y_pred += preds
        losses += loss.cpu().item()

        # metrics(trues, preds) # print 
    print('train metrics')
    f1 = metrics(y_true, y_pred)
    return losses/len(loader), f1

def validate(model, loss_fn, loader):
    model.eval()
    losses = 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in tqdm(loader):
            mask = x['attention_mask'].squeeze(1).to(device)
            input_ids = x['input_ids'].squeeze(1).to(device)
            segment_ids = x['token_type_ids'].squeeze(1).to(device)
            y = y.to(device)

            output = model(input_ids, mask, segment_ids) 

            loss = loss_fn(output, y)

            preds = output.argmax(dim=-1)
            y_true += list(y.cpu().numpy())
            y_pred += list(preds.cpu().numpy())
            losses += loss.cpu().item()
    print('valid metrics')
    f1 = metrics(y_true, y_pred)
    return losses/len(loader), f1

class F1_Loss(nn.Module):
    def __init__(self, n_class=2, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon
        self.n_class = n_class

    def forward(self, y_pred, y_true):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, self.n_class).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=1)
        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision*recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1-self.epsilon)
        return 1 - f1.mean()

class CE_F1_Loss(_Loss):
    def __init__(self, n_class=2):
        super().__init__()
        self.lossCE = nn.CrossEntropyLoss()
        self.lossF1 = F1_Loss(n_class=n_class)
        
    def forward(self, output, y):
        return (self.lossCE(output, y) + self.lossF1(output, y)) / 2