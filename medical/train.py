import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import _LRScheduler


def train_one_epoch(model, criterion, train_loader, scheduler, device, optimizer):
    model.train()

    losses = match = 0
    for idx, (src, trg) in tqdm(enumerate(train_loader)):
        src, trg = src.to(device).float(), trg.to(device).long() # BCE float, CEL long
        
        outs = model(src)
        preds = torch.argmax(outs, dim=-1)
        loss = criterion(outs, trg)

        # preds = outs.flatten() > 0.5 # BCE
        # loss = criterion(outs.flatten(), trg)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        losses += loss.item()
        match += (preds == trg).sum().item()
    return losses, match

    
def valid_one_epoch(model, criterion, valid_loader, device):
    model.eval()

    losses = match = 0
    with torch.no_grad():
        for idx, (src, trg) in tqdm(enumerate(valid_loader)):
            src, trg = src.to(device).float(), trg.to(device).long() # BCE float, CEL long
            
            outs = model(src)
            preds = torch.argmax(outs, dim=-1)
            loss = criterion(outs, trg)

            # preds = outs.flatten() > 0.5
            # loss = criterion(outs.flatten(), trg)
            
            losses += loss.item()
            match += (preds == trg).sum().item()
    return losses, match


class F1_Loss(nn.Module):
    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon
        
    def forward(self, y_pred, y_true,):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, 2).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=1)
        
        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2* (precision*recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1-self.epsilon)
        return 1 - f1.mean()


class MyLoss(_Loss):
    def __init__(self): # 실험환경 세팅으로
        super(MyLoss, self).__init__()
        self.lossCE = nn.CrossEntropyLoss()
        self.lossF1 = F1_Loss()
        # self.BCE = nn.BCEWithLogitsLoss()
        
    def forward(self, preds, trg):
        return (self.lossCE(preds, trg) + self.lossF1(preds, trg)) / 2
        # return self.BCE(preds, trg)


class EarlyStopper():
    def __init__(self, patience: int)-> None:
        self.patience = patience
        self.patience_counter = 0
        self.best_acc = 0
        self.stop = False
        self.save_model = False

    def check_early_stopping(self, score: float)-> None:
        if self.best_acc == 0:
            self.best_acc = score
            return None

        elif score <= self.best_acc:
            self.patience_counter += 1
            self.save_model = False
            if self.patience_counter == self.patience:
                self.stop = True
                
        elif score > self.best_acc:
            self.patience_counter = 0
            self.save_model = True
            self.best_acc = score
            print('best score :', self.best_acc)
        print("best_acc", self.best_acc)
        
        
class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 first_cycle_steps: int,
                 cycle_mult: float = 1.,
                 max_lr: float = 0.1,
                 min_lr: float = 0.001,
                 warmup_steps: int = 0,
                 gamma: float = 1.,
                 last_epoch: int = -1
                 ):
        assert warmup_steps < first_cycle_steps

        self.first_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle_mult = cycle_mult  # cycle steps magnification
        self.base_max_lr = max_lr  # first max learning rate
        self.max_lr = max_lr  # max learning rate in the current cycle
        self.min_lr = min_lr  # min learning rate
        self.warmup_steps = warmup_steps  # warmup step size
        self.gamma = gamma  # decrease rate of max learning rate by cycle

        self.cur_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle = 0  # cycle count
        self.step_in_cycle = last_epoch  # step size of the current cycle

        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)

        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr for base_lr in
                    self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle - self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int(
                    (self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(
                        self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr