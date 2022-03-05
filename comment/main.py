import time
import wandb
import numpy as np
import pandas as pd
from copy import deepcopy
from pprint import pprint

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from model import BertModel
from utils import seed_everything, save_history
from train import CE_F1_Loss, train_epoch, validate
from data import preprocess_df, split_stratified_shuffle_split, CustomDataset



def main(file_name, fold, n_split, data_path, model_name, input_col, target_col, batch_size, select, weight_decay,
         max_len, dropout, lr, scheduler_ratio, loss_fn_name, optim_name, epochs, freeze, save_path, seed, wandb_log, config):
    pprint(config)
    seed_everything(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    df = pd.read_csv(data_path)
    n_class = 3 if target_col == "bias" else 2
    bias_map = {'none': 0, 'gender': 1, 'others': 2}
    hate_map = {'none': 0, 'hate': 1}

    df = df.replace({'bias': bias_map, 'hate': hate_map})
    df['comment_title'] = df['comment'] + ' ' + df['title']
    df['title_comment'] = df['title'] + ' ' + df['comment']

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    df = preprocess_df(df, col=input_col)

    x_data = np.array([i for i in df[input_col].values])
    y_data = df[target_col].values

    train_index, valid_index = split_stratified_shuffle_split(df=df, fold=fold, n_split=n_split, input_col=input_col, target_col=target_col, seed=seed)

    x_train, y_train = x_data[train_index], y_data[train_index]
    x_valid, y_valid = x_data[valid_index], y_data[valid_index]
    print(f"train {len(x_train)} valid {len(x_valid)}")

    train_set = CustomDataset(x_train, y_train, tokenizer, max_len)
    valid_set = CustomDataset(x_valid, y_valid, tokenizer, max_len)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, pin_memory=True)

    model = BertModel(model_name=model_name, n_class=n_class, p=dropout, freeze=freeze, select=select)
    model.to(device)

    if wandb_log:
        run = wandb.init(config=config,
                        project=f"exp-{model_name.split('/')[1]}", 
                        settings=wandb.Settings(start_method="thread"), # start_method="fork" for linux / "thread" for colab
                        name=f"{fold}_{target_col}_{file_name}",
                        reinit=True)

    if loss_fn_name == "CELoss":
        loss_fn = nn.CrossEntropyLoss()
    elif loss_fn_name == "F1Loss": 
        loss_fn = CE_F1_Loss(n_class=n_class)

    if optim_name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optim_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: scheduler_ratio ** epoch)

    start = time.time()
    best_model, best_epoch, best_f1 = None, 0, 0

    print('--- train start ---')
    for i in range(epochs):
        train_loss, train_f1 = train_epoch(model, optimizer, loss_fn, train_loader)
        valid_loss, valid_f1 = validate(model, loss_fn, valid_loader)

        learning_rate = optimizer.param_groups[0]['lr']
        scheduler.step()

        if valid_f1 > best_f1:
            best_model = deepcopy(model.state_dict())
            best_f1 = valid_f1
            best_epoch = i
            print(f'best f1 updated {best_f1}')

        if wandb_log:
            wandb_dict = {
                'train loss': train_loss,
                'train F1': train_f1,
                'valid loss': valid_loss,
                'valid F1': valid_f1,
                'best F1': best_f1,
                'learning rate': learning_rate,
            }
            wandb.log(wandb_dict)
        print(f'epoch {i+1 :2d} train {train_loss :.4f} valid {valid_loss :.4f} bf1 {best_f1 :.3f} lr {learning_rate :.5f} time {round(time.time() - start, 2)}s')
    state_dict = {
        'config': config,
        'model': best_model,
    }
    save_history(save_path, model_name.split('/')[1], state_dict, file_name=f"{fold}_{file_name}_{target_col}_{best_epoch}_{best_f1 :.3f}")



if __name__ == '__main__':
    main()