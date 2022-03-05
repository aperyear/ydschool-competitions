import torch
import wandb
import argparse
import pandas as pd
from torch.utils.data import DataLoader

from train import run_epoch
from utils import save_history, seed_everything
from data import scaling, CustomDataset, kf_split, preprocess_input_data

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Pass variables for training')

parser.add_argument('--root', type=str, default='./data', help='Select root dir')

parser.add_argument('--seed', type=int, default=2022, help='Select random seed number')
parser.add_argument('--shuffle', action='store_false', help='Select to not shuffle folds')

parser.add_argument('--fold', type=int, default=0, help='Select fold number')
parser.add_argument('--split', type=int, default=5, help='Select split number')
parser.add_argument('--batch', type=int, default=32, help='Select batch size')
parser.add_argument('--epoch', type=int, default=120, help='Select epoch number')

parser.add_argument('--dhour', type=int, default=168, help='Select number')
parser.add_argument('--steps', type=int, default=1, help='Select number')
parser.add_argument('--week', type=int, default=7, help='Select number')

parser.add_argument('--init', type=str, default=None, help='Select weight init kaiming, xavier') # kaiming, xavier
parser.add_argument('--lr', type=float, default=3e-4, help='Select warm-up learning rate')

parser.add_argument('--path', type=str, default='saved', help='Select save path of the model and checkpoint')
parser.add_argument('--name', type=str, default='default', help='Select unique name for the train purpose')
parser.add_argument('--proj', type=str, default='MLP', help='Select project name for wandb')

parser.add_argument('--verbose', action='store_true', help='Print training logs')
parser.add_argument('--wandb', action='store_true', help='Log wandb training logs')

args = parser.parse_args()

seed = args.seed
seed_everything(seed)

y_cols = ['10','100','1000','101','1020','1040','1100','120','1200','121','140','150',
          '1510','160','200','201','251','2510','270','300','3000','301','351','352',
          '370','400','450','4510','500','550','5510','600','6000','650','652']

device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")


def main(root=args.root, fold=args.fold, path=args.path, save_name=args.name, steps=args.steps, week=args.week, shuffle=args.shuffle,
         n_split=args.split, epoch=args.epoch, batch=args.batch, lr=args.lr, proj_name=args.proj, d_hour=args.dhour):
    
    CFG = {
        'proj_name': proj_name,
        'save_name': save_name,
        'seed': seed,
        'fold': fold, 
        'batch_size': batch,
        'epoch': epoch,
        'week': week,
        'device': device
    }

    if args.wandb:
        wandb.config = CFG
        run = wandb.init(config=CFG,
                        project=f"{CFG['proj_name']}", 
                        settings=wandb.Settings(start_method="thread"), 
                        name=f"{save_name}_fold{CFG['fold']}",
                        reinit=True)
        
    df = pd.read_csv(f'{root}/final_train.csv')

    df, y_scaler = scaling(df, y_cols)    
    df, x_cols = preprocess_input_data(df, y_cols, d_hour, steps, week=week)

    x_data = df[x_cols].values
    y_data = df[y_cols].values
    
    print(f'x_cols {len(x_cols)}, y_cols{len(y_cols)}, scaler {y_scaler}')
    
    train_index, valid_index = kf_split(x_data, fold, n_split, seed, shuffle)
    
    x_train, y_train = x_data[train_index], y_data[train_index]
    x_valid, y_valid = x_data[valid_index], y_data[valid_index]
    
    train_set = CustomDataset(x_train, y_train)
    valid_set = CustomDataset(x_valid, y_valid)

    train_loader = DataLoader(train_set, batch_size=batch, pin_memory=True, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_set, batch_size=batch, pin_memory=True)

    history_dict = run_epoch(epoch, train_loader, valid_loader, in_f=len(x_cols), out_f=len(y_cols), y_scaler=y_scaler,
                             init=args.init, lr=lr, wandb=wandb, log=args.wandb)
    
    history_dict['week'] = week
    save_history(path, proj_name, history_dict, 
                 f"{fold}_{save_name}_{history_dict['epoch']}_{history_dict['rmse']:.0f}")
    return

    
if __name__ == '__main__':
    main()