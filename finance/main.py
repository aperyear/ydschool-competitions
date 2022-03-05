import torch
import argparse
import pandas as pd
from train import run_epoch
from utils import seed_everything, save_history
from data import preprocess_data, get_data_loader, split_data


parser = argparse.ArgumentParser(description='Pass variables for training')

parser.add_argument('--data', type=str, default='./data/train.csv', help='Select csv data file')
parser.add_argument('--fold', type=int, default=5, help='Select fold numbers... 5 folds -> 5 loops')
parser.add_argument('--all', action='store_true', help='type --all to loop all the folds, otherwise run only one fold')

parser.add_argument('--batch', type=int, default=32, help='Select batch size')
parser.add_argument('--epoch', type=int, default=50, help='Select epoch number')
parser.add_argument('--lr', type=float, default=1e-3, help='Select learning rate of the optimizer')

parser.add_argument('--path', type=str, default='saved', help='Select save path of the model and checkpoint')
parser.add_argument('--name', type=str, default='unique_name', help='Select unique name for the train purpose')

parser.add_argument('--seed', type=int, default=50, help='Select random seed number')
parser.add_argument('--verbose', action='store_true', help='Print training logs')

args = parser.parse_args()


def main(csv_file=args.data, fold_num=args.fold, path=args.path, name=args.name, 
            epoch=args.epoch, batch=args.batch, lr=args.lr, verbose=args.verbose):
    seed_everything(args.seed)
    
    df_data = pd.read_csv(csv_file)
    x_data, y_data, scaler, x_cols = preprocess_data(df=df_data)

    fold_list = range(0, len(x_data) + 1, int(len(x_data) / fold_num)) if args.all else [0, int(len(x_data) * 0.2)]
    
    history_dict = {'scaler': scaler, 'x_cols': x_cols}

    for i in range(len(fold_list) - 1):
        x_train, x_valid = split_data(fold_list[i], fold_list[i+1], x_data.values)
        y_train, y_valid = split_data(fold_list[i], fold_list[i+1], y_data.values)

        train_loader, valid_loader = get_data_loader(x_train, y_train, x_valid, y_valid, batch_size=batch)

        history_dict = run_epoch(epoch, train_loader, valid_loader, history_dict, i, lr, verbose=verbose)
        save_history(path, name, history_dict)
    return



if __name__ == '__main__':
    main()