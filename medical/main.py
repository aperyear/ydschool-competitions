import torch
import wandb
import argparse
import pandas as pd
from copy import deepcopy

from utils import seed_everything, save_history
from torch.utils.data import DataLoader

from model import MyModel, init_weight
from train import train_one_epoch, valid_one_epoch, MyLoss, EarlyStopper, CosineAnnealingWarmupRestarts
from data import MyDataset, get_train_transforms, get_valid_transforms, split_stratified_shuffle_split


parser = argparse.ArgumentParser(description='Pass variables for training')

parser.add_argument('--root', type=str, default='./data/train', help='Select root dir')
parser.add_argument('--data', type=str, default='./data/train.csv', help='Select csv data file')

parser.add_argument('--seed', type=int, default=2022, help='Select random seed number')
parser.add_argument('--fold', type=int, default=0, help='Select fold number')
parser.add_argument('--split', type=int, default=5, help='Select split number')
parser.add_argument('--batch', type=int, default=16, help='Select batch size')
parser.add_argument('--epoch', type=int, default=120, help='Select epoch number')
parser.add_argument('--model', type=str, default='inception_v3', help='Select timm model')
parser.add_argument('--init', type=str, default='kaiming', help='Select weight init kaiming, xavier')
parser.add_argument('--patience', type=int, default=30, help='Select early stopping patience number')
parser.add_argument('--lr', type=float, default=3e-4, help='Select warm-up learning rate')

parser.add_argument('--path', type=str, default='saved', help='Select save path of the model and checkpoint')
parser.add_argument('--name', type=str, default='default', help='Select unique name for the train purpose')

parser.add_argument('--verbose', action='store_true', help='Print training logs')
parser.add_argument('--wandb', action='store_true', help='Log wandb training logs')

args = parser.parse_args()

'''
python run.py --name [exp name] --model [model name] --batch [batch size]
image models: timm.list_models('*')
filter densenet: timm.list_models('*dense*')
filter(lambda x : 'dense' in x, timm.list_models('*'))
'''

def main(seed=args.seed, fold=args.fold, split=args.split, batch=args.batch, epoch=args.epoch, 
         model_name=args.model, initialization=args.init, save_name=args.name, lr=args.lr):
    
    CFG = {
        'seed': seed,
        'fold': fold, # 학습시킬 fold
        'n_split': split, # fold 개수
        'batch_size': batch,
        'num_classes': 2, # 1 for BCE
        'epoch': epoch,
        'model_name': model_name, # seresnet18, resnext50_32x4d...
        'initialization': initialization, # kaiming, xavier
        'color': 'rgb'
    }

    root_dir = args.root
    seed_everything(CFG['seed'])
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if args.wandb:
        experiment_name = save_name
        run = wandb.init(config=CFG,
                    project=f"{CFG['model_name']}", 
                    settings=wandb.Settings(start_method="thread"), 
                    name=f"{experiment_name}_fold{CFG['fold']}",
                    reinit=True)

    df = pd.read_csv(args.data)
    train_data, valid_data = split_stratified_shuffle_split(df, CFG['fold'], CFG['n_split'], CFG['seed'])

    train_transforms = get_train_transforms() # augmentation
    valid_transforms = get_valid_transforms()

    train_dataset = MyDataset(train_data, train_transforms, CFG['color'], root_dir)
    valid_dataset = MyDataset(valid_data, valid_transforms, CFG['color'], root_dir)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=CFG['batch_size'],
        num_workers=0,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,)

    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=CFG['batch_size'],
        num_workers=0,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False)

    model = MyModel(CFG['num_classes'], CFG['model_name']).to(device)
    init_weight(model, kind=CFG['initialization'])

    criterion = MyLoss().to(device)

    cosine_annealing_scheduler_arg = dict(
        first_cycle_steps=len(train_dataset)//CFG['batch_size'] * CFG['epoch'],
        cycle_mult=1.0,
        max_lr=lr,
        min_lr=1e-07,
        warmup_steps=len(train_dataset)//CFG['batch_size'] * 3,
        gamma=0.9
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.000, weight_decay=0)
    scheduler = CosineAnnealingWarmupRestarts(optimizer, **cosine_annealing_scheduler_arg)
    early_stopper = EarlyStopper(patience=args.patience)

    print('Start Training!')
    best_loss, best_score = float('inf'), 0
    acc_model, acc_optim, acc_sched = None, None, None
    loss_model, loss_optim, loss_sched = None, None, None
    loss_epoch, acc_epoch = None, None

    for i in range(CFG['epoch']):
        print(f"Epoch : {i}")
        train_losses, train_match = train_one_epoch(model, criterion, train_loader, scheduler, device, optimizer)
        train_loss, train_acc = train_losses / len(train_loader), train_match / len(train_loader.dataset)
        print(f"Train loss {train_loss :.4f}, score {train_acc :.4f}")
        
        valid_losses, valid_match = valid_one_epoch(model, criterion, valid_loader, device)
        valid_loss, valid_score = valid_losses / len(valid_loader), valid_match / len(valid_loader.dataset)
        print(f"Valid loss {valid_loss :.4f}, score {valid_score :.4f}")
        
        early_stopper.check_early_stopping(valid_score)

        if args.wandb:
            wandb_dict = {
                'train loss': train_loss,
                'train score': train_acc,
                'valid loss': valid_loss,
                'valid score': valid_score,
                'learning rate': scheduler.get_lr()[0]
            }
            wandb.log(wandb_dict)

        if best_loss > valid_loss:
            best_loss = valid_loss
            print(f'---best loss updated {best_loss :.4f}')
            
            loss_epoch = i
            loss_model = deepcopy(model.state_dict())
            loss_optim = deepcopy(optimizer.state_dict())
            loss_sched = deepcopy(scheduler.state_dict())

        if best_score < valid_score:
            best_score = valid_score
            print(f'---best acc updated {best_score :.4f}')
            
            acc_epoch = i
            acc_model = deepcopy(model.state_dict())
            acc_optim = deepcopy(optimizer.state_dict())
            acc_sched = deepcopy(scheduler.state_dict())

        if early_stopper.stop:
            break


    loss_dic = {
        'model': loss_model,
        'optimizer': loss_optim,
        'scheduler': loss_sched,
    }

    acc_dic = {
        'model': acc_model,
        'optimizer': acc_optim,
        'scheduler': acc_sched,
    }

    save_history(args.path, model_name, loss_dic, f"{fold}_{save_name}_{seed}_best_loss_{loss_epoch}_{best_loss :.4f}")
    save_history(args.path, model_name, acc_dic, f"{fold}_{save_name}_{seed}_best_acc_{acc_epoch}_{best_score :.4f}")
    return

if __name__ == '__main__':
    main()