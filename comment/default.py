config = {
    'file_name': 'default',                     # default: default
    'fold': 0,                                  # default: 0
    'n_split': 5,                               # default: 5
    'model_name': 'beomi/KcELECTRA-base',       # beomi/kcbert-base, beomi/kcbert-large, beomi/KcELECTRA-base
    'input_col': 'comment',                     # comment, title, comment_title, title_comment
    'target_col': 'bias',                       # bias, hate
    'batch_size': 32,                           # default: 32
    'dropout': 0.1,                             # default: 0.1
    'loss_fn_name': 'CELoss',                   # default: CELoss # CELoss, F1Loss
    'optim_name': 'AdamW',                      # default: AdamW # AdamW, SGD
    'lr': 3e-5,                                 # default: 3e-5
    'scheduler_ratio': 1,                       # defulat: 1
    'weight_decay': 0.01,                       # default: 0.01
    'epochs': 4,                                # default: 4
    'freeze': False,                            # default: False
    'save_path': './saved',                     # default: './saved'       
    'seed': 2022,                               # default: 2022
    'data_path': './train.csv',                 # default: './train.csv'
    'max_len': 128,                             # default: 128
    'select': False,                            # default: False
    'wandb_log': False,                         # default: False
}