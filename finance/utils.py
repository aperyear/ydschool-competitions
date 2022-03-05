import os
import pytz
import torch
import random
import numpy as np
from datetime import datetime


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    return


def save_history(path: str, name: str, state_dict: dict, file_name: str = None) -> None:
    if not os.path.exists(f"{path}/{name}"):
        os.makedirs(f"{path}/{name}")
    
    if file_name is None:
        now = datetime.now(pytz.timezone('Asia/Seoul')).strftime('%Y-%m-%d-%H.%M.%S')
        file_name = f"{len(state_dict)}-{now}"
    torch.save(state_dict, f"{path}/{name}/{file_name}.pt")
    return