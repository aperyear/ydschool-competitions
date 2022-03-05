import os
import torch
import random
import numpy as np


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
    torch.save(state_dict, f"{path}/{name}/{file_name}.pt")
    return