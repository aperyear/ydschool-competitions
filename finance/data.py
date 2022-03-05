import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler


class CustomDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        x = self.x_data[idx]
        y = self.y_data[idx]

        try: len(y)
        except: y = [y]
        return torch.tensor(x).float(), torch.tensor(y).long()


def preprocess_data(df: pd.DataFrame, target: str = 'depvar', seed: int = 50):
    df = shuffle(df, random_state=seed)
    y_data = df[target]
    x_data = df.drop([target], axis=1)
    x_cols = x_data.columns.tolist()

    scaler = StandardScaler().fit(x_data[x_cols])
    x_data[x_cols] = scaler.transform(x_data[x_cols])
    return x_data, y_data, scaler, x_cols


def get_data_loader(x_train: np.array, y_train: np.array, x_valid: np.array, y_valid: np.array, batch_size: int):
    train_set = CustomDataset(x_train, y_train)
    valid_set = CustomDataset(x_valid, y_valid)

    train_loader = DataLoader(train_set, batch_size=batch_size, drop_last=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, drop_last=False)
    return train_loader, valid_loader


def split_data(i_idx: int, j_idx: int, data: np.array) -> np.array:
    train = np.concatenate([data[:i_idx], data[j_idx:]])
    valid = data[i_idx:j_idx]
    return train, valid