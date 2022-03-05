import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import KFold


def kf_split(data, fold, n_split, seed, shuffle=True):
    if shuffle:
        kf = KFold(n_splits=n_split, random_state=seed, shuffle=True)
    else:
        kf = KFold(n_splits=n_split)
        
    for i, (train_index, valid_index) in enumerate(kf.split(data)):
        if i == fold:
            _train, _valid = train_index, valid_index
    print(f'shuffle {shuffle} train {len(_train)} valid {len(_valid)}')
    return _train, _valid

def preprocess_input_data(df, y_cols, d_hour=168, steps=1, week=6, train=True):    
    weeks = [i * d_hour for i in range(1, week + 1)]
    for i in weeks:
        for y in y_cols:
            df[f'{y}D-{i}'] = df[y].shift(i)

    x_cols = None
    if train:
        df = df.dropna(axis=0)
        # only traffic info
        x_cols = [i for i in df.columns if i not in y_cols and 'Date' not in i] 
    return df, x_cols
    
def scaling(df, y_cols, scaler=None):
    if scaler is None:
        scaler = PowerTransformer()
        df[y_cols] = scaler.fit_transform(df[y_cols])
        return df, scaler
    df[y_cols] = scaler.transform(df[y_cols])
    return df

def inverse_scaling(df, y_cols, scaler=None):
    try: df[y_cols] = scaler.inverse_transform(df[y_cols])
    except: df = scaler.inverse_transform(df)
    return df

class CustomDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        x = self.x_data[idx]
        ys = self.y_data[idx]
        
        return torch.tensor(x).float(), torch.tensor(ys).float()