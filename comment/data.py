import re
import emoji
from soynlp.normalizer import repeat_normalize

import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedShuffleSplit


# kc-electra pre-train preprocess function  
def preprocess_df(df, col='comment'):
    emojis = ''.join(emoji.UNICODE_EMOJI.keys())
    pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-힣{emojis}]+')
    url_pattern = re.compile(
        r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')
    
    def clean(x):
        x = pattern.sub(' ', x)
        x = url_pattern.sub('', x)
        x = x.strip()
        x = repeat_normalize(x, num_repeats=2)
        return x

    df[col] = df[col].map(lambda x: clean(str(x)))
    return df


def split_stratified_shuffle_split(df, fold, n_split, input_col='comment', target_col='hate', seed=2022):
    skf = StratifiedShuffleSplit(n_splits=n_split, test_size=1/n_split, random_state=seed)
    for idx, (train_index, valid_index) in enumerate(skf.split(df[input_col], df[target_col])):
        if idx == fold:
            return train_index, valid_index


class CustomDataset(Dataset):
    def __init__(self, x_data, y_data, tokenizer, max_len):
        self.x_data = x_data
        self.y_data = y_data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = self.x_data[idx]
        y = self.y_data[idx]
        
        tokenized_text = self.tokenizer(x,
                             padding= 'max_length',
                             max_length=self.max_len,
                             truncation=True,
                             return_token_type_ids=True,
                             return_attention_mask=True,
                             return_tensors = "pt")
        
        data = {'input_ids': tokenized_text['input_ids'].clone().detach().long(),
               'attention_mask': tokenized_text['attention_mask'].clone().detach().long(),
               'token_type_ids': tokenized_text['token_type_ids'].clone().detach().long(),
               }
        return data, torch.tensor(y).long()