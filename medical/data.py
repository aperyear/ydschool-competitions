import cv2
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedShuffleSplit
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


class MyDataset(Dataset):
    def __init__(self, data, transforms, color=None, root='./data/train'):
        self.data = data
        self.transforms = transforms
        self.root = root
        self.color = color
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        src, trg = self.data[idx]
        img = cv2.imread(self.root + '/' + src)
        
        if self.color == 'gray':
            img = np.mean(img, axis=0)
        
        if self.transforms:
            img = self.transforms(image=img)['image']
        return img, trg


def get_train_transforms(): # CT image augmentation
    return A.Compose([
        A.HorizontalFlip(p=0.5), # 기본
        A.ShiftScaleRotate(scale_limit=0.15, rotate_limit=10, p=0.5), # 추가
        A.RandomBrightnessContrast(p=0.5), # 추가
        A.OpticalDistortion(p=0.5), # 실험1
        A.GridDistortion(p=0.5), # 실험1
        A.GaussNoise(p=0.5), # 실험2 
        A.Normalize(mean=[0.59534838, 0.5949003 , 0.59472117], 
                    std=[0.29705278, 0.29707744, 0.29702731]),
        ToTensorV2()
    ])


def get_valid_transforms():
    return A.Compose([
        A.Normalize(mean=[0.59534838, 0.5949003 , 0.59472117], 
                    std=[0.29705278, 0.29707744, 0.29702731]),
        ToTensorV2()
    ])


def split_stratified_shuffle_split(df, fold, n_split, seed):
    skf = StratifiedShuffleSplit(n_splits=n_split, train_size=1-(1/n_split), test_size=(1/n_split), random_state=seed)
    for idx, i in enumerate(skf.split(df['file_name'], df['COVID'])):
        if idx == fold:
            train_data = df.values[i[0]]
            valid_data = df.values[i[1]]
    return train_data, valid_data