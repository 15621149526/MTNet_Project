import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class RMLDataset(Dataset):
    def __init__(self, path, mode='train'):
        super().__init__()
        with open(path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        
        X, y = [], []
        
        mods = sorted(list(set(map(lambda x: x[0], data.keys()))))
        self.num_classes = len(mods) 
        
        for k in data.keys():
            for i in range(len(data[k])):
                X.append(data[k][i])
                y.append(mods.index(k[0]))
        
        X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42)
        
        mapping = {'train': (X_train, y_train), 'val': (X_val, y_val), 'test': (X_test, y_test)}
        self.X, self.y = mapping[mode]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx])