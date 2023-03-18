import os
import numpy as np
import pandas as pd
import mrcfile
import torch
from torch.utils.data import Dataset


class Subtomograms(Dataset):
    def __init__(self,dataset):
        # dataset == 'train' OR 'test'
        
        self.dataset = dataset
        assert dataset in ['train','test']
        
        if dataset=='train':
            self.df = pd.read_csv('train.csv')
        
        else:
            self.df = pd.read_csv('test.csv')
            
        self.n = df.shape[0]
        
    def __len__(self):
        
        return self.n
    
    def __getitem__(self, index):
        subtomogram_path = self.df.iloc[index][0]
        
        with mrcfile.open(subtomogram_path,permissive=True) as f:
            subtomogram = f.data
            f.close()
            
        subtomogram_processed = torch.Tensor(subtomogram)
        subtomogram_processed = subtomogram_processed.view(1,91,91,91)
        
        rescale = tio.RescaleIntensity(out_min_max=(0, 1))
        subtomogram_processed = rescale(subtomogram_processed)
        
        label = df.iloc[index][1]
        pdb_id = df.iloc[index][2]
        
        item = {'data':subtomogram_processed, 'label':label, 'index':index}
        
        return item
        