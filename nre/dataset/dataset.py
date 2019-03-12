"""
    Super class for data loader
"""

import torch
import os
from torch.utils import data


class DataSet(data.Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_data(self, data_name):
        data_file = os.path.join(self.data_dir, data_name + '.pt')
        return torch.load(data_file)

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class Batch(object):
    def __init__(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])
