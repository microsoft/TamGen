#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Some helper datasets."""

from torch.utils.data import Dataset


class SingleTensorDataset(Dataset):
    def __init__(self, tensor):
        super().__init__()
        self.tensor = tensor

    def __getitem__(self, index):
        return self.tensor[index]

    def __len__(self):
        return len(self.tensor)
