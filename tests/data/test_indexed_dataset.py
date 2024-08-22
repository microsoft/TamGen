#! /usr/bin/python
# -*- coding: utf-8 -*-

import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import torch

from fairseq.data import indexed_dataset as ind
from fairseq.data import data_utils


class TestIndexedDataset(unittest.TestCase):
    def testNewMMapIndexedDataset(self):
        np_data = [np.ones((4, 3, 2), dtype=np.float32), np.zeros((5, 3, 2), dtype=np.float32)]

        tmp_dir = tempfile.TemporaryDirectory()
        tmp_path = tmp_dir.name
        prefix = tmp_path + '/temp_indexed_dataset'
        ind.binarize_data(np_data, prefix, dtype=None, dim='auto')
        dataset = data_utils.load_indexed_dataset(prefix, dictionary=None, dataset_impl='mmap')
        self.assertEqual(dataset[0].shape, torch.Size([4, 3, 2]))
        self.assertEqual(dataset[1].dtype, torch.float32)

        prefix = tmp_path + '/temp_indexed_dataset_2'
        ind.binarize_data(np_data, prefix, dtype=None, dim=None)
        dataset = data_utils.load_indexed_dataset(prefix, dictionary=None, dataset_impl='mmap')
        self.assertEqual(dataset[0].shape, torch.Size([24]))
        self.assertEqual(dataset[1].dtype, torch.int64)

        prefix = tmp_path + '/temp_indexed_dataset_3'
        np_1d_data = [np.ones((4,), dtype=np.float32), np.zeros((5,), dtype=np.float32)]
        ind.binarize_data(np_1d_data, prefix, dtype=None, dim=())
        dataset = data_utils.load_indexed_dataset(prefix, dictionary=None, dataset_impl='mmap')
        self.assertEqual(dataset[0].shape, torch.Size([4]))
        self.assertEqual(dataset[1].dtype, torch.float32)

        try:
            tmp_dir.cleanup()
        except PermissionError:
            pass


if __name__ == '__main__':
    unittest.main()
