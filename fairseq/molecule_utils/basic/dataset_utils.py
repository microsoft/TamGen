#! /usr/bin/python
# -*- coding: utf-8 -*-

"""fairseq binary dataset utils."""

import itertools
import os
from typing import Iterable

import numpy as np

from ...data import indexed_dataset
from ...data import data_utils


def filter_text_file_by_mask(selector, in_filename: str, out_filename: str):
    if os.path.exists(out_filename) or not os.path.exists(in_filename):
        return
    if selector is None:
        selector = itertools.repeat(True)
    with open(in_filename, 'r', encoding='utf-8') as f_in_idx, \
            open(out_filename, 'w', encoding='utf-8') as f_out_idx:
        for line in itertools.compress(f_in_idx, selector):
            f_out_idx.write(line)


def filter_indexed_dataset_by_mask(
        masks, in_filename: str, out_filename: str,
        selector: Iterable[bool] = None, add_eos: bool = False):
    """Filter an indexed dataset and create a new one.
    
    Args:
        masks: None or list of np.ndarray of shape (seq_len,), dtype np.int32, value 0 (masked) or 1 (remained).
        in_filename: 
        out_filename: 
        selector: Bool iterable indicator to keep or discard this data case.
        add_eos: Usually for sequence dataset, append the final token (EOS) into the mask.

    Returns:

    """
    if indexed_dataset.dataset_exists(out_filename, impl='mmap'):
        return 
    if not indexed_dataset.dataset_exists(in_filename, impl='mmap'):
        return 
        
    dataset = data_utils.load_indexed_dataset(      # type: indexed_dataset.MMapIndexedDataset
        in_filename, dictionary=None, dataset_impl='mmap', default='mmap',
    )
    dim = dataset.dim
    dtype = dataset.dtype
    if selector is None:
        selector = itertools.repeat(True)
    if masks is None:
        masks = itertools.repeat(None)
    with indexed_dataset.mmap_idx_builder_env(
            prefix_path=out_filename, dtype=dtype, dim=dim) as builder:
        for i, (tensor, mask) in enumerate(itertools.compress(zip(dataset, masks), selector)):
            if mask is None:
                new_tensor = tensor
            else:
                assert len(tensor) == len(mask) + 1 if add_eos else len(tensor) == len(mask), \
                    f'data {i}: {len(tensor)} != {len(mask)}'
                where = np.where(mask)[0].tolist()
                if add_eos:
                    where.append(len(tensor) - 1)
                new_tensor = tensor[where, ...]
            builder.add_item(new_tensor)
