#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Build binarized dataset from arrays."""

import argparse
import pickle
from pathlib import Path
from typing import Union

import numpy as np
import torch

from fairseq.data import indexed_dataset


def _list2dict(data):
    if isinstance(data, (list, tuple)):
        return {index: value for index, value in enumerate(data)}
    return data


def _check_input(data: Union[dict, list]) -> bool:
    if not isinstance(data, dict):
        print('| ERROR: Input is not a dict.')
        return False
    if not data:
        print('| ERROR: Input is empty.')
        return False
    for key, value in data.items():
        if not isinstance(key, int):
            print(f'| ERROR: Input key {key} is not an int.')
            return False
        if not isinstance(value, (np.ndarray, torch.Tensor)):
            print(f'| ERROR: Input value data[{key}] is not a np.ndarray nor a torch.Tensor.')
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description='Build binarized dataset from arrays.')
    parser.add_argument('input', help='Input pickle path')
    parser.add_argument('output', nargs='?', help='Output dataset path, default to input.')
    parser.add_argument('-t', '--type', choices=['pickle', 'torch'], default='pickle',
                        help='Input format, default to %(default)s.')
    parser.add_argument('--keep-dtype', action='store_true',
                        help='Keep data type for 1D dataset (convert to int64 by default), '
                             'have no effect on ND datasets')

    args = parser.parse_args()

    input_fn = Path(args.input)
    if args.output is None:
        output_fn = input_fn.with_suffix('')
    else:
        output_fn = Path(args.output)

    if args.type == 'pickle':
        with open(input_fn, 'rb') as f_in:
            data = pickle.load(f_in)
    else:   # args.type == 'torch'
        data = torch.load(input_fn)
    data = _list2dict(data)
    if not _check_input(data):
        return

    sample_value = next(iter(data.values()))
    ndim = len(sample_value.shape)
    if args.keep_dtype and ndim == 1:
        dim = tuple()
    else:
        dim = 'auto'
    indexed_dataset.binarize_data(data, str(output_fn), dtype=None, dim=dim, verbose=True)
    print(f'| Binarize data {input_fn} => {indexed_dataset.index_file_path(str(output_fn))}, '
          f'{indexed_dataset.data_file_path(str(output_fn))}.')


if __name__ == '__main__':
    main()
