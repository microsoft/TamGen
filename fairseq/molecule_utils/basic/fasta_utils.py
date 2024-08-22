#! /usr/bin/python
# -*- coding: utf-8 -*-

import logging
import math
from typing import Optional, Iterable, Set

from editdistance import eval as editdistance

try:
    from rich import print as rprint
except ImportError:
    rprint = None

__all__ = [
    'aligned_print',
    'editdistance', 'normalized_editdistance'
]


def aligned_print(seq: str, highlight_indices: Iterable[int] = None):
    if highlight_indices is not None and rprint is None:
        logging.warning('FASTA highlighting is not supported.')
        highlight_indices = None

    highlight_set = set()
    if highlight_indices is not None:
        for index in highlight_indices:
            highlight_set.add(index + 1)

    print('\t' + ''.join(f'         {i}' for i in range(1, 10)))
    print('\t' + '1234567890' * 10)
    print('0', end='\t')
    for i, c in enumerate(seq, start=1):
        if i in highlight_set:
            rprint('[bold black on white]' + c, end='')
        else:
            print(c, end='')
        if i % 100 == 0:
            print()
            print(i // 100, end='\t')
    print()


def normalized_editdistance(qry, ref) -> float:
    ed = editdistance(qry, ref)
    if not ref:
        return math.inf
    return ed / len(ref)
