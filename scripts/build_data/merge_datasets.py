#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Merge two existed dataset."""

import argparse
import logging
from pathlib import Path
from os.path import dirname as parent

from fairseq.molecule_utils.external.fairseq_dataset_build_utils import merge_and_build
from fairseq.molecule_utils.basic import smiles_utils as smu


def main():
    parser = argparse.ArgumentParser(description='Merge two existed dataset.')
    parser.add_argument('-i1', '--input-dir-1', dest='input_dir_1', type=Path, required=True, help='Input data directory 1')
    parser.add_argument('--in1', '--input-name-1', dest='input_name_1', type=str, required=True, help='Input name 1')
    parser.add_argument('-i2', '--input-dir-2', dest='input_dir_2', type=Path, required=True, help='Input data directory 2')
    parser.add_argument('--in2', '--input-name-2', dest='input_name_2', type=str, required=True, help='Input name 2')
    parser.add_argument('-o', '--output-dir', type=Path, required=True, help='Output data directory')
    parser.add_argument('--on', '--output-name', dest='output_name', type=str, default=None,
                        help='Output name, default to input name')
    parser.add_argument('--keep-duplicates', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()

    logging.basicConfig(
        format='%(levelname)s:%(filename)s:%(message)s', level=logging.DEBUG if args.verbose else logging.INFO)
    smu.disable_rdkit_log()


    merge_and_build(
        input_dir_1=args.input_dir_1,
        input_name_1=args.input_name_1,
        input_dir_2=args.input_dir_2,
        input_name_2=args.input_name_2,
        output_dir=args.output_dir,
        output_name=args.output_name,
        fairseq_root=Path(parent(parent(parent(__file__)))), 
        pre_dicts_root=Path(parent(parent(parent(__file__)))) / 'dict',
        keep_duplicates=args.keep_duplicates,
    )


if __name__ == '__main__':
    main()
