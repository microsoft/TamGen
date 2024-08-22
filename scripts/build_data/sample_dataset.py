#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Sample exist dataset."""

import argparse
import logging
from pathlib import Path
from os.path import dirname as parent

from fy_common_ext.io import csv_reader

from fairseq.molecule_utils.external.fairseq_dataset_build_utils import sample_and_build
from fairseq.molecule_utils.basic import smiles_utils as smu


def main():
    parser = argparse.ArgumentParser(description='Sample dataset.')
    parser.add_argument('-i', '--input-dir', type=Path, required=True, help='Input data directory')
    parser.add_argument('--in', '--input-name', dest='input_name', type=str, required=True, help='Input name')
    parser.add_argument('-o', '--output-dir', type=Path, required=True, help='Output data directory')
    parser.add_argument('--on', '--output-name', dest='output_name', type=str, default=None,
                        help='Output name, default to input name')
    parser.add_argument('--data-id-list', type=Path, default=None,
                        help='Sample Data ID list filename (this will OVERWRITE all other filters).')
    parser.add_argument('-n', '--num-sample', type=int, default=100, help='Sample number')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()

    logging.basicConfig(
        format='%(levelname)s:%(filename)s:%(message)s', level=logging.DEBUG if args.verbose else logging.INFO)
    smu.disable_rdkit_log()

    data_id_list = None
    if args.data_id_list is not None:
        with open(args.data_id_list, 'r', encoding='utf-8') as f_di:
            data_id_list = sorted(int(line.strip()) for line in f_di)

    sample_and_build(
        input_dir=args.input_dir,
        input_name=args.input_name,
        output_dir=args.output_dir,
        output_name=args.output_name,
        fairseq_root=Path(parent(parent(parent(__file__)))), 
        pre_dicts_root=Path(parent(parent(parent(__file__)))) / 'dict',
        data_id_list=data_id_list,
        num_sample=args.num_sample,
    )


if __name__ == '__main__':
    main()
