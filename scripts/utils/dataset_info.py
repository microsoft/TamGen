#! /usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import os
from collections import Counter
from pathlib import Path
from reprlib import repr

import numpy as np

from fairseq.data import data_utils, Dictionary, indexed_dataset
from fy_common_ext.io import csv_reader

from fairseq.molecule_utils.basic import inchi2smi, disable_rdkit_log


def _report_one_indexed_dataset(name: str, dictionary: Dictionary = None):
    dataset = data_utils.load_indexed_dataset(
        name,
        dictionary,
        dataset_impl='mmap',
        default='mmap',
    )
    sample = dataset[0]  # type: np.ndarray

    if dictionary is None or sample.ndim != 1:
        ends_with_eos = False
    else:
        ends_with_eos = sample[-1] == dictionary.eos()
    all_lengths = [len(e) for e in dataset]

    print('===== Binarize file information =====')
    print(f'''\
Type        : {type(dataset)}
Index file  : {indexed_dataset.index_file_path(name)}
Data file   : {indexed_dataset.data_file_path(name)}
Size        : {len(dataset)}
Average len : {np.mean(all_lengths):g} +- {np.std(all_lengths):g}
Dtype[0]    : {sample.dtype}
Shape[0]    : {sample.shape}
EOS?        : {ends_with_eos}
Data[0][:5] : {repr(sample[:5, ...].tolist())}
Data[0][-5:]: {repr(sample[-5:, ...].tolist())}
''')


def _report_info_csv(input_dir: str, ligand_top_n: int = 100):
    disable_rdkit_log()

    src_dir_name = Path(input_dir) / 'src'
    if not src_dir_name.exists() or not src_dir_name.is_dir():
        print(f'| ERROR: source file directory {src_dir_name} does not exist.')
        return
    info_filenames = list(src_dir_name.glob('*-info.csv'))
    for info_filename in info_filenames:
        with csv_reader(info_filename, dict_reader=True) as reader:
            all_rows = []
            ligand_counter = Counter()
            unique_fasta = set()
            for row in reader:
                all_rows.append(row)

                ligand_inchi = row.get('ligand_inchi', None)
                if ligand_inchi is not None:
                    ligand_counter[ligand_inchi] += 1

                unique_fasta.add(row['chain_fasta'])

        top_n_ligand = ligand_counter.most_common(ligand_top_n)
        print(f'''\
===== Info file {info_filename} information =====
Total size    : {len(all_rows)}
Unique ligands: {len(ligand_counter)}
Unique FASTA  : {len(unique_fasta)}
Top-{ligand_top_n} ligands: 
''')
        for i, (inchi, freq) in enumerate(top_n_ligand):
            smiles = inchi2smi(inchi)
            print(f'  {i:>3}: ({freq})')
            print(f'    {smiles}')
            print(f'    {inchi}')


def main():
    parser = argparse.ArgumentParser(description='Show binarized dataset information.')
    parser.add_argument('input', help='Binarize file or the dataset directory to read.')
    parser.add_argument('-m', '--mode', default='index', choices=['index', 'info'],
                        help='Report mode, default to %(default)r')
    parser.add_argument('-d', '--dict', help='Dictionary path', default=None)

    args = parser.parse_args()

    mode = args.mode
    if mode == 'info':
        _report_info_csv(input_dir=args.input)
        return

    if not os.path.isdir(args.input):
        dictionary = Dictionary.load(args.dict) if args.dict is not None else None
        print(f'''\
        Dict file   : {args.dict}
        ''')
        _report_one_indexed_dataset(args.input, dictionary)
        return

    input_path = Path(args.input)
    print(f'| Input {input_path} is a folder, automatically detect its datasets ...')

    for filename in sorted(input_path.iterdir()):
        if filename.suffix != '.idx':
            continue
        dataset_name = filename.with_name(filename.name[:-4])
        _report_one_indexed_dataset(str(dataset_name), dictionary=None)


if __name__ == '__main__':
    main()
