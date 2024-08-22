#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Build dataset from PDB ID list. Automatically find binding sites according to the ligands in the structure file.

PDB ID list format: CSV format with columns ([] means optional):
pdb_id,[ligand_inchi,uniprot_id]

For custom pdb strcuture files, you can put your structure files to the `--pdb-path` folder, and in the csv file, put the filenames in the `pdb_id` column. 
"""

import argparse
import logging
import random
from pathlib import Path
from os.path import dirname as parent

import numpy as np
from fairseq.molecule_utils.database.caching_utils import get_cache
from fairseq.molecule_utils.external.fairseq_dataset_build_utils import process_one_pdb, dump_data
from fairseq.molecule_utils import config
from fy_common_ext.io import csv_reader
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description='Build dataset from PDB ID list. Automatically find binding sites according to the ligands in the structure file.')
    parser.add_argument('pdb_id_list', type=Path, help='PDB ID list csv file (with optional ligand inchi)')
    parser.add_argument('name', help='Dataset name')
    parser.add_argument('-t', '--threshold', type=float, default=10.0,
                        help='Near ligand threshold, default is %(default)s')
    parser.add_argument('-o', '--output', type=Path, required=True,
                        help='Output data folder')
    parser.add_argument('-pp', '--pdb-path', type=Path, default=config.pdb_cache_path(), help='PDB file path')
    parser.add_argument('-c', '--cache-size', type=int, default=200, help='PDB file cache size')
    parser.add_argument('--min-heavy-atoms', type=int, default=8, help='minumum number of heavy atoms in a binding site')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()

    logging.basicConfig(
        format='%(levelname)s:%(filename)s:%(message)s', level=logging.DEBUG if args.verbose else logging.INFO)
    get_cache('af2_mmcif').max_size = args.cache_size
    np.random.seed(1234)
    random.seed(1234)

    with csv_reader(args.pdb_id_list, dict_reader=True) as reader:
        input_list = list(reader)

    all_data = []
    for index, input_row in enumerate(tqdm(input_list)):
        print(f'Processing {input_row["pdb_id"]}')
        data = process_one_pdb(
            index, input_row, threshold=args.threshold, pdb_mmcif_path=args.pdb_path,
            pdb_ccd_path=config.pdb_ccd_path(),
            min_heavy_atoms=args.min_heavy_atoms
        )
        if data is not None:
            all_data.append(data)

    dump_data(all_data, args.name, output_dir=args.output,
              fairseq_root=Path(parent(parent(parent(__file__)))), 
              pre_dicts_root=Path(parent(parent(parent(__file__)))) / 'dict', 
              max_len=1023)


if __name__ == '__main__':
    main()
