#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Split a PDB file that contains target-ligand complex into separate PDB files."""

import argparse
import logging
from pathlib import Path
from pprint import pformat

from fairseq.molecule_utils.config import pdb_cache_path, split_pdb_cache_path, pdb_ccd_path, set_dataset_root
from fairseq.molecule_utils.database import split_pdb_complex_paths


def main():
    parser = argparse.ArgumentParser(
        description='Split a structure file that contains target-ligand complex into separate structure files.')
    parser.add_argument('id', type=str, help='PDB ID')

    args = parser.parse_args()

    split_result = split_pdb_complex_paths(
        args.id,
        split_ext='.pdb',
        split_cache_path=split_pdb_cache_path(),
        pdb_cache_path=pdb_cache_path(),
        ccd_cache_path=pdb_ccd_path(),
    )
    if split_result.target_filename is None:
        print(f'[ERROR] Structure file {args.id} not found.')
    else:
        print(f'''\
Structure file {args.id} split success.
Structure without ligands and water: {split_result.target_filename}
Ligand information:
{pformat(split_result.ligand_info)}
Ligand files:''')
        for lf in split_result.ligand_filenames:
            print(lf)


if __name__ == '__main__':
    main()
