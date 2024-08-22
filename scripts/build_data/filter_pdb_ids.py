#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Filter exist dataset by PDB ID list."""

import argparse
import logging
from pathlib import Path
from os.path import dirname as parent

from fy_common_ext.io import csv_reader

from fairseq.molecule_utils.external.fairseq_dataset_build_utils import modify_and_build
from fairseq.molecule_utils.basic import smiles_utils as smu


def main():
    parser = argparse.ArgumentParser(description='Filter PDB IDs.')
    parser.add_argument('pdb_id_lists', nargs='*', type=Path, help='List of filter PDB ID lists.')
    parser.add_argument('-i', '--input-dir', type=Path, required=True, help='Input data directory')
    parser.add_argument('--in', '--input-name', dest='input_name', type=str, required=True, help='Input name')
    parser.add_argument('-o', '--output-dir', type=Path, required=True, help='Output data directory')
    parser.add_argument('--on', '--output-name', dest='output_name', type=str, default=None,
                        help='Output name, default to input name')
    parser.add_argument('--data-id-list', type=Path, default=None,
                        help='Data ID list filename (this will OVERWRITE all other filters).')
    parser.add_argument('--ma', '--min-heavy-atoms', dest='min_heavy_atoms', type=int, default=None,
                        help='Min heavy atoms')
    parser.add_argument('--fr', '--filter-ligand-rule', dest='filter_ligand_rule', type=str, default='none',
                        help='Extra filter ligand rule')
    parser.add_argument('--fl', '--filter-ligand-list', dest='filter_ligand_list', type=Path, default=None,
                        help='Path to filter ligand InChI list')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()

    logging.basicConfig(
        format='%(levelname)s:%(filename)s:%(message)s', level=logging.DEBUG if args.verbose else logging.INFO)
    smu.disable_rdkit_log()

    rm_pdb_ids = []
    for pdb_id_list in args.pdb_id_lists:
        with csv_reader(pdb_id_list, dict_reader=True) as reader:
            for row in reader:
                rm_pdb_ids.append(row['pdb_id'])

    if args.filter_ligand_list is not None:
        with open(args.filter_ligand_list, 'r', encoding='utf-8') as f_fl:
            rm_ligand_inchi = [line.strip() for line in f_fl]
    else:
        rm_ligand_inchi = []

    data_id_list = None
    if args.data_id_list is not None:
        with open(args.data_id_list, 'r', encoding='utf-8') as f_di:
            data_id_list = sorted(int(line.strip()) for line in f_di)

    modify_and_build(
        input_dir=args.input_dir,
        input_name=args.input_name,
        output_dir=args.output_dir,
        output_name=args.output_name,
        fairseq_root=Path(parent(parent(parent(__file__)))), 
        pre_dicts_root=Path(parent(parent(parent(__file__)))) / 'dict',
        data_id_list=data_id_list,
        rm_pdb_ids=rm_pdb_ids,
        rm_ligand_inchi=rm_ligand_inchi,
        min_heavy_atoms=args.min_heavy_atoms,
        rm_ligand_rule=args.filter_ligand_rule,
    )


if __name__ == '__main__':
    main()
