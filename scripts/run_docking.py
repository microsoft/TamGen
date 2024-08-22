#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Entry point of run docking.

Examples
--------

# Run with autobox
python run_docking.py 1t09 c1ccccc1

# Run with box center
python run_docking.py 1t09 c1ccccc1 --xyz 0 0.5 1
"""

import argparse
import logging
import pickle
from pathlib import Path

from fairseq.molecule_utils.basic.run_docking import docking
from fairseq.molecule_utils.config import pdb_cache_path, split_pdb_cache_path, pdb_ccd_path


def main():
    parser = argparse.ArgumentParser(
        description='Run docking.')
    parser.add_argument('id', type=str, help='PDB ID')
    parser.add_argument('ligand_smiles', type=str, help='Ligand SMILES')
    parser.add_argument('--xyz', '--box-center', nargs=3, metavar='FLOAT', dest='box_center', type=float,
                        default=None, help='Optional box center coordinate')
    parser.add_argument('--lwh', '--box-size', nargs=3, metavar='L W H', dest='box_size', type=float,
                        default=None, help='Optional box size of length, width, height')
    parser.add_argument('-b', '--smina-bin-path', metavar='PATH', type=Path,
                        help='AutoDock-smina binary path')
    parser.add_argument('-o', '--output-complex-path', metavar='PATH', type=Path, default=None,
                        help='Output receptor-ligand complex path')
    parser.add_argument('-c', '--cache-path', metavar='PATH', type=Path, default=pdb_cache_path(),
                        help='PDB file cache path, default to %(default)s')
    parser.add_argument('-s', '--split-cache-path', metavar='PATH', type=Path, default=split_pdb_cache_path(),
                        help='Split PDB file cache path, default to %(default)s')
    parser.add_argument('--ccd', '--ccd-cache-path', metavar='PATH', dest='ccd_cache_path', type=Path,
                        default=pdb_ccd_path(), help='CCD file cache path, default to %(default)s')
    parser.add_argument('--drc', '--docking-result-cache-path', metavar='PATH', dest='docking_result_cache_path',
                        type=Path, default=None, help='Docking result cache path, default to %(default)s')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose mode')

    args = parser.parse_args()

    logging_level = logging.DEBUG if args.verbose else None
    logging.basicConfig(format='%(levelname)s:%(filename)s:%(message)s', level=logging_level)

    docking_result_cache = None
    if args.docking_result_cache_path is not None:
        with open(args.docking_result_cache_path, 'rb') as f_drc:
            docking_result_cache = pickle.load(f_drc)

    affinity = docking(
        pdb_id=args.id,
        ligand_smiles=args.ligand_smiles,
        output_complex_path=args.output_complex_path,
        smina_bin_path=args.smina_bin_path,
        split_cache_path=args.split_cache_path,
        pdb_cache_path=args.cache_path,
        ccd_cache_path=args.ccd_cache_path,
        docking_result_cache=docking_result_cache,
        box_center=args.box_center,
        box_size=args.box_size,
    )
    print(affinity)

    if args.docking_result_cache_path is not None:
        with open(args.docking_result_cache_path, 'wb') as f_drc:
            pickle.dump(docking_result_cache, f_drc)


if __name__ == '__main__':
    main()
