#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Script to get FASTA sequence file from PDB."""

import argparse
from pathlib import Path

from fairseq.molecule_utils.config import pdb_cache_path
from fairseq.molecule_utils.database import get_af2_mmcif_object


def main():
    parser = argparse.ArgumentParser(description='Get FASTA sequence from PDB.')
    parser.add_argument('id', type=str, help='PDB ID')
    parser.add_argument('chain_id', nargs='?', type=str, default=None, help='Chain ID, default to all')
    parser.add_argument('-c', '--cache-path', metavar='PATH', type=Path, default=pdb_cache_path(),
                        help='PDB file cache path, default to %(default)s')
    parser.add_argument('-H', '--no-fasta-header', action='store_true', help='Do not print FASTA header')

    args = parser.parse_args()

    mmcif_object = get_af2_mmcif_object(args.id, pdb_cache_path=args.cache_path)
    if args.chain_id is None:
        for chain_id, sequence in mmcif_object.chain_to_seqres.items():
            if not args.no_fasta_header:
                print(f'> {args.id} chain {chain_id}')
            print(sequence)
    else:
        if args.chain_id not in mmcif_object.chain_to_seqres:
            print(f'| ERROR: {args.id} chain {args.chain_id} does not exist.')
            exit(1)
        else:
            if not args.no_fasta_header:
                print(f'> {args.id} chain {args.chain_id}')
            print(mmcif_object.chain_to_seqres[args.chain_id])


if __name__ == '__main__':
    main()
