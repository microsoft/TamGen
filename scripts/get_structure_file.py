#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Script to get (download) structure file."""

import argparse
from pathlib import Path

from fairseq.molecule_utils.config import pdb_cache_path
from fairseq.molecule_utils.database import download_structure_file


def main():
    parser = argparse.ArgumentParser(description='Get (download) structure file (mmCIF or PDB).')
    parser.add_argument('id', type=str, help='PDB ID')
    parser.add_argument('-c', '--cache-path', metavar='PATH', type=Path, default=pdb_cache_path(),
                        help='Structure file cache path, default to %(default)s')
    parser.add_argument('-f', '--format', choices=['pdb', 'cif'], default='cif',
                        help='Format to download, default to "cif".')

    args = parser.parse_args()

    Path(args.cache_path).mkdir(exist_ok=True, parents=True)
    pdb_path = download_structure_file(
        args.id, pdb_cache_path=args.cache_path, ext='.' + args.format)

    if pdb_path is None:
        print(f'[ERROR] Structure file {args.id} ({args.format}) not found.')
    else:
        print(f'Structure {args.id} ({args.format}) file downloaded to {pdb_path}.')


if __name__ == '__main__':
    main()
