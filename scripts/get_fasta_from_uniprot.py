#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Script to get FASTA sequence file from UniProt."""

import argparse
from pathlib import Path

from fairseq.molecule_utils.config import uniprot_fasta_cache_path
from fairseq.molecule_utils.database import get_fasta_from_uniprot, download_fasta_file_from_uniprot


def main():
    parser = argparse.ArgumentParser(description='Get (download) FASTA file from UniProt.')
    parser.add_argument('id', type=str, help='UniProt ID')
    parser.add_argument('-c', '--cache-path', metavar='PATH', type=Path, default=uniprot_fasta_cache_path(),
                        help='UniProt FASTA file cache path, default to %(default)s')
    parser.add_argument('-s', '--get-seq', action='store_true', help='Return sequence value to stdout directly')

    args = parser.parse_args()

    if args.get_seq:
        sequence = get_fasta_from_uniprot(args.id, uniprot_fasta_cache_path=args.cache_path, get_str=True)
        print(sequence)
    else:
        fasta_path = download_fasta_file_from_uniprot(args.id, uniprot_fasta_cache_path=args.cache_path)
        if fasta_path is None:
            print(f'[ERROR] UniProt FASTA file {args.id} not found.')
        else:
            print(f'UniProt FASTA file {args.id} file downloaded to {fasta_path}.')


if __name__ == '__main__':
    main()
