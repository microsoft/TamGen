#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Quick query amino acids."""

import argparse

from fairseq.molecule_utils.database.common_utils import aa_3to1, aa_1to3


def main():
    parser = argparse.ArgumentParser(description='Quick query amino acids.')
    parser.add_argument('queries', nargs='+', help='Query string list, in 1 or 3 letters.')

    args = parser.parse_args()

    for query in args.queries:
        query = query.upper()
        if len(query) == 3:
            result = aa_3to1(query)
        elif len(query) == 1:
            result = aa_1to3(query)
        else:
            print(f'ERROR: Only accept 1 or 3 letters query, got {query}.')
            continue
        print(f'{query} => {result}')


if __name__ == '__main__':
    main()
