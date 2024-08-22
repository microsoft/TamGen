#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Script to plot SMILES."""

import argparse

from rdkit import Chem
from rdkit.Chem import Draw


def main():
    parser = argparse.ArgumentParser(description='Script to plot SMILES.')
    parser.add_argument('smiles', help='SMILES string')

    args = parser.parse_args()

    mol = Chem.MolFromSmiles(args.smiles)
    if mol is None:
        print('ERROR: Invalid SMILES')
        exit(1)
    fig = Draw.MolToImage(mol)
    fig.show()


if __name__ == '__main__':
    main()
