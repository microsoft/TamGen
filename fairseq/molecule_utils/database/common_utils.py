#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Common functions for structure files, proteins and amino acids."""

from Bio.Data import PDBData

_1TO3_TABLE = {
    'A': 'ALA', 'B': 'ASX', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE',
    'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'J': 'XLE', 'K': 'LYS', 'L': 'LEU',
    'M': 'MET', 'N': 'ASN', 'O': 'PYL', 'P': 'PRO', 'Q': 'GLN', 'R': 'ARG',
    'S': 'SER', 'T': 'THR', 'U': 'SEC', 'V': 'VAL', 'W': 'TRP', 'X': '<Any>',
    'Y': 'TYR', 'Z': 'GLX',
}


def check_ext(ext: str) -> str:
    if ext not in {'.cif', '.pdb', '.ent'}:
        raise ValueError('ext must in {.cif, .pdb, .ent}')
    if ext == '.ent':
        ext = '.pdb'
    return ext


def is_empty(s: str) -> bool:
    return s in ('?', '.')


def norm_cif_empty_value(s: str) -> str:
    """Normalize mmCIF empty value ('?' or '.')."""
    if s == '?' or s == '.':
        return '.'
    return s


def aa_3to1(res_name: str) -> str:
    code = PDBData.protein_letters_3to1.get(res_name, 'X')
    if len(code) != 1:
        return 'X'
    return code


def aa_1to3(res_code: str) -> str:
    res_name = _1TO3_TABLE.get(res_code, '<Unknown>')
    return res_name
