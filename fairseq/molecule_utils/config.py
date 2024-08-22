#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Molecule configuration."""

import logging as _logging
import pickle as _pkl
import os as _os
from datetime import date as _date
from pathlib import Path
from socket import gethostname as _hostname

DATASET_ROOT = Path(r'./database')


def root():
    return DATASET_ROOT


def dataset_path(path: str):
    """Get dataset path."""
    return DATASET_ROOT / path


def pdb_cache_path():
    return DATASET_ROOT / 'PDBlib'


def split_pdb_cache_path():
    """Split target-ligand complex in PDB files."""
    return DATASET_ROOT / 'SplitPdb'


def uniprot_fasta_cache_path():
    return DATASET_ROOT / 'UniProtFastaCache'


def uniprot_mappings_path():
    return DATASET_ROOT / 'UniProtMappingsCache'


def pdb_sws_path():
    return DATASET_ROOT / 'PdbSws'


def pdb_ccd_path():
    return DATASET_ROOT / 'PdbCCD'


def pdb_split_dataset_path():
    return DATASET_ROOT / 'PdbSplit'


def set_dataset_root(root: Path):
    """Setup dataset roots."""
    global DATASET_ROOT
    DATASET_ROOT = root

    if DATASET_ROOT.exists():
        pdb_cache_path().mkdir(exist_ok=True)
        split_pdb_cache_path().mkdir(exist_ok=True)
        uniprot_fasta_cache_path().mkdir(exist_ok=True)
        uniprot_mappings_path().mkdir(exist_ok=True)
        pdb_sws_path().mkdir(exist_ok=True)
        pdb_ccd_path().mkdir(exist_ok=True)

set_dataset_root(DATASET_ROOT)


def latest_uniprot_mapping_file(prefix, to, mappings_path=None):
    """UniProt mapping file format: <prefix>-<to>-<iso-format-date>.pkl.

    Example: drugbank-PDB-2021-12-17.pkl"""

    if mappings_path is None:
        mappings_path = uniprot_mappings_path()

    max_date, out_filename = _date.min, None
    for filename in mappings_path.iterdir():
        stem, suffix = filename.stem, filename.suffix
        if not stem.startswith(f'{prefix}-{to}-') or suffix != '.pkl':
            continue
        date = _date.fromisoformat(stem[len(prefix) + len(to) + 2:])
        if date > max_date:
            max_date = date
            out_filename = filename
    return out_filename


def save_latest_uniprot_mapping_file(mapping, prefix, to, mappings_path=None):
    if mappings_path is None:
        mappings_path = uniprot_mappings_path()
    iso_today = _date.today().isoformat()
    filename = mappings_path / f'{prefix}-{to}-{iso_today}.pkl'
    with filename.open('wb') as f:
        _pkl.dump(mapping, f)


def pdb_sws_mapping_file(pdb_sws_root=None):
    if pdb_sws_root is None:
        pdb_sws_root = pdb_sws_path()
    return pdb_sws_root / 'mapping.txt'


def pdb_sws_full_mapping_file(pdb_sws_root=None):
    if pdb_sws_root is None:
        pdb_sws_root = pdb_sws_path()
    return pdb_sws_root / 'mapping-full.txt'


def u2p_mapping_pickle_file(pdb_sws_root=None):
    if pdb_sws_root is None:
        pdb_sws_root = pdb_sws_path()
    return pdb_sws_root / 'mapping-u2p.pkl'


def p2u_mapping_pickle_file(pdb_sws_root=None):
    if pdb_sws_root is None:
        pdb_sws_root = pdb_sws_path()
    return pdb_sws_root / 'mapping-p2u.pkl'


def pdb_all_ccd_ids_file(pdb_ccd_root=None):
    if pdb_ccd_root is None:
        pdb_ccd_root = pdb_ccd_path()
    return pdb_ccd_root / 'all_ccd_ids.txt'


def pdb_ccd_basic_info_file(pdb_ccd_root=None):
    if pdb_ccd_root is None:
        pdb_ccd_root = pdb_ccd_path()
    return pdb_ccd_root / 'basic_info.csv'


# Query URLs.
GET_PDB_URL = 'https://files.rcsb.org/view'
GET_UNIPROT_FASTA_URL = 'https://www.uniprot.org/uniprot'
GET_UNIPROT_MAPPINGS_URL = 'https://www.uniprot.org/uploadlists/'


# Constants.
SKIP_INORGANIC_LIGANDS = {'HOH'}
LIGAND_MAX_FORMULA_WEIGHT = 10000.0
