#! /usr/bin/python
# -*- coding: utf-8 -*-

"""PDB helper datasets (PdbSws and PdbCCD)."""

import csv
import logging
import pickle as pkl
from collections import namedtuple
from pathlib import Path
from typing import Dict, List, Tuple

import requests

from . import caching_utils
from .. import config

CcdInfo = namedtuple('CcdInfo', 'id name smiles formula formula_weight inchi')


def get_pdb_ccd_info(pdb_ccd_path: Path = None) -> Dict[str, CcdInfo]:
    """Get PDB-CCD information file (basic_info.csv).

    Args:
        pdb_ccd_path:

    Returns:

    """

    cache = caching_utils.get_cache('pdb_ccd', None)
    if cache is not None:
        return cache

    basic_info_file = config.pdb_ccd_basic_info_file(pdb_ccd_path)
    if not basic_info_file.exists():
        error_msg = (
            f'PDB CCD dataset file not found. '
            f'Please copy it from /blob/v-yaf/fairseq-data-bin/tgt2drug/orig/PdbCCD/basic_info.csv '
            f'to {basic_info_file}.')
        raise RuntimeError(error_msg)

    cache = {}
    with basic_info_file.open('r', encoding='utf-8') as f_basic_info:
        reader = csv.reader(f_basic_info)
        next(reader)
        for row in reader:
            row = list(row)
            row[4] = float(row[4])
            info = CcdInfo(*row)
            cache[row[0]] = info
    caching_utils.add_cache('pdb_ccd', cache)
    return cache


def get_pdb_ccd_info_inchi_key(pdb_ccd_path: Path = None) -> Dict[str, CcdInfo]:
    """Get PDB-CCD information file, use InChI as keys.

    Warnings
    --------

    Some different CCD entries have the same InChI.
    """

    cache = caching_utils.get_cache('pdb_ccd_inchi', None)
    if cache is not None:
        return cache

    pdb_ccd_info = get_pdb_ccd_info(pdb_ccd_path)
    cache = {ccd_info.inchi: ccd_info for ccd_info in pdb_ccd_info.values()}
    return cache


PDB_CHAIN_ID = Tuple[str, str]
PDB_CHAIN_IDS = List[PDB_CHAIN_ID]
U2P_MAPPING = Dict[str, PDB_CHAIN_IDS]


def _build_u2p_mapping(pdb_sws_path) -> U2P_MAPPING:
    with config.pdb_sws_mapping_file(pdb_sws_path).open('r', encoding='utf-8') as f_raw:
        mapping = {}
        for line in f_raw:
            words = line.strip().split()
            if len(words) == 2:     # Some special line: '1c04 E       '
                words.append('?')
            pdb_id, chain_id, uniprot_id = words
            mapping.setdefault(uniprot_id.lower(), []).append((pdb_id.lower(), chain_id))
        return mapping


def get_pdb_sws_u2p_mapping(pdb_sws_path: Path = None) -> U2P_MAPPING:
    cache = caching_utils.get_cache('pdb_sws_u2p', None)
    if cache is not None:
        return cache
    if config.u2p_mapping_pickle_file(pdb_sws_path).exists():
        with config.u2p_mapping_pickle_file(pdb_sws_path).open('rb') as f_pkl:
            cache = pkl.load(f_pkl)
    else:
        if config.pdb_sws_mapping_file(pdb_sws_path).exists():
            cache = _build_u2p_mapping(pdb_sws_path)
        else:
            raise FileNotFoundError('UniProt-PDB mapping file not found.')
        with config.u2p_mapping_pickle_file(pdb_sws_path).open('wb') as f_pkl:
            pkl.dump(cache, f_pkl)
    caching_utils.add_cache('pdb_sws_u2p', cache)
    return cache


def _chunked(seq, chunk_size):
    for i in range(0, len(seq), chunk_size):
        chunk = seq[i:i + chunk_size]
        if chunk:
            yield chunk


def _get_resp(uniprot_ids, mapping):
    if not uniprot_ids:
        return
    params = {
        'from': 'ACC',
        'to': 'PDB_ID',
        'format': 'tab',
        'query': ' '.join(uniprot_ids),
    }
    resp = requests.get(config.GET_UNIPROT_MAPPINGS_URL, params=params)
    if resp.status_code != 200:
        logging.warning(f'Failed to retrieve UniProt mapping: {resp}')
        return
    for row in resp.text.splitlines():
        uniprot_id, pdb_id = row.split('\t')
        uniprot_id = uniprot_id.lower()
        pdb_id = pdb_id.lower()
        if uniprot_id == 'From':
            continue
        mapping.setdefault(uniprot_id, []).append((pdb_id, '*'))


def download_latest_uniprot_u2p_mapping(
        uniprot_ids, prefix, uniprot_mappings_path: Path = None,
) -> U2P_MAPPING:
    """Download latest UniProt-PDB ID mapping from UniProt official site."""
    logging.warning(f'Downloading latest UniProt-PDB ID mapping of {prefix} from UniProt official site')
    mapping = {}
    for chunk in _chunked(uniprot_ids, chunk_size=800):
        _get_resp(chunk, mapping)
    config.save_latest_uniprot_mapping_file(mapping, prefix, to='PDB', mappings_path=uniprot_mappings_path)
    return mapping


def get_latest_uniprot_u2p_mapping(
        prefix, uniprot_mappings_path: Path = None,
) -> U2P_MAPPING:
    filename = config.latest_uniprot_mapping_file(prefix, to='PDB', mappings_path=uniprot_mappings_path)
    if filename is None:
        raise FileNotFoundError('Mapping not found, please call `download_latest_uniprot_u2p_mapping` to download it.')
    with filename.open('rb') as f:
        return pkl.load(f)


def get_pdb_ids_list() -> List[str]:
    filename = config.pdb_split_dataset_path() / 'pdb-ids-list.txt'
    with open(filename, 'r', encoding='utf-8') as f:
        result = [line.strip() for line in f]
    return result
