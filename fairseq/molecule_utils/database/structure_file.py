#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Get structure file (PDB, mmCIF)."""

import logging
import multiprocessing
from io import TextIOBase
from pathlib import Path
from typing import Optional

import requests
from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.Structure import Structure as PdbStructure

from . import caching_utils
from .af2_mmcif_parsing import parse as parse_af2_mmcif, AF2MmcifObject
from .common_utils import check_ext
from .. import config


def download_structure_file(
        pdb_id: str, pdb_cache_path: Path = None, ext: str = '.cif', mp_lock: 'multiprocessing.synchronize.Lock' = None,
) -> Optional[Path]:
    """Download structure file from the RCSB PDB website.

    Args:
        pdb_id:
        pdb_cache_path:
        ext (str): Extension to download
        mp_lock: Optional multiprocessing lock for file exist check.

    Returns:
        Downloaded structure file path (if exist) or None if failed.

    Notes:
        This function is not thread safe when multiple threads / processes try to download same file.
    """
    assert ext in {'.cif', '.pdb'}

    if pdb_cache_path is None:
        pdb_cache_path = config.pdb_cache_path()

    pdb_id = pdb_id.lower()
    dest_filename = pdb_cache_path / f'{pdb_id}{ext}'

    if mp_lock is None:
        if dest_filename.exists():
            return dest_filename
    else:
        with mp_lock:
            if dest_filename.exists():
                return dest_filename

    url = config.GET_PDB_URL + f'/{pdb_id}{ext}'
    logging.info(f'Downloading {pdb_id}{ext} from {url} to {dest_filename}')
    resp = requests.get(url)
    if resp.status_code != 200:
        logging.warning(f'Failed to retrieve {pdb_id}{ext}.')
        return None
    else:
        with dest_filename.open('w', encoding='utf-8') as f:
            f.write(resp.text)
        return dest_filename


caching_utils.add_cache('pdb', caching_utils.LRUCache(
    get_fn=None, max_size=0, description='PDB structure cache',
))
caching_utils.add_cache('cif', caching_utils.LRUCache(
    get_fn=None, max_size=0, description='mmCIF structure cache',
))


def get_pdb_structure(
        pdb_id: str, ext: str = '.cif', mmcif_add_dict: bool = True,
        pdb_cache_path: Path = None,
) -> PdbStructure:
    """Get PDB structure of given PDB ID from mmCIF or PDB file.

    Args:
        pdb_id:
        ext (str): Structure format ext to get, can be '.cif' or '.pdb',
        mmcif_add_dict (bool): Add a MMCIF2Dict object into `structure.mmcif_dict` attribute.
        pdb_cache_path:

    Returns:
        PdbStructure: Structure of the mmCIF / PDB file.

    Raises:
        FileNotFoundError if it cannot download structure files.
        AF2MmCIFParseError if failed to parse mmCIF file.
    """
    ext = check_ext(ext)
    pdb_id = pdb_id.lower()

    dest_filename = download_structure_file(pdb_id, pdb_cache_path, ext=ext)
    if dest_filename is None:
        raise FileNotFoundError(f'Structure file of {pdb_id} not found.')

    cache = caching_utils.get_cache(ext[1:])

    def _mmcif_get_fn(_):
        parser = MMCIFParser(QUIET=True)
        with dest_filename.open('r', encoding='utf-8') as cif_f:
            structure = parser.get_structure('none', cif_f)
        if mmcif_add_dict:
            mmcif_dict = getattr(parser, '_mmcif_dict')
            # Ensure all values are lists, even if singletons.
            for key, value in mmcif_dict.items():
                if not isinstance(value, list):
                    mmcif_dict[key] = [value]
            structure.mmcif_dict = mmcif_dict
        return structure

    def _pdb_get_fn(_):
        parser = PDBParser(QUIET=True)
        with dest_filename.open('r', encoding='utf-8') as pdb_f:
            return parser.get_structure('none', pdb_f)

    structure = cache.get_by_fn(pdb_id, get_fn=(_mmcif_get_fn if ext == '.cif' else _pdb_get_fn))
    return structure


# Caching.
caching_utils.add_cache('af2_mmcif', caching_utils.LRUCache(
    get_fn=None, max_size=0, description='AF2 mmCIF file cache',
))


def get_af2_mmcif_object(
        pdb_id: str, pdb_cache_path: Path = None, mp_lock: 'multiprocessing.synchronize.Lock' = None,
) -> AF2MmcifObject:
    """Get AlphaFold2-compatible mmCIF object of given PDB ID from mmCIF file.

    Args:
        pdb_id:
        pdb_cache_path:
        mp_lock:  Optional multiprocessing lock to avoid downloading files twice.

    Returns:
        AlphaFold2-compatible AF2MmcifObject of the file.

    Raises:
        FileNotFoundError if it cannot download structure files.
        AF2MmCIFParseError if failed to parse mmCIF file.
    """
    pdb_id = pdb_id.lower()
    dest_filename = download_structure_file(pdb_id, pdb_cache_path, ext='.cif', mp_lock=mp_lock)
    if dest_filename is None:
        raise FileNotFoundError(f'Structure file of {pdb_id} not found.')

    cache = caching_utils.get_cache('af2_mmcif')

    def _get_fn(_pdb_id):
        with dest_filename.open('r', encoding='utf-8') as cif_f:  # type: TextIOBase
            return parse_af2_mmcif(file_id=pdb_id, mmcif_handle=cif_f, catch_all_errors=False)

    parsing_result = cache.get_by_fn(pdb_id, get_fn=_get_fn)
    return parsing_result.mmcif_object
