#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Handle mappings between different databases."""

import logging
import random
from enum import Enum
from pathlib import Path
from typing import Union, Dict, Optional

from editdistance import eval as eval_ed

from .af2_mmcif_parsing import AF2MmcifObject, AF2MmCIFParseError
from .pdb_helper_datasets import PDB_CHAIN_ID, PDB_CHAIN_IDS, U2P_MAPPING
from .structure_file import get_af2_mmcif_object
from .. import config


class UniProt2PdbChoosePolicy(Enum):
    DEFAULT = 0
    FIRST = 1
    MIN_RESOLUTION = 2
    RANDOM = 3


def uniprot_to_pdb(
        uniprot_id: str, mapping: U2P_MAPPING,
) -> PDB_CHAIN_IDS:
    """Get corresponding list of (PDB ID, chain ID) tuples of the given UniProt ID.

    We use PdbSws (http://www.bioinf.org.uk/servers/pdbsws) to query the mapping.
    We download the mapping file from http://www.bioinf.org.uk/servers/pdbsws/pdbsws_chain.txt.gz at **2021.12.04**,
    and convert the text file into pickle file for faster query.

    PdbSws paper link: https://academic.oup.com/bioinformatics/article/21/23/4297/194996

    Args:
        uniprot_id: UniProt ID.
        mapping: Uniprot-PDB ID mapping dict.

    Returns:
        List of all related (PDB ID, chain ID) tuples.
    """
    uniprot_id = uniprot_id.lower()
    return mapping.get(uniprot_id, [])


def _preprocess_pdb_chain_ids(
        pdb_chain_id_list: PDB_CHAIN_IDS, all_structures: Dict[str, AF2MmcifObject]) -> PDB_CHAIN_IDS:
    """Preprocess PDB-chain IDs.

    1. Remove PDB IDs that not exist in all_structures (fail to parse mmCIF file)
    2. Expand wildcard chain ID (e.g. [('2rbg', '*')] => [('2rbg', 'A'), ('2rbg', 'B')]
    """
    result = []
    for pdb_chain_id in pdb_chain_id_list:
        pdb_id, wildcard_chain_id = pdb_chain_id
        if pdb_id not in all_structures:
            continue
        if wildcard_chain_id == '*':
            result.extend((pdb_id, c) for c in all_structures[pdb_id].chain_to_seqres)
        else:
            if wildcard_chain_id in all_structures[pdb_id].chain_to_seqres:
                result.append(pdb_chain_id)
    return result


def _min_resolution_policy(
        pdb_chain_id_list: PDB_CHAIN_IDS,
        all_structures: Dict[str, AF2MmcifObject],
) -> PDB_CHAIN_IDS:
    pdb_chain_id_list = _preprocess_pdb_chain_ids(pdb_chain_id_list, all_structures)

    # Sort key: resolution.
    sort_keys = []
    for index, (pdb_id, chain_id) in enumerate(pdb_chain_id_list):
        resolution = all_structures[pdb_id].header['resolution']
        # NMR & Distillation have resolution == 0. Default value is also 0.
        if resolution < 0.1:
            resolution = 1e9
        sort_keys.append((resolution, index))
    sort_keys.sort()
    return [pdb_chain_id_list[sort_key[-1]] for sort_key in sort_keys]


def _default_policy(
        uniprot_id: str,
        pdb_chain_id_list: PDB_CHAIN_IDS,
        all_structures: Dict[str, AF2MmcifObject],
        ref_sequence: str = None,
) -> PDB_CHAIN_IDS:
    if ref_sequence is None:
        # If no reference sequence, use min resolution directly.
        return _min_resolution_policy(pdb_chain_id_list, all_structures)

    pdb_chain_id_list = _preprocess_pdb_chain_ids(pdb_chain_id_list, all_structures)

    # Sort key: edit distance, resolution.
    sort_keys = []
    for index, (pdb_id, chain_id) in enumerate(pdb_chain_id_list):
        edit_distance = eval_ed(ref_sequence, all_structures[pdb_id].chain_to_seqres[chain_id])
        resolution = all_structures[pdb_id].header['resolution']
        # NMR & Distillation have resolution == 0. Default value is also 0.
        if resolution < 0.1:
            resolution = 1e9
        sort_keys.append((edit_distance, resolution, index))
    sort_keys.sort()
    return [pdb_chain_id_list[sort_key[-1]] for sort_key in sort_keys]


def uniprot_to_best_pdb(
        uniprot_id: str,
        mapping: U2P_MAPPING,
        ref_sequence: str = None,
        policy: Union[UniProt2PdbChoosePolicy, str] = UniProt2PdbChoosePolicy.DEFAULT,
        pdb_cache_path: Path = None,
        return_all: bool = False,
) -> Union[None, PDB_CHAIN_ID, PDB_CHAIN_IDS]:
    """Choose the best (PDB ID, chain ID) tuple from a candidate sequence for the given UniProt ID.

    Args:
        uniprot_id:
        mapping: Uniprot-PDB ID mapping dict.
        ref_sequence: Reference FASTA sequence.
        policy: The selection policy.
        pdb_cache_path:
        return_all: If set to True, will return all candidates, ordered by policy.

    Returns:
        The chosen best (PDB ID, chain ID) tuple, or chain list if return_all, or None if no candidates.

    Notes:
        Internally use AF2MmcifObject.
    """
    uniprot_id = uniprot_id.lower()
    pdb_chain_id_list = uniprot_to_pdb(uniprot_id, mapping=mapping)
    if not pdb_chain_id_list:
        return None

    if isinstance(policy, str):
        policy = UniProt2PdbChoosePolicy[policy]

    if pdb_cache_path is None:
        pdb_cache_path = config.pdb_cache_path()

    def _get_all_structures() -> Dict[str, AF2MmcifObject]:
        all_pdb_ids = {pc[0] for pc in pdb_chain_id_list}
        _all_structures = {}
        for p in all_pdb_ids:
            try:
                mmcif_object = get_af2_mmcif_object(p, pdb_cache_path)
                _all_structures[p] = mmcif_object
            except AF2MmCIFParseError as e:
                logging.warning(f'{e}. cause: {e.__cause__!r}')
        return _all_structures

    all_structures = _get_all_structures()

    if policy == UniProt2PdbChoosePolicy.FIRST:
        candidates = _preprocess_pdb_chain_ids(pdb_chain_id_list, all_structures)
    elif policy == UniProt2PdbChoosePolicy.RANDOM:
        candidates = _preprocess_pdb_chain_ids(pdb_chain_id_list, all_structures)
    elif policy == UniProt2PdbChoosePolicy.DEFAULT:
        candidates = _default_policy(uniprot_id, pdb_chain_id_list, all_structures, ref_sequence)
    elif policy == UniProt2PdbChoosePolicy.MIN_RESOLUTION:
        candidates = _min_resolution_policy(pdb_chain_id_list, all_structures)
    else:
        raise ValueError(f'Unknown policy {policy}.')
    if not candidates:
        return None
    if return_all:
        return candidates
    else:
        return candidates[0]

