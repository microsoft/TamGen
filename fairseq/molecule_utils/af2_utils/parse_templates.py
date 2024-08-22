#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Parse templates / PDB files.

Copied from AlphaFold2 project `alphafold/data/templates.py`."""

from typing import Dict, Tuple

import numpy as np
from Bio import PDB
from Bio.PDB.Structure import Structure as PdbStructure

from .common import residue_constants
from ..database.af2_mmcif_parsing import AF2MmcifObject
from ..exceptions import MultipleChainsError, CaDistanceError


def _check_residue_distances(all_positions: np.ndarray,
                             all_positions_mask: np.ndarray,
                             max_ca_ca_distance: float):
    """Checks if the distance between unmasked neighbor residues is ok."""
    ca_position = residue_constants.atom_order['CA']
    prev_is_unmasked = False
    prev_calpha = None
    for i, (coords, mask) in enumerate(zip(all_positions, all_positions_mask)):
        this_is_unmasked = bool(mask[ca_position])
        if this_is_unmasked:
            this_calpha = coords[ca_position]
            if prev_is_unmasked:
                distance = np.linalg.norm(this_calpha - prev_calpha)
                if distance > max_ca_ca_distance:
                    raise CaDistanceError(
                        'The distance between residues %d and %d is %f > limit %f.' % (
                            i, i + 1, distance, max_ca_ca_distance))
            prev_calpha = this_calpha
        prev_is_unmasked = this_is_unmasked


def get_atom_positions(
        mmcif_object: AF2MmcifObject,
        auth_chain_id: str,
        max_ca_ca_distance: float = 150.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Gets atom positions and mask from a list of Biopython Residues."""
    num_res = len(mmcif_object.chain_to_seqres[auth_chain_id])

    relevant_chains = [c for c in mmcif_object.structure.get_chains()
                       if c.id == auth_chain_id]
    if len(relevant_chains) != 1:
        raise MultipleChainsError(
            f'Expected exactly one chain in structure with id {auth_chain_id}.')
    chain = relevant_chains[0]

    all_positions = np.zeros([num_res, residue_constants.atom_type_num, 3])
    all_positions_mask = np.zeros([num_res, residue_constants.atom_type_num],
                                  dtype=np.int64)
    for res_index in range(num_res):
        pos = np.zeros([residue_constants.atom_type_num, 3], dtype=np.float32)
        mask = np.zeros([residue_constants.atom_type_num], dtype=np.float32)
        res_at_position = mmcif_object.seqres_to_structure[auth_chain_id][res_index]
        if not res_at_position.is_missing:
            res = chain[(res_at_position.hetflag,
                         res_at_position.position.residue_number,
                         res_at_position.position.insertion_code)]
            for atom in res.get_atoms():
                atom_name = atom.get_name()
                x, y, z = atom.get_coord()
                if atom_name in residue_constants.atom_order.keys():
                    pos[residue_constants.atom_order[atom_name]] = [x, y, z]
                    mask[residue_constants.atom_order[atom_name]] = 1.0
                elif atom_name.upper() == 'SE' and res.get_resname() == 'MSE':
                    # Put the coordinates of the selenium atom in the sulphur column.
                    pos[residue_constants.atom_order['SD']] = [x, y, z]
                    mask[residue_constants.atom_order['SD']] = 1.0

        all_positions[res_index] = pos
        all_positions_mask[res_index] = mask
    _check_residue_distances(
        all_positions, all_positions_mask, max_ca_ca_distance)
    return all_positions, all_positions_mask
