#! /usr/bin/python
# -*- coding: utf-8 -*-

from typing import Union, Tuple

import numpy as np
from Bio.PDB.Residue import Residue as PdbResidue
from rdkit import Chem

from ..af2_utils.common import residue_constants
from ..database.af2_mmcif_parsing import AF2MmcifObject
from ..exceptions import MultipleChainsError


# TODO: Refactor -- Move functions in `af2_utils/parse_templates.py` here.


def get_residue_atom_coordinates(
        residue: PdbResidue, allow_invalid: bool = False, dtype=np.float32, only_aa: bool = True,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Get atom coordinates array of the residue.

    Args:
        residue:
        allow_invalid: If set to True, will return *2* arrays: coordinates include invalid atoms, and a mask array.
        dtype:
        only_aa: If set to False, will not filter non-amino-acid atoms (used for ligand input).

    Returns:
        allow_invalid: tuple of (np.ndarray coordinates in shape (N_atoms, 3), np.ndarray mask in shape (N_atoms,))
        only valid: np.ndarray coordinates in shape (N_valid_atoms, 3)
    """

    atom_coord_matrix = []
    mask = []

    for atom in residue.get_atoms():
        atom_name = atom.get_name()
        if not only_aa or atom_name in residue_constants.atom_order.keys() or (
                atom_name.upper() == 'SE' and residue.get_resname() == 'MSE'):
            # Valid atom in the residue.
            atom_coord_matrix.append(atom.get_coord())
            mask.append(1.0)
        else:
            mask.append(0.0)
            if allow_invalid:
                atom_coord_matrix.append((0., 0., 0.))
    atom_coord_array = np.asarray(atom_coord_matrix, dtype=dtype)
    if allow_invalid:
        mask_array = np.asarray(mask, dtype=dtype)
        return atom_coord_array, mask_array
    else:
        return atom_coord_array


_PERIODIC_TABLE = None


def get_atom_weights(residue: PdbResidue, dtype=np.float32, only_aa: bool = True) -> np.ndarray:
    """Get atom weights."""
    global _PERIODIC_TABLE
    if _PERIODIC_TABLE is None:
        _PERIODIC_TABLE = Chem.rdchem.GetPeriodicTable()
    weights = []
    for atom in residue.get_atoms():
        atom_name = atom.get_name()
        if not only_aa or atom_name in residue_constants.atom_order.keys() or (
                atom_name.upper() == 'SE' and residue.get_resname() == 'MSE'):
            element = atom.element.capitalize()
            # [NOTE]: Just a simple fix, change Deuterium to Hydrogenium.
            if element == 'D':
                element = 'H'
            weight = _PERIODIC_TABLE.GetAtomicWeight(element)
            weights.append(weight)
    return np.asarray(weights, dtype=dtype)


def get_residue_average_position(
        residue: PdbResidue, dtype=np.float32, only_aa: bool = True,
        center_of_gravity: bool = False,
) -> np.ndarray:
    """Get average position of a PdbResidue."""
    atom_coord = get_residue_atom_coordinates(residue, allow_invalid=False, dtype=dtype, only_aa=only_aa)
    if atom_coord.size == 0:
        pos = np.zeros([3], dtype=dtype)
    else:
        if center_of_gravity:
            atom_weights = get_atom_weights(residue, dtype=dtype, only_aa=only_aa)
            assert atom_coord.shape[0] == atom_weights.shape[0]
            # FIXME: Bug version!!!
            # atom_coord = atom_coord * atom_weights[:, None] / atom_weights.sum()
            atom_coord = atom_coord * atom_weights[:, None] / atom_weights.mean()
        pos = np.mean(atom_coord, axis=0)
    return pos


def simple_get_residue_positions(
        mmcif_object: AF2MmcifObject,
        auth_chain_id: str,
        center_of_gravity: bool = False,
) -> np.ndarray:
    """Gets residue positions (simple version).

    Will average on all atoms in each residue.

    Returns:
        (N, 3) residue positions.
    """
    num_res = len(mmcif_object.chain_to_seqres[auth_chain_id])

    relevant_chains = [c for c in mmcif_object.structure.get_chains()
                       if c.id == auth_chain_id]
    if len(relevant_chains) != 1:
        raise MultipleChainsError(
            f'Expected exactly one chain in structure with id {auth_chain_id}.')
    chain = relevant_chains[0]

    all_positions = np.zeros([num_res, 3], dtype=np.float32)
    for res_index in range(num_res):
        pos = np.zeros([3], dtype=np.float32)
        res_at_position = mmcif_object.seqres_to_structure[auth_chain_id][res_index]
        if not res_at_position.is_missing:
            res = chain[(res_at_position.hetflag,
                         res_at_position.position.residue_number,
                         res_at_position.position.insertion_code)]
            pos = get_residue_average_position(res, dtype=np.float32, only_aa=True, center_of_gravity=center_of_gravity)
        all_positions[res_index] = pos
    return all_positions
