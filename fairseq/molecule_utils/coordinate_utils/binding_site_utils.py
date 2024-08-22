#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Binding site functions."""

import logging
from typing import Union, Iterable, Sequence

import numpy as np
from Bio import PDB
from Bio.PDB.Structure import Structure as PdbStructure
from Bio.PDB.PDBIO import PDBIO
from scipy.spatial.distance import cdist

from ..database.af2_mmcif_parsing import AF2MmcifObject
from ..database import mmcif_utils as mu
from . import atom_positions as apos


class NearSelect(PDB.Select):
    def __init__(self, coordinate, threshold, chain_id=None):
        super().__init__()
        self.coordinate = coordinate
        self.threshold = threshold
        self.chain_id = chain_id

    def accept_chain(self, chain):
        if self.chain_id is not None and self.chain_id != chain.id:
            return 0
        return 1

    def accept_residue(self, residue):
        if self.coordinate is None:
            return 1
        res_coordinate = apos.get_residue_atom_coordinates(residue, only_aa=False)
        if res_coordinate.size == 0:
            return 0
        if np.min(cdist(res_coordinate, self.coordinate)) <= self.threshold:
            return 1
        return 0


class SingleResidueSelect(PDB.Select):
    def __init__(self, residue_index):
        super().__init__()
        self.residue_index = residue_index

    def accept_residue(self, residue):
        if self.residue_index == residue.id:
            return 1
        return 0


def residues_near_ligand_pdb(
        input_structure_filename,
        input_ligand_filename,
        output_structure_filename,
        threshold: float = 5.0,
):
    """Filter input PDB structure that near to the ligand, and write to the output structure.

    This function should be used with the script `split_complex.py`.

    Args:
        input_structure_filename: Protein without ligands (output of split_complex.py)
        input_ligand_filename: Ligand (output of split_complex.py)
        output_structure_filename:
        threshold:

    Returns:

    """

    parser = PDB.PDBParser()
    with open(input_structure_filename, 'r', encoding='utf-8') as f_in:
        structure = parser.get_structure('none-protein', f_in)

    parser = PDB.PDBParser()
    with open(input_ligand_filename, 'r', encoding='utf-8') as f_ligand:
        ligand = parser.get_structure('none-ligand', f_ligand)

    ligand_residues = list(ligand.get_residues())
    if len(ligand_residues) != 1:
        logging.warning('Ligand contains > 1 residues, only use the first')
    ligand_residue = ligand_residues[0]
    ligand_coordinate = apos.get_residue_atom_coordinates(ligand_residue, only_aa=False)
    if ligand_coordinate.size == 0:
        logging.warning('Cannot parse ligand coordinates, return the original input file.')
        ligand_coordinate = None
    select = NearSelect(ligand_coordinate, threshold)

    io = PDBIO()
    io.set_structure(structure)
    io.save(str(output_structure_filename), select=select)


def residues_near_ligand_adapt_3d_generative_SBDD(
        mmcif_object: AF2MmcifObject,
        output_structure_filename,
        output_ligand_filename,
        auth_chain_id: str, ligand_res_id: int,
        ligand_id: str, ligand_insertion_code: str = '.',
        threshold: float = 5.0,
):
    """Save residues near ligand and the ligand itself into protein.pdb and ligand.sdf respectively.

    Compatible with 3D-Generative-SBDD (Luo et al.)

    Args:
        mmcif_object:
        output_structure_filename:
        output_ligand_filename:
        auth_chain_id: If set to None, will save near ligands on all chains.
        ligand_res_id:
        ligand_id:
        ligand_insertion_code:
        threshold:

    Returns:

    """

    wrapped_structure = PdbStructure(id=mmcif_object.file_id)
    wrapped_structure.add(mmcif_object.structure)
    _index_insertion_code = ' ' if ligand_insertion_code == '.' else ligand_insertion_code
    residue_index = ('H_' + ligand_id, ligand_res_id, _index_insertion_code)

    io = PDBIO()
    io.set_structure(wrapped_structure)

    try:
        io.save(str(output_ligand_filename), select=SingleResidueSelect(residue_index))
    except TypeError as e:
        if str(e) == '%c requires int or char':
            raise RuntimeError(f'Chain id {auth_chain_id} is not a single char') from e
        else:
            raise

    chain = mu.get_bio_chain(mmcif_object, auth_chain_id)
    try:
        ligand_residue = chain[residue_index]
    except KeyError as e:
        raise RuntimeError(
            f'Cannot find residue {residue_index} in {mmcif_object.file_id} chain {auth_chain_id}.') from e
    if ligand_residue.resname != ligand_id:
        raise RuntimeError(f'Mismatch ligand: expected {ligand_id}, but got {ligand_residue.resname}.')
    ligand_coordinate = apos.get_residue_atom_coordinates(ligand_residue, only_aa=False)

    io = PDBIO()
    io.set_structure(wrapped_structure)
    io.save(str(output_structure_filename), select=NearSelect(ligand_coordinate, threshold, chain_id=auth_chain_id))


def _truncate_by_max_len(mask, max_len):
    if max_len is not None:
        return mask[:max_len, ...]
    return mask


class InvalidCoordinates(RuntimeError):
    pass


def residues_near_positions(
        mmcif_object: AF2MmcifObject, auth_chain_id: str,
        coordinates: Sequence[np.ndarray],
        threshold: float = 5.0, max_len: int = None, allow_zeros: bool = True,
):
    """Select residues that near the positions.

    Args:
        mmcif_object:
        auth_chain_id:
        coordinates: Sequence of numpy arrays of (N_atoms, 3) or (3,)
            (N_atoms, 3): atom coordinates of molecules
            (3,): a single 3D point (center, etc.)
        threshold:
        max_len:
        allow_zeros:

    Returns:

    """
    num_res = len(mmcif_object.chain_to_seqres[auth_chain_id])
    valid_coordinates = []
    for coordinate in coordinates:
        if coordinate.size == 0:
            continue
        if coordinate.ndim == 1:
            coordinate = coordinate[None]
        valid_coordinates.append(coordinate)
    if not valid_coordinates:
        if allow_zeros:
            return _truncate_by_max_len(np.zeros((num_res,), dtype=np.int32), max_len)
        else:
            raise InvalidCoordinates('No valid coordinates')

    residue_iterator = mu.bio_residue_iterator(mmcif_object, auth_chain_id)

    mask = []
    res_ids = []
    for res_at_position, res in residue_iterator:
        if res is None:
            mask.append(0)
            continue
        res_coord = apos.get_residue_atom_coordinates(res)
        if res_coord.size == 0:
            mask.append(0)
            continue
        if any(np.min(cdist(res_coord, coordinate)) <= threshold for coordinate in valid_coordinates):
            mask.append(1)
            res_ids.append(res_at_position.position.residue_number)
        else:
            mask.append(0)
    return _truncate_by_max_len(np.asarray(mask, dtype=np.int32), max_len), np.asarray(res_ids, dtype=np.int32)


def residues_near_residue(
        mmcif_object: AF2MmcifObject, auth_chain_id: str, res_id: int,
        threshold: float = 5.0, max_len: int = None, allow_zeros: bool = True,
        query_chain_id: str = None, extra_information: dict = None,
) -> np.ndarray:
    """Select residues that near to the residue.

    Args:
        mmcif_object:
        auth_chain_id:
        res_id: Index to the center residue.
        threshold:
        max_len:
        allow_zeros:
        query_chain_id: Query chain ID, default to auth_chain_id.
        extra_information (dict): Extra information, like center residue object itself.

    Returns:
        A mask array in shape of (N,) of np.int32. 0/1 means this residue is skipped / accepted respectively.
    """
    if query_chain_id is None:
        query_chain_id = auth_chain_id
    num_res = len(mmcif_object.chain_to_seqres[auth_chain_id])
    res_at_position, center_residue = mu.get_bio_residue(mmcif_object, auth_chain_id, res_id)
    if center_residue is None:
        error_msg = f'Cannot find residue {res_id} in {mmcif_object.file_id} chain {auth_chain_id}.'
        if allow_zeros:
            logging.warning(error_msg)
            return _truncate_by_max_len(np.zeros((num_res,), dtype=np.int32), max_len)
        else:
            raise InvalidCoordinates(error_msg)
    coordinate = apos.get_residue_atom_coordinates(center_residue, only_aa=False)

    if extra_information is not None:
        extra_information.update({
            'center_residue': center_residue,
        })

    return residues_near_positions(
        mmcif_object, query_chain_id, coordinates=[coordinate], threshold=threshold, max_len=max_len,
        allow_zeros=allow_zeros,
    )


def residues_near_ligand(
        mmcif_object: AF2MmcifObject, auth_chain_id: str, ligand_res_id: int,
        ligand_id: str, ligand_insertion_code: str = '.',
        threshold: float = 5.0, max_len: int = None, allow_zeros: bool = True,
        query_chain_id: str = None, extra_information: dict = None,
) -> np.ndarray:
    """Select residues that near to the ligand.

    Args:
        mmcif_object:
        auth_chain_id: ligand chain ID.
        ligand_res_id:
        ligand_id: Ligand CCD ID.
        ligand_insertion_code: Ligand insertion code.
        threshold:
        max_len: If not None, will truncate output to max length.
        allow_zeros:
        query_chain_id: Query chain ID, default to auth_chain_id.
        extra_information (dict): Extra information, like ligand residue object itself.

    Returns:
        A mask array in shape of (N,) of np.int32. 0/1 means this residue is skipped / accepted respectively.
    """
    if query_chain_id is None:
        query_chain_id = auth_chain_id
    num_res = len(mmcif_object.chain_to_seqres[auth_chain_id])

    # Get ligand residue and coordinate.
    chain = mu.get_bio_chain(mmcif_object, auth_chain_id)
    _index_insertion_code = ' ' if ligand_insertion_code == '.' else ligand_insertion_code
    residue_index = ('H_' + ligand_id, ligand_res_id, _index_insertion_code)
    try:
        ligand_residue = chain[residue_index]
    except KeyError:
        error_msg = f'Cannot find residue {residue_index} in {mmcif_object.file_id} chain {auth_chain_id}.'
        if allow_zeros:
            logging.warning(error_msg)
            return _truncate_by_max_len(np.zeros((num_res,), dtype=np.int32), max_len)
        else:
            raise InvalidCoordinates(error_msg)
    if ligand_residue.resname != ligand_id:
        raise RuntimeError(f'Mismatch ligand: expected {ligand_id}, but got {ligand_residue.resname}.')
    ligand_coordinate = apos.get_residue_atom_coordinates(ligand_residue, only_aa=False)

    if extra_information is not None:
        extra_information.update({
            'ligand_residue': ligand_residue,
        })

    return residues_near_positions(
        mmcif_object, query_chain_id, coordinates=[ligand_coordinate], threshold=threshold, max_len=max_len,
    )


def residues_near_site(
        mmcif_object: AF2MmcifObject, auth_chain_id: str,
        site_ids: Union[None, str, Iterable[str]] = None, threshold: float = 5.0, max_len: int = None,
        allow_zeros: bool = True,
) -> np.ndarray:
    """Select residues that near to binding sites.

    Args:
        mmcif_object:
        auth_chain_id (str): Author chain ID.
        site_ids (Union[None, str, Iterable[str]]):
            If not set, will mark residues that near to *ALL* binding sites.
            If given, will only mark residues that near to the *PROVIDED* binding sites.
        threshold:
        max_len: If not None, will truncate output to max length.
        allow_zeros:

    Returns:
        A mask array in shape of (N,) of np.int32. 0/1 means this residue is skipped / accepted respectively.
    """

    num_res = len(mmcif_object.chain_to_seqres[auth_chain_id])

    # Get ligand residues and coordinates.
    binding_sites = mu.get_binding_sites(mmcif_object, auth_id=False)
    if site_ids is None:
        site_ids = binding_sites.keys()
    elif isinstance(site_ids, str):
        site_ids = [site_ids]
    binding_sites = {site_name: binding_sites[site_name] for site_name in site_ids}
    if not binding_sites:
        if allow_zeros:
            return _truncate_by_max_len(np.zeros((num_res,), dtype=np.int32), max_len)
        else:
            raise InvalidCoordinates('No binding sites')

    binding_residues = []
    for site_name, site in binding_sites.items():
        for binding_position in site.binding_positions:     # type: mu.SingleBindingPosition
            _, binding_residue = mu.get_bio_residue(mmcif_object, binding_position.chain_id, binding_position.res_id)
            binding_residues.append(binding_residue)

    site_coordinates = [apos.get_residue_atom_coordinates(br) for br in binding_residues]

    return residues_near_positions(
        mmcif_object, auth_chain_id, coordinates=site_coordinates, threshold=threshold, max_len=max_len,
    )
