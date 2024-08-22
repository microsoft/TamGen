#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Run docking on given target-ligand pair."""

import logging
import tempfile
from pathlib import Path
from typing import Optional, MutableMapping, Tuple

from .smiles_utils import smi2pdb
from .. import config
from ..database.split_complex import split_pdb_complex_paths
from ..external_tools.autodock_smina import AutoDockSmina, SminaError

_DOCKING_CACHE_SENTINEL = object()


def docking(
    pdb_id: str,
    ligand_smiles: str,
    *,
    pdb_path: Path = None,
    output_complex_path: Path = None,
    smina_bin_path: Path = None,
    split_cache_path: Path = None,
    pdb_cache_path: Path = None,
    ccd_cache_path: Path = None,
    docking_result_cache: MutableMapping = None,
    box_center: Tuple[float, float, float] = None,
    box_size: Tuple[float, float, float] = None,
) -> Optional[float]:
    """Docking for one PDB-ID and ligand SMILES.

    Args:
        pdb_id:
        ligand_smiles:
        pdb_path: (Optional) user provided PDB file instead of PDB ID.
        output_complex_path: (Optional) output receptor-ligand complex path.
        smina_bin_path:
        split_cache_path:
        pdb_cache_path:
        ccd_cache_path:
        docking_result_cache:
        box_center:

    Returns:

    """
    pdb_id = pdb_id.lower()
    if box_center is not None:
        box_center = (box_center[0], box_center[1], box_center[2])
    if pdb_path is not None:
        # Disable docking result cache when running on user PDB files.
        docking_result_cache = None
        raise NotImplementedError('pdb_path is not implemented now.')

    if docking_result_cache is not None:
        affinity = docking_result_cache.get((pdb_id, ligand_smiles), _DOCKING_CACHE_SENTINEL)
        if affinity is _DOCKING_CACHE_SENTINEL:
            affinity = docking_result_cache.get(
                (pdb_id, ligand_smiles, box_center),
                _DOCKING_CACHE_SENTINEL)
        if affinity is not _DOCKING_CACHE_SENTINEL:
            logging.info('Get docking result from cache')
            return affinity

    smina = AutoDockSmina(binary_path=smina_bin_path)
    if not smina.check_binary():
        raise RuntimeError('Cannot find AutoDock-smina executable.')

    if split_cache_path is None:
        split_cache_path = config.split_pdb_cache_path()
    if pdb_cache_path is None:
        pdb_cache_path = config.pdb_cache_path()
    if ccd_cache_path is None:
        ccd_cache_path = config.pdb_ccd_path()

    try:
        split_result = split_pdb_complex_paths(
            pdb_id, split_cache_path=split_cache_path, pdb_cache_path=pdb_cache_path, ccd_cache_path=ccd_cache_path,
        )
    except RuntimeError as e:
        logging.warning(e)
        return None
    receptor_filename = split_result.target_filename
    if receptor_filename is None:
        logging.warning(f'Cannot find target file of {pdb_id}, skip.')
        return None

    try:
        ligand_pdb_str = smi2pdb(ligand_smiles, compute_coord=True, optimize='MMFF')
    except ValueError as e:
        logging.warning(e)
        return None
    with tempfile.NamedTemporaryFile(suffix='.pdb') as f_tmp_ligand:
        f_tmp_ligand.write(ligand_pdb_str.encode('utf-8'))
        f_tmp_ligand.seek(0)
        affinity = None
        if box_center is not None:
            if box_size is None:
                box_size = (20., 20., 20.)
            candidate_affinities = []
            logging.info(f'Run docking of {receptor_filename} at center {box_center}.')
            
            try:
                affinity = smina.query_box(
                    receptor_path=receptor_filename,
                    ligand_path=Path(f_tmp_ligand.name),
                    center=box_center,
                    box=box_size,
                    output_complex_path=output_complex_path,
                )
            except SminaError as e:
                logging.warning(
                    'Failed to run on target=%s, ligand=%s, center=%s, size=%s.',
                    receptor_filename, f_tmp_ligand.name, box_center, box_size)
                logging.warning('Error: %s', e)
            logging.debug(
                f'Affinity of box (center={box_center}, size={box_size}): {affinity}'
            )
            candidate_affinities.append(affinity)
        else:
            autobox_filenames = split_result.ligand_filenames.copy()
            if not autobox_filenames:
                logging.warning('No autobox ligands found. Try to run without autobox.')
                autobox_filenames.append(None)
            candidate_affinities = []
            logging.info(f'Run docking of {receptor_filename} on {len(autobox_filenames)} candidates.')
            for autobox_filename in autobox_filenames:
                try:
                    affinity = smina.query(
                        receptor_path=receptor_filename,
                        ligand_path=Path(f_tmp_ligand.name),
                        autobox_ligand_path=autobox_filename,
                        output_complex_path=output_complex_path,
                    )
                except SminaError as e:
                    logging.warning('Failed to run on target=%s, ligand=%s, autobox=%s.',
                                    receptor_filename, f_tmp_ligand.name, autobox_filename)
                    logging.warning('Error: %s', e)
                    continue
                logging.debug(f'Affinity of autobox {autobox_filename}: {affinity}')
                candidate_affinities.append(affinity)
    if not candidate_affinities:
        logging.warning(f'Do not get any affinity scores on {pdb_id}.')
        return None
    affinity = min(candidate_affinities)

    if docking_result_cache is not None:
        docking_result_cache[(pdb_id, ligand_smiles, box_center)] = affinity

    return affinity
