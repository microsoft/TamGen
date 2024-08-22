#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Functions to build fairseq datasets."""

import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import List, Iterable
import random

import dataclasses
import numpy as np
from fy_common_ext.io import pickle_load, pickle_dump, csv_writer, csv_reader
from tqdm import tqdm

from fairseq.data import indexed_dataset
from fairseq.molecule_utils.basic import smiles_utils as smu
from ..coordinate_utils.atom_positions import simple_get_residue_positions, get_residue_average_position
from ..coordinate_utils.binding_site_utils import residues_near_ligand, residues_near_residue, residues_near_positions
from ..database import af2_mmcif_parsing
from ..database import mmcif_utils as mmu
from ..database.af2_mmcif_parsing import AF2MmcifObject, AF2MmCIFParseError
from ..database.common_utils import norm_cif_empty_value, aa_3to1
from ..database.pdb_helper_datasets import get_pdb_ccd_info
from ..database.structure_file import get_af2_mmcif_object


@dataclasses.dataclass
class TargetLigandPairV2:
    pdb_id: str
    chain_id: str
    res_id: int
    insertion_code: str
    ligand_id: str
    ligand_inchi: str
    chain_fasta: str = None
    uniprot_id: str = None

    masked_indices: dict = None
    res_ids: dict = None
    coordinate: np.ndarray = None
    ligand_center: np.ndarray = None    # Ligand center coordinate (center of gravity).

    @property
    def center_x(self):
        return self.ligand_center[0]

    @property
    def center_y(self):
        return self.ligand_center[1]

    @property
    def center_z(self):
        return self.ligand_center[2]


def get_target_ligand_pairs_from_pdb_v2(
        pdb_id: str, mmcif_object: AF2MmcifObject, *,
        ccd_cache_path: Path, min_heavy_atoms: int = 8,
) -> List[TargetLigandPairV2]:
    """Get target-ligand pairs from PDB (new version)."""
    all_ccd_info = get_pdb_ccd_info(ccd_cache_path)
    parsed_info = mmcif_object.raw_string

    # 1. Get ligands and filter them.
    ligands = af2_mmcif_parsing.mmcif_loop_to_dict('_chem_comp.', '_chem_comp.id', parsed_info)
    filtered_ligands = {}
    for ccd_id, v in ligands.items():   # type: str, dict
        if v['_chem_comp.type'] != 'non-polymer':
            if v['_chem_comp.type'] != 'peptide-like':
                logging.debug(f'Filtered {ccd_id} because this is not non-polymer.')
                continue
        ccd_info = all_ccd_info.get(ccd_id, None)
        if ccd_info is None:
            logging.debug(f'Filtered {ccd_info} because CCD information not found.')
            continue
        ccd_mol = smu.inchi2mol(ccd_info.inchi)
        if ccd_mol is None:
            logging.debug(f'Filtered {ccd_id} because cannot convert its InChI to RDKit molecule.')
            continue
        # v['rdkit_mol'] = ccd_mol
        num_heavy_atoms = smu.num_atoms(ccd_mol, only_heavy=True)
        if num_heavy_atoms < min_heavy_atoms:
            logging.debug(f'Filtered {ccd_id} because less than {min_heavy_atoms} heavy atoms.')
            continue
        filtered_ligands[ccd_id] = v

    # 2. Get index of each ligand.
    all_ligand_indices = af2_mmcif_parsing.mmcif_loop_to_dict(
        '_pdbx_nonpoly_scheme',
        ('_pdbx_nonpoly_scheme.pdb_mon_id', '_pdbx_nonpoly_scheme.pdb_strand_id', '_pdbx_nonpoly_scheme.pdb_seq_num',
         '_pdbx_nonpoly_scheme.pdb_ins_code'),
        parsed_info)
    pairs_id = []
    result_pairs = []
    for k, v in all_ligand_indices.items():
        ccd_id, chain_id, seq_id, insertion_code = k
        if ccd_id not in filtered_ligands:
            continue
        try:
            seq_num = int(seq_id)
        except ValueError:
            logging.warning(f'Cannot parse pdb_seq_num {seq_id} of ligand {ccd_id}.')
            continue
        insertion_code = norm_cif_empty_value(insertion_code)
        if (pdb_id, ccd_id) not in pairs_id:
            pairs_id.append((pdb_id, ccd_id))
            result_pairs.append(TargetLigandPairV2(
                pdb_id=pdb_id,
                chain_id=chain_id,
                res_id=seq_num,
                insertion_code=insertion_code,
                ligand_id=ccd_id,
                ligand_inchi=all_ccd_info[ccd_id].inchi,
            ))
    return result_pairs


def process_one_pdb(
        index: int, input_row: dict, *,
        threshold: float, pdb_mmcif_path: Path, pdb_ccd_path: Path,
        min_heavy_atoms: int = 8,
):
    """Process one PDB file.

    Args:
        index:
        input_row:
        threshold:
        pdb_mmcif_path:
        pdb_ccd_path:

    Returns:
        Dict of PDB data, or None if parse failed.
    """
    pdb_id: str = input_row['pdb_id'].lower()
    ligand_inchi = input_row.get('ligand_inchi', None)
    uniprot_id: str = input_row.get('uniprot_id', None)

    logging.debug(f'Processing {index}: {pdb_id}, ligand inchi = {ligand_inchi}')
    logging.debug(f'mmCIF path: {pdb_mmcif_path}; CCD path: {pdb_ccd_path}')

    try:
        mmcif_object = get_af2_mmcif_object(pdb_id, pdb_cache_path=pdb_mmcif_path)
    except AF2MmCIFParseError:
        logging.warning(f'{index}: Failed to parse {pdb_id}.')
        return None

    if mmcif_object is None:
        # No valid chains.
        logging.warning(f'{index}: No protein chains found in mmCIF file {pdb_id}.')
        return None

    # Sequences.
    sequences = mmcif_object.chain_to_seqres

    # UniProt refs.
    uniprot_refs = mmu.get_uniprot_ref(mmcif_object)

    # Binding sites.
    mismatches = {
        'no_sequence': 0,
        'res_id_out_range': 0,
        'code_mismatch': 0,
    }
    binding_sites = mmu.get_binding_sites(mmcif_object, auth_id=False)
    out_binding_sites = {}
    for site_id, binding_site in binding_sites.items():
        for binding_position in binding_site.binding_positions:
            sequence = sequences.get(binding_position.chain_id, None)
            if sequence is None:
                mismatches['no_sequence'] += 1
                break
            try:
                # Residue ID start at 1.
                seq_code = sequence[binding_position.res_id - 1]
            except IndexError:
                mismatches['res_id_out_range'] += 1
                break
            site_code = aa_3to1(binding_position.res_name)
            if seq_code != site_code:
                logging.warning(f'{index}: Site code mismatch, seq code {seq_code}, site code {site_code}')
                mismatches['code_mismatch'] += 1
                break
            out_binding_sites[site_id] = binding_site

    # Coordinates: (chain_id,)
    # Zero masks: (chain_id,)
    coordinates = {}
    zero_masks = {}
    for chain_id in mmcif_object.chain_to_seqres:
        residue_positions = simple_get_residue_positions(mmcif_object, auth_chain_id=chain_id,
                                                         center_of_gravity=True)
        coordinates[chain_id] = residue_positions
        zero_mask = np.any(residue_positions != 0, axis=1).astype(np.int32)
        zero_masks[chain_id] = zero_mask
        assert residue_positions.shape[0] == zero_mask.shape[0]

    # Data pairs.
    data_pairs = []
    for pair in get_target_ligand_pairs_from_pdb_v2(
            pdb_id, mmcif_object, ccd_cache_path=pdb_ccd_path, min_heavy_atoms=min_heavy_atoms):
        if pair.chain_id not in mmcif_object.chain_to_seqres:
            logging.warning(f'{index}: Ligand chain id {pair.chain_id} not found in mmCIF object.')
            continue
        if ligand_inchi is not None and pair.ligand_inchi != ligand_inchi:
            logging.warning(f'{index}: Ligand inchi {pair.ligand_inchi} != expected {ligand_inchi}.')
            continue
        if uniprot_id is not None:
            pair.uniprot_id = uniprot_id
        data_pairs.append(pair)

    # Near ligand: (chain_id, res_id, ligand_id, insertion_code) => {query_chain_id: mask}
    near_ligand_masks = {}
    res_ids = {}
    for pair in data_pairs:
        near_ligand_masks[(pair.chain_id, pair.res_id, pair.ligand_id, pair.insertion_code)] = this_nl_masks = {}
        res_ids[(pair.chain_id, pair.res_id, pair.ligand_id, pair.insertion_code)] = this_res_ids = {}
        for query_chain_id in mmcif_object.chain_to_seqres:
            extra_information = {}
            near_ligand_mask, res_id = residues_near_ligand(
                mmcif_object, auth_chain_id=pair.chain_id, ligand_res_id=pair.res_id, ligand_id=pair.ligand_id,
                ligand_insertion_code=pair.insertion_code, threshold=threshold, query_chain_id=query_chain_id,
                extra_information=extra_information,
            )
            assert coordinates[query_chain_id].shape[0] == near_ligand_mask.shape[0]
            assert len(sequences[query_chain_id]) == near_ligand_mask.shape[0]
            this_nl_masks[query_chain_id] = near_ligand_mask
            this_res_ids[query_chain_id] = res_id
            ligand_residue = extra_information['ligand_residue']
            pair.ligand_center = get_residue_average_position(ligand_residue, only_aa=False, center_of_gravity=True)

    result = {
        'index': index,
        'pdb_id': pdb_id,
        'pairs': data_pairs,
        'sequences': sequences,
        'uniprot_refs': uniprot_refs,
        'binding_sites': out_binding_sites,
        'binding_site_mismatches': mismatches,
        'coordinates': coordinates,
        'zero_masks': zero_masks,
        'near_ligand_masks': near_ligand_masks,
        'res_ids': res_ids,
        'all_chain_ids': sorted(mmcif_object.chain_to_seqres),
    }
    return result


def binarize_single_test_set(
        binary_dir: Path, subset_name: str, *,
        fairseq_root: Path, pre_dicts_root: Path,
):
    """Binarize single test set."""
    src_dir = binary_dir / 'src'
    tmp_dir = binary_dir / 'tmp'
    tmp_dir.mkdir(exist_ok=True, parents=True)

    proc = subprocess.Popen(
        args=f'''\
    python {fairseq_root}/preprocess.py \
    -s tg -t m1 --workers 4 \
    --testpref {src_dir}/{subset_name} \
    --destdir {tmp_dir} \
    --srcdict {pre_dicts_root / "dict.tg.txt"} \
    --tgtdict {pre_dicts_root / "dict.m1.txt"} \
    '''.strip().split()
    )
    proc.communicate()
    proc.kill()

    for test_filename in tmp_dir.glob('test*'):
        name = test_filename.name.replace('test', subset_name)
        dest_fn = binary_dir / name
        if dest_fn.exists():
            dest_fn.unlink()
        shutil.copyfile(test_filename, dest_fn)

    # copy dict
    shutil.copy(pre_dicts_root / "dict.tg.txt", binary_dir)
    shutil.copy(pre_dicts_root / "dict.m1.txt", binary_dir)
    
    # Binarize coordinates.
    coord_fn = src_dir / f'{subset_name}-coordinates.pkl'
    if coord_fn.exists():
        coord_data = pickle_load(coord_fn)
        coord_bin_fn = binary_dir / f'{subset_name}.tg-m1.tg.coord'
        indexed_dataset.binarize_data(coord_data, str(coord_bin_fn), dtype=np.float32, dim=(3,))
        print(f'| Binarize coordinates {coord_fn} to {coord_bin_fn}.')

    # Binarize sites.
    sites_fn = src_dir / f'{subset_name}-sites.pkl'
    if sites_fn.exists():
        sites_data = pickle_load(sites_fn)
        sites_bin_fn = binary_dir / f'{subset_name}.tg-m1.tg.sites'
        indexed_dataset.binarize_data(sites_data, str(sites_bin_fn), dtype=np.int32, dim=None)
        print(f'| Binarize coordinates {sites_fn} to {sites_bin_fn}.')


@dataclasses.dataclass
class TargetDataOnly:
    pdb_id: str

    # Secondary data.
    masked_indices: dict = None
    chain_fasta: str = None
    coordinate: np.ndarray = None
    res_ids: dict = None
    center: np.ndarray = None  # Binding site center coordinate (center of gravity).

    @property
    def center_x(self):
        return self.center[0]

    @property
    def center_y(self):
        return self.center[1]

    @property
    def center_z(self):
        return self.center[2]


def process_one_pdb_given_center_coord(
        index: int, input_row: dict, *,
        threshold: float, pdb_mmcif_path: Path,
):
    pdb_id: str = input_row['pdb_id'].lower()
    center = np.asarray([
        float(input_row['center_x']),
        float(input_row['center_y']),
        float(input_row['center_z']),])
    logging.debug(f'Processing {index}: {pdb_id}')
    logging.debug(f'mmCIF path: {pdb_mmcif_path}')

    try:
        mmcif_object = get_af2_mmcif_object(pdb_id, pdb_cache_path=pdb_mmcif_path)
    except AF2MmCIFParseError:
        logging.warning(f'{index}: Failed to parse {pdb_id}.')
        return None

    if mmcif_object is None:
        # No valid chains.
        logging.warning(f'{index}: No protein chains found in mmCIF file {pdb_id}.')
        return None

    # Sequences.
    sequences = mmcif_object.chain_to_seqres

    # UniProt refs.
    uniprot_refs = mmu.get_uniprot_ref(mmcif_object)
    
    # Coordinates: (chain_id,)
    # Zero masks: (chain_id,)
    coordinates = {}
    zero_masks = {}
    for chain_id in mmcif_object.chain_to_seqres:
        residue_positions = simple_get_residue_positions(mmcif_object, auth_chain_id=chain_id,
                                                         center_of_gravity=True)
        coordinates[chain_id] = residue_positions
        zero_mask = np.any(residue_positions != 0, axis=1).astype(np.int32)
        zero_masks[chain_id] = zero_mask
        assert residue_positions.shape[0] == zero_mask.shape[0]
        
    # Near center: {query_chain_id: mask}
    near_center_masks = {}
    res_ids = {}
    for query_chain_id in mmcif_object.chain_to_seqres:
        near_center_mask, res_id = residues_near_positions(
            mmcif_object, auth_chain_id=query_chain_id, 
            coordinates=[center],
            threshold=threshold,
        )

        assert coordinates[query_chain_id].shape[0] == near_center_mask.shape[0]
        assert len(sequences[query_chain_id]) == near_center_mask.shape[0]
        near_center_masks[query_chain_id] = near_center_mask
        res_ids[query_chain_id] = res_id
    result = {
        'index': index,
        'pdb_id': pdb_id,
        'sequences': sequences,
        'uniprot_refs': uniprot_refs,
        'coordinates': coordinates,
        'zero_masks': zero_masks,
        'near_center_masks': near_center_masks,
        'res_ids': res_ids,
        'center': center,
        'all_chain_ids': sorted(mmcif_object.chain_to_seqres),
    }
    return result


def process_one_pdb_given_res_ids(
        index: int, input_row: dict, *,
        res_ids_fn: Path,
        threshold: float, pdb_mmcif_path: Path,
):
    pdb_id: str = input_row['pdb_id'].lower()

    logging.debug(f'Processing {index}: {pdb_id}')
    logging.debug(f'mmCIF path: {pdb_mmcif_path}')

    try:
        mmcif_object = get_af2_mmcif_object(pdb_id, pdb_cache_path=pdb_mmcif_path)
    except AF2MmCIFParseError:
        logging.warning(f'{index}: Failed to parse {pdb_id}.')
        return None

    if mmcif_object is None:
        # No valid chains.
        logging.warning(f'{index}: No protein chains found in mmCIF file {pdb_id}.')
        return None

    # Sequences.
    sequences = mmcif_object.chain_to_seqres

    # UniProt refs.
    uniprot_refs = mmu.get_uniprot_ref(mmcif_object)
    
    # Coordinates: (chain_id,)
    # Zero masks: (chain_id,)
    coordinates = {}
    zero_masks = {}
    for chain_id in mmcif_object.chain_to_seqres:
        residue_positions = simple_get_residue_positions(mmcif_object, auth_chain_id=chain_id,
                                                         center_of_gravity=True)
        coordinates[chain_id] = residue_positions
        zero_mask = np.any(residue_positions != 0, axis=1).astype(np.int32)
        zero_masks[chain_id] = zero_mask
        assert residue_positions.shape[0] == zero_mask.shape[0]
        
    # Res ids.
    res_ids = pickle_load(res_ids_fn)[index]
    near_center_masks = {}
    new_res_ids = {}
    res_at_positions = mmcif_object.seqres_to_structure
    # Res id starts from 1.
    if threshold <= 0.0:
        # Only include the selected residues
        for chain_id, chain_res_ids in res_ids.items():
            near_center_mask = np.zeros(coordinates[chain_id].shape[0])
            for i, residue_position in res_at_positions[chain_id].items():
                if residue_position.position.residue_number in chain_res_ids:
                    near_center_mask[i] = 1
            near_center_masks[chain_id] = near_center_mask
            new_res_ids[chain_id] = chain_res_ids
        for query_chain_id in mmcif_object.chain_to_seqres:
            if query_chain_id not in near_center_masks.keys():
                near_center_masks[query_chain_id] = np.zeros(coordinates[query_chain_id].shape[0])
                new_res_ids[query_chain_id] = np.array([], dtype=np.int32)
    else:
        given_res_coords = []
        for chain_id, chain_res_ids in res_ids.items():
            for i, residue_position in res_at_positions[chain_id].items():
                if residue_position.position.residue_number in chain_res_ids:
                    given_res_coords.append(coordinates[chain_id][i])
        for query_chain_id in mmcif_object.chain_to_seqres:
            near_center_mask, res_id = residues_near_positions(
                mmcif_object, auth_chain_id=query_chain_id, 
                coordinates=given_res_coords,
                threshold=threshold,
                )
            assert coordinates[query_chain_id].shape[0] == near_center_mask.shape[0]
            assert len(sequences[query_chain_id]) == near_center_mask.shape[0]
            near_center_masks[query_chain_id] = near_center_mask
            new_res_ids[query_chain_id] = res_id
            
    # Compute center
    given_res_coords = []
    for chain_id, chain_res_ids in new_res_ids.items():
        for i, residue_position in res_at_positions[chain_id].items():
            if residue_position.position.residue_number in chain_res_ids:
                given_res_coords.append(coordinates[chain_id][i])
    center = np.average(np.asarray(given_res_coords), axis=0)
    result = {
        'index': index,
        'pdb_id': pdb_id,
        'sequences': sequences,
        'uniprot_refs': uniprot_refs,
        'coordinates': coordinates,
        'zero_masks': zero_masks,
        'near_center_masks': near_center_masks,
        'res_ids': new_res_ids,
        'center': center,
        'all_chain_ids': sorted(mmcif_object.chain_to_seqres),
    }
    return result


def dump_center_data(
    all_data: list, name: str, output_dir: Path, *,
    fairseq_root: Path, pre_dicts_root: Path, max_len: int = 1023,
):
    logging.debug(f'Dump data to {output_dir}/{name}*.')
    
    output_dir.mkdir(exist_ok=True, parents=True)
    src_dir = output_dir / 'src'
    src_dir.mkdir(exist_ok=True)

    processed_data = []
    for data in all_data:
        all_chain_ids = data['all_chain_ids']
        sequences = data['sequences']
        coordinates = data['coordinates']
        nm_masks = data['near_center_masks']

        # Get masked indices.
        masked_indices = {
            query_chain_id: np.nonzero(nm_masks[query_chain_id])[0]
            for query_chain_id in all_chain_ids
        }

        processed_sequence = []
        for query_chain_id in all_chain_ids:
            processed_sequence.extend(sequences[query_chain_id][i] for i in masked_indices[query_chain_id])
        processed_fasta = ''.join(processed_sequence)

        processed_coord_stack = []
        for query_chain_id in all_chain_ids:
            processed_coord_stack.append(coordinates[query_chain_id][masked_indices[query_chain_id]])
        processed_coord = np.concatenate(processed_coord_stack)
        
        processed_data.append(TargetDataOnly(
            pdb_id=data['pdb_id'],
            masked_indices=masked_indices,
            chain_fasta=processed_fasta,
            coordinate=processed_coord,
            res_ids=data['res_ids'],
            center=data['center'],
        ))


    # Pairs file.
    pairs_fn = src_dir / f'{name}-info.csv'
    fieldnames = ['pdb_id', 
                  'chain_fasta', 'center_x', 'center_y', 'center_z']
    with csv_writer(pairs_fn, fieldnames=fieldnames) as writer:
        writer.writeheader()
        for data in processed_data:
            writer.writerow({key: getattr(data, key) for key in fieldnames})
    # tg.
    tg_orig_fn = src_dir / f'{name}.tg.orig'
    tg_fn = src_dir / f'{name}.tg'
    with open(tg_orig_fn, 'w', encoding='utf-8') as f_tg_orig, \
            open(tg_fn, 'w', encoding='utf-8') as f_tg:
        for data in processed_data:
            print(data.chain_fasta, file=f_tg_orig)
            print(' '.join(data.chain_fasta[:max_len]), file=f_tg)

    # m1 (only stub).
    m1_orig_fn = src_dir / f'{name}.m1.orig'
    m1_fn = src_dir / f'{name}.m1'
    with open(m1_orig_fn, 'w', encoding='utf-8') as f_m1_orig, \
            open(m1_fn, 'w', encoding='utf-8') as f_m1:
        for _ in processed_data:
            smiles = tokenized_smiles = 'C'
            print(smiles, file=f_m1_orig)
            print(tokenized_smiles, file=f_m1)

    # Coordinates orig.
    coord_orig_fn = src_dir / f'{name}-coordinates.orig.pkl'
    coord_orig_data = {data['pdb_id']: data['coordinates'] for data in all_data}
    pickle_dump(coord_orig_data, coord_orig_fn)

    # Coordinates.
    coord_fn = src_dir / f'{name}-coordinates.pkl'
    truncated_coord = {
        index: data.coordinate[:max_len, ...]
        for index, data in enumerate(processed_data)
    }
    pickle_dump(truncated_coord, coord_fn)

    # All-1 site mask.
    site_mask_fn = src_dir / f'{name}-sites.pkl'
    truncated_all1_site_mask = {
        index: np.ones((truncated_coord[index].shape[0],), dtype=np.int32)
        for index in truncated_coord
    }
    pickle_dump(truncated_all1_site_mask, site_mask_fn)

    # All sequences.
    exist_pdb_chain_ids = set()
    with csv_writer(src_dir / f'{name}-all-sequences.csv', fieldnames=['pdb_id', 'chain_id', 'fasta']) as writer:
        writer.writeheader()
        for data in all_data:
            pdb_id = data['pdb_id']
            for chain_id, sequence in data['sequences'].items():
                if (pdb_id, chain_id) in exist_pdb_chain_ids:
                    continue
                exist_pdb_chain_ids.add((pdb_id, chain_id))
                writer.writerow({
                    'pdb_id': pdb_id,
                    'chain_id': chain_id,
                    'fasta': sequence,
                })

    # Mask indices.
    all_mask_indices = {}
    for index, data in enumerate(processed_data):
        all_mask_indices[index] = data.masked_indices
    mask_indices_fn = src_dir / f'{name}-mask-indices.pkl'
    pickle_dump(all_mask_indices, mask_indices_fn)
    
    # Res ids.
    all_res_ids = {}
    for index, data in enumerate(processed_data):
        all_res_ids[index] = data.res_ids
    res_ids_fn = src_dir / f'{name}-res-ids.pkl'
    pickle_dump(all_res_ids, res_ids_fn)

    binarize_single_test_set(output_dir, name, fairseq_root=fairseq_root, pre_dicts_root=pre_dicts_root)
    shutil.rmtree(output_dir / 'tmp')

def dump_center_data_ligand(
    all_data: list, name: str, output_dir: Path, *,
    fairseq_root: Path, pre_dicts_root: Path, max_len: int = 1023,
    ligand_list: list = None
):
    logging.debug(f'Dump data to {output_dir}/{name}*.')
    
    output_dir.mkdir(exist_ok=True, parents=True)
    src_dir = output_dir / 'src'
    src_dir.mkdir(exist_ok=True)

    processed_data = []
    for data in all_data:
        all_chain_ids = data['all_chain_ids']
        sequences = data['sequences']
        coordinates = data['coordinates']
        nm_masks = data['near_center_masks']

        # Get masked indices.
        masked_indices = {
            query_chain_id: np.nonzero(nm_masks[query_chain_id])[0]
            for query_chain_id in all_chain_ids
        }

        processed_sequence = []
        for query_chain_id in all_chain_ids:
            processed_sequence.extend(sequences[query_chain_id][i] for i in masked_indices[query_chain_id])
        processed_fasta = ''.join(processed_sequence)
        if len(processed_fasta) == 0:
            continue

        processed_coord_stack = []
        for query_chain_id in all_chain_ids:
            processed_coord_stack.append(coordinates[query_chain_id][masked_indices[query_chain_id]])
        processed_coord = np.concatenate(processed_coord_stack)
        
        processed_data.append(TargetDataOnly(
            pdb_id=data['pdb_id'],
            masked_indices=masked_indices,
            chain_fasta=processed_fasta,
            coordinate=processed_coord,
            res_ids=data['res_ids'],
            center=data['center'],
        ))


    # Pairs file.
    pairs_fn = src_dir / f'{name}-info.csv'
    fieldnames = ['pdb_id', 
                  'chain_fasta', 'center_x', 'center_y', 'center_z']
    with csv_writer(pairs_fn, fieldnames=fieldnames) as writer:
        writer.writeheader()
        for data in processed_data:
            writer.writerow({key: getattr(data, key) for key in fieldnames})
    # tg.
    tg_orig_fn = src_dir / f'{name}.tg.orig'
    tg_fn = src_dir / f'{name}.tg'
    with open(tg_orig_fn, 'w', encoding='utf-8') as f_tg_orig, \
        open(tg_fn, 'w', encoding='utf-8') as f_tg:
        for _ in range(len(ligand_list)):
                for data in processed_data:
                    print(data.chain_fasta, file=f_tg_orig)
                    print(' '.join(data.chain_fasta[:max_len]), file=f_tg)

    # m1 (only stub).
    m1_orig_fn = src_dir / f'{name}.m1.orig'
    m1_fn = src_dir / f'{name}.m1'
    with open(m1_orig_fn, 'w', encoding='utf-8') as f_m1_orig, \
        open(m1_fn, 'w', encoding='utf-8') as f_m1:
        for ll in ligand_list:
            for _ in processed_data:
                tokenized_smiles = smu.tokenize_smiles(ll)
                print(ll, file=f_m1_orig)
                print(tokenized_smiles, file=f_m1)

    # Coordinates orig.
    coord_orig_fn = src_dir / f'{name}-coordinates.orig.pkl'
    coord_orig_data = {data['pdb_id']: data['coordinates'] for data in all_data}
    pickle_dump(coord_orig_data, coord_orig_fn)

    # Coordinates.
    coord_fn = src_dir / f'{name}-coordinates.pkl'
    truncated_coord = {}
    for i in range(len(ligand_list)):
        for j, data in enumerate(processed_data):
            truncated_coord[i*len(processed_data)+j] = data.coordinate[:max_len, ...]
    # truncated_coord = {
    #     index: data.coordinate[:max_len, ...]
    #     for index, data in enumerate(processed_data)
    # }
    pickle_dump(truncated_coord, coord_fn)

    # All-1 site mask.
    site_mask_fn = src_dir / f'{name}-sites.pkl'
    truncated_all1_site_mask = {
        index: np.ones((truncated_coord[index].shape[0],), dtype=np.int32)
        for index in truncated_coord
    }
    pickle_dump(truncated_all1_site_mask, site_mask_fn)

    # All sequences.
    exist_pdb_chain_ids = set()
    with csv_writer(src_dir / f'{name}-all-sequences.csv', fieldnames=['pdb_id', 'chain_id', 'fasta']) as writer:
        writer.writeheader()
        for data in all_data:
            pdb_id = data['pdb_id']
            for chain_id, sequence in data['sequences'].items():
                if (pdb_id, chain_id) in exist_pdb_chain_ids:
                    continue
                exist_pdb_chain_ids.add((pdb_id, chain_id))
                writer.writerow({
                    'pdb_id': pdb_id,
                    'chain_id': chain_id,
                    'fasta': sequence,
                })

    # Mask indices.
    all_mask_indices = {}
    for i in range(len(ligand_list)): 
        for j, data in enumerate(processed_data):
            all_mask_indices[i*len(processed_data)+j] = data.masked_indices
    mask_indices_fn = src_dir / f'{name}-mask-indices.pkl'
    pickle_dump(all_mask_indices, mask_indices_fn)
    
    # Res ids.
    all_res_ids = {}
    for i in range(len(ligand_list)): 
        for index, data in enumerate(processed_data):
            all_res_ids[i*len(processed_data)+index] = data.res_ids
    res_ids_fn = src_dir / f'{name}-res-ids.pkl'
    pickle_dump(all_res_ids, res_ids_fn)

    binarize_single_test_set(output_dir, name, fairseq_root=fairseq_root, pre_dicts_root=pre_dicts_root)
    shutil.rmtree(output_dir / 'tmp')

def dump_center_data_multi_ligand(
    all_data: list, name: str, output_dir: Path, *,
    fairseq_root: Path, pre_dicts_root: Path, max_len: int = 1023,
    ligand_list: list = None, sample_weights: list = None,
):
    logging.debug(f'Dump data to {output_dir}/{name}*.')
    
    output_dir.mkdir(exist_ok=True, parents=True)
    src_dir = output_dir / 'src'
    src_dir.mkdir(exist_ok=True)

    processed_data = []
    for data in all_data:
        all_chain_ids = data['all_chain_ids']
        sequences = data['sequences']
        coordinates = data['coordinates']
        nm_masks = data['near_center_masks']

        # Get masked indices.
        masked_indices = {
            query_chain_id: np.nonzero(nm_masks[query_chain_id])[0]
            for query_chain_id in all_chain_ids
        }

        processed_sequence = []
        for query_chain_id in all_chain_ids:
            processed_sequence.extend(sequences[query_chain_id][i] for i in masked_indices[query_chain_id])
        processed_fasta = ''.join(processed_sequence)
        if len(processed_fasta) == 0:
            continue

        processed_coord_stack = []
        for query_chain_id in all_chain_ids:
            processed_coord_stack.append(coordinates[query_chain_id][masked_indices[query_chain_id]])
        processed_coord = np.concatenate(processed_coord_stack)
        assert processed_coord.shape[0] == len(processed_fasta)
        
        processed_data.append(TargetDataOnly(
            pdb_id=data['pdb_id'],
            masked_indices=masked_indices,
            chain_fasta=processed_fasta,
            coordinate=processed_coord,
            res_ids=data['res_ids'],
            center=data['center'],
        ))


    # Pairs file.
    pairs_fn = src_dir / f'{name}-info.csv'
    fieldnames = ['pdb_id', 
                  'chain_fasta', 'center_x', 'center_y', 'center_z']
    with csv_writer(pairs_fn, fieldnames=fieldnames) as writer:
        writer.writeheader()
        for data in processed_data:
            writer.writerow({key: getattr(data, key) for key in fieldnames})
    # tg.
    tg_orig_fn = src_dir / f'{name}.tg.orig'
    tg_fn = src_dir / f'{name}.tg'
    with open(tg_orig_fn, 'w', encoding='utf-8') as f_tg_orig, \
            open(tg_fn, 'w', encoding='utf-8') as f_tg:
        for data in processed_data:
            print(data.chain_fasta, file=f_tg_orig)
            print(' '.join(data.chain_fasta[:max_len]), file=f_tg)

    # m1 (only stub).
    if ligand_list is not None:
        m1_orig_fn = src_dir / f'{name}.m1.orig'
        m1_fn = src_dir / f'{name}.m1'
        with open(m1_orig_fn, 'w', encoding='utf-8') as f_m1_orig, \
            open(m1_fn, 'w', encoding='utf-8') as f_m1:
            for ll in ligand_list:
                tokenized_smiles = [smu.tokenize_smiles(x) for x in ll]
                # use <split> as delimiter
                print('<split>'.join(ll), file=f_m1_orig)
                print(' <split> '.join(tokenized_smiles), file=f_m1)
                # tokenized_smiles = smu.tokenize_smiles(ll)
                # print(ll, file=f_m1_orig)
                # print(tokenized_smiles, file=f_m1)
    
    # sample weights
    if sample_weights is not None:
        sample_weights_fn = src_dir / f'{name}-sample-weights.pkl'
        pickle_dump(sample_weights, sample_weights_fn)
        # copy to output_dir
        shutil.copy(sample_weights_fn, output_dir)

    # Coordinates orig.
    coord_orig_fn = src_dir / f'{name}-coordinates.orig.pkl'
    coord_orig_data = {data['pdb_id']: data['coordinates'] for data in all_data}
    pickle_dump(coord_orig_data, coord_orig_fn)

    # Coordinates.
    coord_fn = src_dir / f'{name}-coordinates.pkl'
    truncated_coord = {
        index: data.coordinate[:max_len, ...]
        for index, data in enumerate(processed_data)
    }
    pickle_dump(truncated_coord, coord_fn)

    # All-1 site mask.
    site_mask_fn = src_dir / f'{name}-sites.pkl'
    truncated_all1_site_mask = {
        index: np.ones((truncated_coord[index].shape[0],), dtype=np.int32)
        for index in truncated_coord
    }
    pickle_dump(truncated_all1_site_mask, site_mask_fn)

    # All sequences.
    exist_pdb_chain_ids = set()
    with csv_writer(src_dir / f'{name}-all-sequences.csv', fieldnames=['pdb_id', 'chain_id', 'fasta']) as writer:
        writer.writeheader()
        for data in all_data:
            pdb_id = data['pdb_id']
            for chain_id, sequence in data['sequences'].items():
                if (pdb_id, chain_id) in exist_pdb_chain_ids:
                    continue
                exist_pdb_chain_ids.add((pdb_id, chain_id))
                writer.writerow({
                    'pdb_id': pdb_id,
                    'chain_id': chain_id,
                    'fasta': sequence,
                })

    # Mask indices.
    all_mask_indices = {}
    for index, data in enumerate(processed_data):
        all_mask_indices[index] = data.masked_indices
    mask_indices_fn = src_dir / f'{name}-mask-indices.pkl'
    pickle_dump(all_mask_indices, mask_indices_fn)
    
    # Res ids.
    all_res_ids = {}
    for index, data in enumerate(processed_data):
        all_res_ids[index] = data.res_ids
    res_ids_fn = src_dir / f'{name}-res-ids.pkl'
    pickle_dump(all_res_ids, res_ids_fn)

    binarize_single_test_set(output_dir, name, fairseq_root=fairseq_root, pre_dicts_root=pre_dicts_root)
    shutil.rmtree(output_dir / 'tmp')

def dump_data(
        all_data: list, name: str, output_dir: Path, *,
        fairseq_root: Path, pre_dicts_root: Path, max_len: int = 1023,
):
    logging.debug(f'Dump source data to {output_dir}/{name}*.')

    output_dir.mkdir(exist_ok=True, parents=True)
    src_dir = output_dir / 'src'
    src_dir.mkdir(exist_ok=True)

    processed_pairs = []
    for data in all_data:
        all_chain_ids = data['all_chain_ids']
        sequences = data['sequences']
        coordinates = data['coordinates']
        nl_masks = data['near_ligand_masks']
        res_ids = data['res_ids']
        for pair in data['pairs']:  # type: TargetLigandPairV2
            # Collect subsequences from near ligand masks
            ligand_index = (pair.chain_id, pair.res_id, pair.ligand_id, pair.insertion_code)

            # Get masked indices.
            masked_indices = {
                query_chain_id: np.nonzero(nl_masks[ligand_index][query_chain_id])[0]
                for query_chain_id in all_chain_ids
            }
            pair.masked_indices = masked_indices
            pair.res_ids = {
                query_chain_id: res_ids[ligand_index][query_chain_id]
                for query_chain_id in all_chain_ids
            }

            processed_sequence = []
            for query_chain_id in all_chain_ids:
                processed_sequence.extend(sequences[query_chain_id][i] for i in masked_indices[query_chain_id])
            processed_fasta = ''.join(processed_sequence)

            if not processed_fasta:
                logging.warning(f'Empty processed chain, skip.')
                continue

            processed_coord_stack = []
            for query_chain_id in all_chain_ids:
                processed_coord_stack.append(coordinates[query_chain_id][masked_indices[query_chain_id]])
            processed_coord = np.concatenate(processed_coord_stack)

            pair.chain_fasta = processed_fasta
            pair.coordinate = processed_coord
            processed_pairs.append(pair)

    # Pairs file.
    pairs_fn = src_dir / f'{name}-info.csv'
    fieldnames = ['pdb_id', 'chain_id', 'res_id', 'insertion_code', 'ligand_id', 'ligand_inchi', 'chain_fasta',
                  'center_x', 'center_y', 'center_z']
    if processed_pairs[0].uniprot_id is not None:
        fieldnames.insert(0, 'uniprot_id')
    with csv_writer(pairs_fn, fieldnames=fieldnames) as writer:
        writer.writeheader()
        for pair in processed_pairs:
            row = {key: getattr(pair, key) for key in fieldnames}
            if pair.uniprot_id is not None:
                row['uniprot_id'] = pair.uniprot_id
            writer.writerow(row)
    # tg.
    tg_orig_fn = src_dir / f'{name}.tg.orig'
    tg_fn = src_dir / f'{name}.tg'
    with open(tg_orig_fn, 'w', encoding='utf-8') as f_tg_orig, \
            open(tg_fn, 'w', encoding='utf-8') as f_tg:
        for pair in processed_pairs:
            print(pair.chain_fasta, file=f_tg_orig)
            print(' '.join(pair.chain_fasta[:max_len]), file=f_tg)

    # m1.
    m1_orig_fn = src_dir / f'{name}.m1.orig'
    m1_fn = src_dir / f'{name}.m1'
    with open(m1_orig_fn, 'w', encoding='utf-8') as f_m1_orig, \
            open(m1_fn, 'w', encoding='utf-8') as f_m1:
        for pair in processed_pairs:
            inchi = pair.ligand_inchi
            smiles = smu.inchi2smi(inchi)
            tokenized_smiles = smu.tokenize_smiles(smiles)
            print(smiles, file=f_m1_orig)
            print(tokenized_smiles, file=f_m1)

    # Coordinates orig.
    coord_orig_fn = src_dir / f'{name}-coordinates.orig.pkl'
    coord_orig_data = {data['pdb_id']: data['coordinates'] for data in all_data}
    pickle_dump(coord_orig_data, coord_orig_fn)

    # Coordinates.
    coord_fn = src_dir / f'{name}-coordinates.pkl'
    truncated_coord = {
        index: pair.coordinate[:max_len, ...]
        for index, pair in enumerate(processed_pairs)
    }
    pickle_dump(truncated_coord, coord_fn)

    # All sequences.
    exist_pdb_chain_ids = set()
    with csv_writer(src_dir / f'{name}-all-sequences.csv', fieldnames=['pdb_id', 'chain_id', 'fasta']) as writer:
        writer.writeheader()
        for data in all_data:
            pdb_id = data['pdb_id']
            for chain_id, sequence in data['sequences'].items():
                if (pdb_id, chain_id) in exist_pdb_chain_ids:
                    continue
                exist_pdb_chain_ids.add((pdb_id, chain_id))
                writer.writerow({
                    'pdb_id': pdb_id,
                    'chain_id': chain_id,
                    'fasta': sequence,
                })

    # Mask indices.
    all_mask_indices = {}
    for index, pair in enumerate(processed_pairs):
        all_mask_indices[index] = pair.masked_indices
    mask_indices_fn = src_dir / f'{name}-mask-indices.pkl'
    pickle_dump(all_mask_indices, mask_indices_fn)
    
    # Res ids.
    all_res_ids = {}
    for index, pair in enumerate(processed_pairs):
        all_res_ids[index] = pair.res_ids
    res_ids_fn = src_dir / f'{name}-res-ids.pkl'
    pickle_dump(all_res_ids, res_ids_fn)

    binarize_single_test_set(output_dir, name, fairseq_root=fairseq_root, pre_dicts_root=pre_dicts_root)
    shutil.rmtree(output_dir / 'tmp')


@dataclasses.dataclass
class Mutation:
    uniprot_id: str
    pdb_id: str
    chain_id: str
    orig_aa: str
    new_aa: str
    mutation_id: int
    pdb_mutation_id: int
    uniprot_begin: int
    uniprot_end: int
    pdb_begin: int
    pdb_end: int
    uniprot_seq: str
    pdb_seq: str

    # Secondary data.
    masked_indices: dict = None
    res_ids: dict = None
    chain_fasta: str = None
    coordinate: np.ndarray = None
    mutation_center: np.ndarray = None  # Mutation center coordinate (center of gravity).

    @property
    def center_x(self):
        return self.mutation_center[0]

    @property
    def center_y(self):
        return self.mutation_center[1]

    @property
    def center_z(self):
        return self.mutation_center[2]


def process_one_mutation(
        index: int, mutation: Mutation, *,
        threshold: float, pdb_mmcif_path: Path,
):
    pdb_id = mutation.pdb_id
    chain_id = mutation.chain_id

    logging.debug(
        f'Processing {index}: {mutation.uniprot_id}.{pdb_id}.{mutation.orig_aa}{mutation.mutation_id}{mutation.new_aa}')
    logging.debug(f'mmCIF path: {pdb_mmcif_path}')

    try:
        mmcif_object = get_af2_mmcif_object(pdb_id, pdb_cache_path=pdb_mmcif_path)
    except AF2MmCIFParseError:
        logging.warning(f'{index}: Failed to parse {pdb_id}.')
        return None

    if mmcif_object is None:
        # No valid chains.
        logging.warning(f'{index}: No protein chains found in mmCIF file {pdb_id}.')
        return None

    # Sequences.
    sequences = mmcif_object.chain_to_seqres

    try:
        mut_sequence = sequences[chain_id]
    except KeyError:
        logging.warning(f'{index}: Cannot find chain {chain_id} in {pdb_id} mmCIF file.')
        return None
    if mut_sequence != mutation.pdb_seq:
        logging.warning(f'{index}: PDB sequence mismatch in {pdb_id}.')
        return None

    try:
        mut_aa = mut_sequence[mutation.pdb_mutation_id]
    except IndexError:
        logging.warning(f'{index}: {mutation.pdb_mutation_id} not in chain {chain_id} '
                        f'(total length {len(mut_sequence)}).')
        return None
    if mut_aa != mutation.new_aa:
        logging.warning(f'{index}: PDB mutation aa {mut_aa} != {mutation.new_aa} in mutation data')
        return None

    # Coordinates: (chain_id,)
    # Zero masks: (chain_id,)
    coordinates = {}
    zero_masks = {}
    for query_chain_id in mmcif_object.chain_to_seqres:
        residue_positions = simple_get_residue_positions(mmcif_object, auth_chain_id=query_chain_id,
                                                         center_of_gravity=True)
        coordinates[query_chain_id] = residue_positions
        zero_mask = np.any(residue_positions != 0, axis=1).astype(np.int32)
        zero_masks[query_chain_id] = zero_mask
        assert residue_positions.shape[0] == zero_mask.shape[0]

    # Near mutation: {query_chain_id: mask}
    near_mutation_masks = {}
    res_ids = {}
    for query_chain_id in mmcif_object.chain_to_seqres:
        extra_information = {}
        near_mutation_mask, res_id = residues_near_residue(
            mmcif_object, auth_chain_id=chain_id, res_id=mutation.pdb_mutation_id, threshold=threshold,
            query_chain_id=query_chain_id, extra_information=extra_information,
        )
        center_residue = extra_information.get('center_residue', None)
        if center_residue is None:
            logging.warning(f'{index}: Cannot find center residue {mutation.pdb_mutation_id} in {pdb_id}.{chain_id}.')
            return None
        assert coordinates[query_chain_id].shape[0] == near_mutation_mask.shape[0]
        assert len(sequences[query_chain_id]) == near_mutation_mask.shape[0]
        near_mutation_masks[query_chain_id] = near_mutation_mask
        res_ids[query_chain_id] = res_id
        mutation.mutation_center = get_residue_average_position(center_residue, only_aa=False, center_of_gravity=True)

    return {
        'index': index,
        'pdb_id': pdb_id,
        'mutation': mutation,
        'sequences': sequences,
        'coordinates': coordinates,
        'zero_masks': zero_masks,
        'near_mutation_masks': near_mutation_masks,
        'res_ids': res_ids,
        'all_chain_ids': sorted(mmcif_object.chain_to_seqres),
    }


def dump_mutation_data(
        all_data: list, name: str, output_dir: Path, *,
        fairseq_root: Path, pre_dicts_root: Path, max_len: int = 1023,
):
    logging.debug(f'Dump mutation source data to {output_dir}/{name}*.')

    output_dir.mkdir(exist_ok=True, parents=True)
    src_dir = output_dir / 'src'
    src_dir.mkdir(exist_ok=True)

    processed_mutations = []
    for data in all_data:
        all_chain_ids = data['all_chain_ids']
        mutation: Mutation = data['mutation']
        sequences = data['sequences']
        coordinates = data['coordinates']
        nm_masks = data['near_mutation_masks']
        res_ids = data['res_ids']
        # Get masked indices.
        masked_indices = {
            query_chain_id: np.nonzero(nm_masks[query_chain_id])[0]
            for query_chain_id in all_chain_ids
        }
        mutation.masked_indices = masked_indices
        mutation.res_ids = res_ids
        processed_sequence = []
        for query_chain_id in all_chain_ids:
            processed_sequence.extend(sequences[query_chain_id][i] for i in masked_indices[query_chain_id])
        processed_fasta = ''.join(processed_sequence)

        processed_coord_stack = []
        for query_chain_id in all_chain_ids:
            processed_coord_stack.append(coordinates[query_chain_id][masked_indices[query_chain_id]])
        processed_coord = np.concatenate(processed_coord_stack)

        mutation.chain_fasta = processed_fasta
        mutation.coordinate = processed_coord
        processed_mutations.append(mutation)

    # Pairs file.
    pairs_fn = src_dir / f'{name}-info.csv'
    fieldnames = ['uniprot_id', 'pdb_id', 'chain_id', 'orig_aa', 'new_aa', 'mutation_id', 'pdb_mutation_id',
                  'uniprot_begin', 'uniprot_end', 'pdb_begin', 'pdb_end',
                  'chain_fasta', 'center_x', 'center_y', 'center_z']
    with csv_writer(pairs_fn, fieldnames=fieldnames) as writer:
        writer.writeheader()
        for mutation in processed_mutations:
            writer.writerow({key: getattr(mutation, key) for key in fieldnames})
    # tg.
    tg_orig_fn = src_dir / f'{name}.tg.orig'
    tg_fn = src_dir / f'{name}.tg'
    with open(tg_orig_fn, 'w', encoding='utf-8') as f_tg_orig, \
            open(tg_fn, 'w', encoding='utf-8') as f_tg:
        for mutation in processed_mutations:
            print(mutation.chain_fasta, file=f_tg_orig)
            print(' '.join(mutation.chain_fasta[:max_len]), file=f_tg)

    # m1 (only stub).
    m1_orig_fn = src_dir / f'{name}.m1.orig'
    m1_fn = src_dir / f'{name}.m1'
    with open(m1_orig_fn, 'w', encoding='utf-8') as f_m1_orig, \
            open(m1_fn, 'w', encoding='utf-8') as f_m1:
        for _ in processed_mutations:
            smiles = tokenized_smiles = 'C'
            print(smiles, file=f_m1_orig)
            print(tokenized_smiles, file=f_m1)

    # Coordinates orig.
    coord_orig_fn = src_dir / f'{name}-coordinates.orig.pkl'
    coord_orig_data = {data['pdb_id']: data['coordinates'] for data in all_data}
    pickle_dump(coord_orig_data, coord_orig_fn)

    # Coordinates.
    coord_fn = src_dir / f'{name}-coordinates.pkl'
    truncated_coord = {
        index: mutation.coordinate[:max_len, ...]
        for index, mutation in enumerate(processed_mutations)
    }
    pickle_dump(truncated_coord, coord_fn)

    # All-1 site mask.
    site_mask_fn = src_dir / f'{name}-sites.pkl'
    truncated_all1_site_mask = {
        index: np.ones((truncated_coord[index].shape[0],), dtype=np.int32)
        for index in truncated_coord
    }
    pickle_dump(truncated_all1_site_mask, site_mask_fn)

    # All sequences.
    exist_pdb_chain_ids = set()
    with csv_writer(src_dir / f'{name}-all-sequences.csv', fieldnames=['pdb_id', 'chain_id', 'fasta']) as writer:
        writer.writeheader()
        for data in all_data:
            pdb_id = data['pdb_id']
            for chain_id, sequence in data['sequences'].items():
                if (pdb_id, chain_id) in exist_pdb_chain_ids:
                    continue
                exist_pdb_chain_ids.add((pdb_id, chain_id))
                writer.writerow({
                    'pdb_id': pdb_id,
                    'chain_id': chain_id,
                    'fasta': sequence,
                })

    # Mask indices.
    all_mask_indices = {}
    for index, mutation in enumerate(processed_mutations):
        all_mask_indices[index] = mutation.masked_indices
    mask_indices_fn = src_dir / f'{name}-mask-indices.pkl'
    pickle_dump(all_mask_indices, mask_indices_fn)
    
    # Res ids.
    all_res_ids = {}
    for index, mutation in enumerate(processed_mutations):
        all_res_ids[index] = mutation.res_ids
    res_ids_fn = src_dir / f'{name}-res-ids.pkl'
    pickle_dump(all_res_ids, res_ids_fn)

    binarize_single_test_set(output_dir, name, fairseq_root=fairseq_root, pre_dicts_root=pre_dicts_root)
    shutil.rmtree(output_dir / 'tmp')


def _no_rule(inchi, smiles) -> bool:
    return False


_PATTERN_CO = re.compile(r'[CO()=]+')


def _basic_rule(inchi, smiles) -> bool:
    m = _PATTERN_CO.fullmatch(smiles)
    if m:
        return True
    return False


_RULES = {
    'none': _no_rule,
    'basic': _basic_rule,
}


def modify_and_build(
        input_dir: Path, input_name: str,
        output_dir: Path, output_name: str = None, *,
        fairseq_root: Path, pre_dicts_root: Path,
        data_id_list: Iterable[int] = None,
        rm_pdb_ids: Iterable[str] = (),
        min_heavy_atoms: int = None,
        rm_ligand_inchi: Iterable[str] = (),
        rm_ligand_rule: str = 'none',
):
    """Filter some data and build."""

    in_src_dir = input_dir / 'src'
    if output_name is None:
        output_name = input_name

    logging.info(f'Convert {in_src_dir}:{input_name} to {output_dir}:{output_name}.')
    if data_id_list is not None:
        logging.info(f'Filter by data ID list.')
    else:
        logging.info(f'Min ligand heavy atoms: {min_heavy_atoms}.')
        logging.info(f'First 5 removed ligand InChI: {rm_ligand_inchi[:5]}.')
        logging.info(f'Remove ligand rule: {rm_ligand_rule}.')

    if data_id_list is not None:
        with csv_reader(in_src_dir / f'{input_name}-info.csv', dict_reader=True) as reader:
            total_ids = set(range(len(list(reader))))
        rm_indices_set = total_ids - set(data_id_list)
    else:
        rm_pdb_ids_set = {pdb_id.lower() for pdb_id in rm_pdb_ids}
        rm_ligand_inchi_set = set(rm_ligand_inchi)
        rule_func = _RULES[rm_ligand_rule]

        rm_indices = []
        with csv_reader(in_src_dir / f'{input_name}-info.csv', dict_reader=True) as reader:
            for i, row in enumerate(tqdm(reader)):
                if row['pdb_id'] in rm_pdb_ids_set:
                    rm_indices.append(i)
                    continue
                chain_fasta = row['chain_fasta']
                if not chain_fasta:
                    rm_indices.append(i)
                    continue
                ligand_inchi = row.get('ligand_inchi', None)
                if ligand_inchi in rm_ligand_inchi_set:
                    rm_indices.append(i)
                    continue
                smiles = smu.inchi2smi(ligand_inchi)
                if rule_func(ligand_inchi, smiles):
                    rm_indices.append(i)
                    rm_ligand_inchi_set.add(ligand_inchi)
                    continue
                if min_heavy_atoms is not None and ligand_inchi is not None:
                    mol = smu.inchi2mol(ligand_inchi)
                    num_heavy_atoms = smu.num_atoms(mol)
                    if num_heavy_atoms < min_heavy_atoms:
                        rm_indices.append(i)
                        rm_ligand_inchi_set.add(ligand_inchi)
                        continue
        rm_indices_set = set(rm_indices)

    # Apply filter.
    logging.info(f'{len(rm_indices_set)} data entries will be filtered.')

    out_src_dir = output_dir / 'src'
    out_src_dir.mkdir(exist_ok=True, parents=True)

    # info.
    with csv_reader(in_src_dir / f'{input_name}-info.csv', dict_reader=True) as reader:
        with csv_writer(out_src_dir / f'{output_name}-info.csv', fieldnames=reader.fieldnames) as writer:
            writer.writeheader()
            for i, row in enumerate(reader):
                if i not in rm_indices_set:
                    writer.writerow(row)

    # tg/m1.
    def _filter_text(in_filename: Path, out_filename: Path):
        with open(in_filename, 'r', encoding='utf-8') as f_in, \
                open(out_filename, 'w', encoding='utf-8') as f_out:
            for i_, line in enumerate(f_in):
                if i_ in rm_indices_set:
                    continue
                f_out.write(line)

    _filter_text(in_src_dir / f'{input_name}.tg', out_src_dir / f'{output_name}.tg')
    _filter_text(in_src_dir / f'{input_name}.tg.orig', out_src_dir / f'{output_name}.tg.orig')
    _filter_text(in_src_dir / f'{input_name}.m1', out_src_dir / f'{output_name}.m1')
    _filter_text(in_src_dir / f'{input_name}.m1.orig', out_src_dir / f'{output_name}.m1.orig')

    # coord/sites.
    def _filter_pkl(in_filename: Path, out_filename: Path):
        _data = pickle_load(in_filename)
        _filtered_data = {}
        new_i = 0
        for i_ in sorted(_data):
            if i_ in rm_indices_set:
                continue
            value = _data[i_]
            _filtered_data[new_i] = value
            new_i += 1
        pickle_dump(_filtered_data, out_filename)

    _filter_pkl(in_src_dir / f'{input_name}-coordinates.pkl', out_src_dir / f'{output_name}-coordinates.pkl')
    _filter_pkl(in_src_dir / f'{input_name}-sites.pkl', out_src_dir / f'{output_name}-sites.pkl')
    # _filter_pkl(in_src_dir / f'{input_name}-mask-indices.pkl', out_src_dir / f'{output_name}-mask-indices.pkl')
    # _filter_pkl(in_src_dir / f'{input_name}-res-ids.pkl', out_src_dir / f'{output_name}-res-ids.pkl')

    binarize_single_test_set(output_dir, output_name, fairseq_root=fairseq_root, pre_dicts_root=pre_dicts_root)
    shutil.rmtree(output_dir / 'tmp')
    

def merge_and_build(
        input_dir_1: Path, input_name_1: str,
        input_dir_2: Path, input_name_2: str,
        output_dir: Path, output_name: str = None, *,
        fairseq_root: Path, pre_dicts_root: Path,
        keep_duplicates: bool = False,
):
    """Merge two datasets and build."""

    in_src_dir_1 = input_dir_1 / 'src'
    in_src_dir_2 = input_dir_2 / 'src'
    if output_name is None:
        output_name = input_name_1 + '_' + input_name_2

    logging.info(f'Merging {in_src_dir_1}:{input_name_1} and {in_src_dir_2}:{input_name_2} to {output_dir}:{output_name}.')

    out_src_dir = output_dir / 'src'
    out_src_dir.mkdir(exist_ok=True, parents=True)

    # info.
    infos = []
    rm_indices_1, rm_indices_2 = [], []
    with csv_reader(in_src_dir_1 / f'{input_name_1}-info.csv', dict_reader=True) as reader:
        for i, row in enumerate(reader):
            if keep_duplicates:
                infos.append(row)
            else:
                if row not in infos:
                    infos.append(row)
                else:
                    rm_indices_1.append(i)
    with csv_reader(in_src_dir_2 / f'{input_name_2}-info.csv', dict_reader=True) as reader:
        for i, row in enumerate(reader):
            if keep_duplicates:
                infos.append(row)
            else:
                if row not in infos:
                    infos.append(row)
                else:
                    rm_indices_2.append(i)
        fieldnames = list(infos[0].keys())
        for fieldname in reader.fieldnames:
            if fieldname not in fieldnames:
                fieldnames.append(fieldname)
        with csv_writer(out_src_dir / f'{output_name}-info.csv', fieldnames=fieldnames) as writer:
            writer.writeheader()
            for row in infos:
                writer.writerow(row)
    
    # tg/m1.
    def _merge_text(in_filename_1: Path, in_filename_2: Path, out_filename: Path):
        with open(in_filename_1, 'r', encoding='utf-8') as f_in_1, \
                open(in_filename_2, 'r', encoding='utf-8') as f_in_2, \
                    open(out_filename, 'w', encoding='utf-8') as f_out:
            for i_, line in enumerate(f_in_1):
                if i_ in rm_indices_1:
                    continue
                f_out.write(line)
            for i_, line in enumerate(f_in_2):
                if i_ in rm_indices_2:
                    continue
                f_out.write(line)

    _merge_text(in_src_dir_1 / f'{input_name_1}.tg', in_src_dir_2 / f'{input_name_2}.tg', out_src_dir / f'{output_name}.tg')
    _merge_text(in_src_dir_1 / f'{input_name_1}.tg.orig', in_src_dir_2 / f'{input_name_2}.tg.orig', out_src_dir / f'{output_name}.tg.orig')
    _merge_text(in_src_dir_1 / f'{input_name_1}.m1', in_src_dir_2 / f'{input_name_2}.m1', out_src_dir / f'{output_name}.m1')
    _merge_text(in_src_dir_1 / f'{input_name_1}.m1.orig', in_src_dir_2 / f'{input_name_2}.m1.orig', out_src_dir / f'{output_name}.m1.orig')

    # coord/sites.
    def _merge_pkl(in_filename_1: Path, in_filename_2: Path, out_filename: Path):
        _data_1 = pickle_load(in_filename_1)
        _data_2 = pickle_load(in_filename_2)
        _new_data = {}
        new_i = 0
        for i_ in sorted(_data_1):
            if i_ in rm_indices_1:
                continue
            value = _data_1[i_]
            _new_data[new_i] = value
            new_i += 1
        for i_ in sorted(_data_2):
            if i_ in rm_indices_2:
                continue
            value = _data_2[i_]
            _new_data[new_i] = value
            new_i += 1
        pickle_dump(_new_data, out_filename)

    _merge_pkl(in_src_dir_1 / f'{input_name_1}-coordinates.pkl', in_src_dir_2 / f'{input_name_2}-coordinates.pkl', out_src_dir / f'{output_name}-coordinates.pkl')
    _merge_pkl(in_src_dir_1 / f'{input_name_1}-sites.pkl', in_src_dir_2 / f'{input_name_2}-sites.pkl', out_src_dir / f'{output_name}-sites.pkl')
    # _merge_pkl(in_src_dir_1 / f'{input_name_1}-mask-indices.pkl', in_src_dir_2 / f'{input_name_2}-mask-indices.pkl', out_src_dir / f'{output_name}-mask-indices.pkl')
    # _merge_pkl(in_src_dir_1 / f'{input_name_1}-res-ids.pkl', in_src_dir_2 / f'{input_name_2}-res-ids.pkl', out_src_dir / f'{output_name}-res-ids.pkl')
    
    
    binarize_single_test_set(output_dir, output_name, fairseq_root=fairseq_root, pre_dicts_root=pre_dicts_root)
    shutil.rmtree(output_dir / 'tmp')


def sample_and_build(
        input_dir: Path, input_name: str,
        output_dir: Path, output_name: str = None, *,
        fairseq_root: Path, pre_dicts_root: Path,
        data_id_list: Iterable[int] = None,
        num_sample: int = 100,
):
    """Filter some data and build."""

    in_src_dir = input_dir / 'src'
    if output_name is None:
        output_name = input_name

    with csv_reader(in_src_dir / f'{input_name}-info.csv', dict_reader=True) as reader:
        total_ids = range(len(list(reader)))
    logging.info(f'Convert {in_src_dir}:{input_name} to {output_dir}:{output_name}.')
    if data_id_list is not None:
        logging.info(f'Sample by data ID list.')
    else:
        logging.info(f'Randomly sample {num_sample} data records.')
        data_id_list = random.sample(total_ids, num_sample)

    out_src_dir = output_dir / 'src'
    out_src_dir.mkdir(exist_ok=True, parents=True)

    # info.
    with csv_reader(in_src_dir / f'{input_name}-info.csv', dict_reader=True) as reader:
        with csv_writer(out_src_dir / f'{output_name}-info.csv', fieldnames=reader.fieldnames) as writer:
            writer.writeheader()
            for i, row in enumerate(reader):
                if i in data_id_list:
                    writer.writerow(row)

    # tg/m1.
    def _sample_text(in_filename: Path, out_filename: Path):
        with open(in_filename, 'r', encoding='utf-8') as f_in, \
                open(out_filename, 'w', encoding='utf-8') as f_out:
            for i_, line in enumerate(f_in):
                if i_ in data_id_list:
                    f_out.write(line)

    _sample_text(in_src_dir / f'{input_name}.tg', out_src_dir / f'{output_name}.tg')
    _sample_text(in_src_dir / f'{input_name}.tg.orig', out_src_dir / f'{output_name}.tg.orig')
    _sample_text(in_src_dir / f'{input_name}.m1', out_src_dir / f'{output_name}.m1')
    _sample_text(in_src_dir / f'{input_name}.m1.orig', out_src_dir / f'{output_name}.m1.orig')

    # coord/sites.
    def _sample_pkl(in_filename: Path, out_filename: Path):
        _data = pickle_load(in_filename)
        _filtered_data = {}
        new_i = 0
        for i_ in sorted(_data):
            if i_ in data_id_list:
                value = _data[i_]
                _filtered_data[new_i] = value
                new_i += 1
        pickle_dump(_filtered_data, out_filename)

    _sample_pkl(in_src_dir / f'{input_name}-coordinates.pkl', out_src_dir / f'{output_name}-coordinates.pkl')
    _sample_pkl(in_src_dir / f'{input_name}-sites.pkl', out_src_dir / f'{output_name}-sites.pkl')
    # _sample_pkl(in_src_dir / f'{input_name}-mask-indices.pkl', out_src_dir / f'{output_name}-mask-indices.pkl')
    # _sample_pkl(in_src_dir / f'{input_name}-res-ids.pkl', out_src_dir / f'{output_name}-res-ids.pkl')


    binarize_single_test_set(output_dir, output_name, fairseq_root=fairseq_root, pre_dicts_root=pre_dicts_root)
    shutil.rmtree(output_dir / 'tmp')
