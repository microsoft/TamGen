#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Prepare CrossDocked dataset.

Will output:
    Data pairs
    Protein sequences
    Uniprot References
    Coordinates
    Coordinate masks
    Near ligand masks
    Binding sites
    PDB ID list
"""

import csv
import pickle
from collections import namedtuple
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Sequence
import argparse
from os.path import dirname as parent

import numpy as np
import torch
from Bio.PDB import PDBParser
from Bio.PDB.Residue import Residue
from rdkit import Chem
from tqdm import tqdm
from fy_common_ext.io import copyfile_if_not_exist

from fairseq.molecule_utils.basic import smiles_utils as smu
from fairseq.molecule_utils.coordinate_utils.atom_positions import get_residue_average_position
from fairseq.molecule_utils.database.common_utils import aa_3to1
from fairseq.molecule_utils.external.fairseq_dataset_build_utils import binarize_single_test_set
import shutil

# Types.
DataEntry = Tuple[str, str]
DataEntryList = Sequence[DataEntry]
DatasetEntrySplit = Dict[str, DataEntryList]


def _get_dataset_split() -> DatasetEntrySplit:
    """Get dataset split.

    Train: 100000
    Test: 100
    """
    split_fn = CROSSDOCK_PATH / 'split_by_name.pt'
    data = torch.load(split_fn)
    return data


def _get_test_list() -> DataEntryList:
    """Get test list.

    Equal to dataset_split['test']."""
    test_list_fn = CROSSDOCK_PATH / 'test_list.tsv'
    with test_list_fn.open('r', encoding='utf-8') as f_test_list:
        reader = csv.reader(f_test_list, dialect=csv.excel_tab)
        next(reader)
        return list(reader)

_PARSER = PDBParser(QUIET=True)


ResidueInfo = namedtuple('ResidueInfo', 'chain_id res_id res_code pos')


def _process_one_data_entry(subset_name: str, index: int, pocket_fn: str, ligand_fn: str) -> Optional[Dict]:
    pocket_path = CROSSDOCK_PATH / 'crossdocked_pocket10' / pocket_fn
    ligand_path = CROSSDOCK_PATH / 'crossdocked_pocket10' / ligand_fn

    # Get PDB ID and Chain ID.
    pocket_words = pocket_fn.split('/')[1].split('_')
    pdb_id = pocket_words[0].lower()
    chain_id = pocket_words[1]
    assert len(pdb_id) == 4

    # Get ligand inchi.
    # No invalid SDF file in CrossDocked dataset.
    mol = Chem.MolFromMolFile(str(ligand_path))
    if mol is None:
        mol = Chem.MolFromMolFile(str(ligand_path), strictParsing=False)
        if mol is None:
            print(f'| {subset_name} subset | {index}: failed to load SDF file.')
            return None
    ligand_smiles = Chem.MolToSmiles(mol)
    ligand_inchi = Chem.MolToInchi(mol)

    # Another choice: use SBDD code.
    # pocket_dict = sbdd.PDBProtein(str(pocket_path)).to_dict_atom()

    # Get pocket sequence and coordinates.
    protein_structure = _PARSER.get_structure(pdb_id, str(pocket_path))
    # Some PDB file include multiple models.
    try:
        first_model = next(protein_structure.get_models())
    except StopIteration:
        print(f'| {subset_name} subset | {index}: no models found in structure.')
        return None

    real_chain_ids = set()

    # Sort residues by (chain_id, res_id)
    # The final sequence will across multiple chains.
    all_residues = []
    for residue in first_model.get_residues():  # type: Residue
        _, _, _this_chain_id, res_index = residue.get_full_id()
        res_type, res_id, insertion_code = res_index
        if res_type != ' ':
            continue

        real_chain_ids.add(_this_chain_id)
        res_name = residue.get_resname()
        res_code = aa_3to1(res_name)

        pos = get_residue_average_position(residue, dtype=np.float32, only_aa=True, center_of_gravity=True)

        residue_info = ResidueInfo(_this_chain_id, res_id, res_code, pos)

        all_residues.append(residue_info)
    all_residues.sort(key=lambda info: (info.chain_id, info.res_id))
    sequence = ''.join(info.res_code for info in all_residues)
    coordinate = np.stack([info.pos for info in all_residues])

    result = {
        'subset_name': subset_name,
        'index': index,
        'pdb_id': pdb_id,
        'chain_id': chain_id,
        'sequence': sequence,
        'real_chain_ids': sorted(real_chain_ids),
        'ligand_smiles': ligand_smiles,
        'ligand_inchi': ligand_inchi,
        'coordinate': coordinate,
        'pocket_fn': pocket_fn,
        'ligand_fn': ligand_fn,
    }

    return result


def _dump(subset_name: str, all_data: List[Optional[Dict]], fairseq_root: Path, pre_dicts_root: Path, max_len: int = 1023):
    valid_all_data = []
    missing = []
    for index, data in enumerate(all_data):
        if data is None:
            missing.append(index)
        else:
            valid_all_data.append(data)
    print(f'| {subset_name} subset | Missing: {len(missing)} / {len(all_data)} (first 5: {missing[:5]}).')
    all_data = valid_all_data
    
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    src_dir = OUTPUT_DIR / 'src'
    src_dir.mkdir(exist_ok=True)

    # Data pairs.
    pairs_fn = OUTPUT_DIR / f'{subset_name}-info.csv'
    with pairs_fn.open('w', encoding='utf-8') as f_pairs:
        writer = csv.writer(f_pairs)
        writer.writerow(['dataset', 'index', 'pdb_id', 'chain_id', 'ligand_inchi','real_chain_ids'])

        for data in all_data:
            writer.writerow([
                f'CrossDocked-{subset_name}', 
                data['index'], 
                data['pdb_id'], 
                data['chain_id'], 
                data['ligand_inchi'], 
                data['real_chain_ids'],
            ])
            
    # tg.
    tg_orig_fn = src_dir / f'{subset_name}.tg.orig'
    tg_fn = src_dir / f'{subset_name}.tg'
    with tg_orig_fn.open('w', encoding='utf-8') as f_tg_orig, tg_fn.open('w', encoding='utf-8') as f_tg:
        for data in all_data:
            print(data['sequence'], file=f_tg_orig)
            print(' '.join(data['sequence'][:max_len]), file=f_tg)
    
    # m1
    m1_orig_fn = src_dir / f'{subset_name}.m1.orig'
    m1_fn = src_dir / f'{subset_name}.m1'
    with m1_orig_fn.open('w', encoding='utf-8') as f_m1_orig, m1_fn.open('w', encoding='utf-8') as f_m1:
        for data in all_data:
            smi = data['ligand_smiles']
            tokenized_smi = smu.tokenize_smiles(smi)
            print(smi, file=f_m1_orig)
            print(tokenized_smi, file=f_m1)
            
    # Coordinates.
    coord_fn = OUTPUT_DIR / f'{subset_name}-coordinates.orig.pkl'
    out_data = {}
    for new_index, data in enumerate(all_data):
        out_data[new_index] = data['coordinate']
    with coord_fn.open('wb') as f_coord:
        pickle.dump(out_data, f_coord)
        
    # Coordinate truncation.
    coord_trunc_fn = OUTPUT_DIR / f'{subset_name}-coordinates.pkl'
    truncated_coord = {
        new_index: data['coordinate'][:max_len]
        for new_index, data in enumerate(all_data)
    }
    with coord_trunc_fn.open('wb') as f_coord_trunc:
        pickle.dump(truncated_coord, f_coord_trunc)
        
    binarize_single_test_set(OUTPUT_DIR, subset_name, fairseq_root=fairseq_root, pre_dicts_root=pre_dicts_root)
    shutil.rmtree(OUTPUT_DIR / 'tmp')
    # Copy test structure files.
    structure_file_dir = OUTPUT_DIR / f'structure-files-{subset_name}'
    structure_file_dir.mkdir(exist_ok=True)
    for new_index, data in enumerate(tqdm(all_data)):
        copyfile_if_not_exist(
            CROSSDOCK_PATH / 'crossdocked_pocket10' / data['pocket_fn'], structure_file_dir / f'{new_index}-protein.pdb', quiet=True)
        copyfile_if_not_exist(
            CROSSDOCK_PATH / 'crossdocked_pocket10' / data['ligand_fn'], structure_file_dir / f'{new_index}-ligand.sdf', quiet=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('crossdocked_path', type=Path, default=Path(r'data'))
    parser.add_argument('-o', '--output-dir', type=Path, default=Path(r'dataset/crossdocked'))
    args = parser.parse_args()
    
    global CROSSDOCK_PATH, OUTPUT_DIR
    CROSSDOCK_PATH = args.crossdocked_path
    OUTPUT_DIR = args.output_dir
    
    smu.disable_rdkit_log()

    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    dataset_split = _get_dataset_split()

    for subset_name in ['train', 'test']:
        structure_file_dir = OUTPUT_DIR / f'structure-files-{subset_name}'
        structure_file_dir.mkdir(exist_ok=True)
        
        # make the last 100 samples in the training set as validation set
        subset_entries = dataset_split[subset_name]
        if subset_name == 'train':
            train_entries = subset_entries[:-100]
            valid_entries = subset_entries[-100:]

            all_train_data = []
            for index, entry in enumerate(tqdm(train_entries)):
                data = _process_one_data_entry(subset_name, index, *entry)
                all_train_data.append(data)
            _dump('train', all_train_data,
                fairseq_root=Path(parent(parent(parent(__file__)))),
                pre_dicts_root=Path(parent(parent(parent(__file__)))) / 'dict',
                max_len=1023)

            all_valid_data = []
            for index, entry in enumerate(tqdm(valid_entries)):
                data = _process_one_data_entry('valid', index, *entry)
                all_valid_data.append(data)
            _dump('valid', all_valid_data,
                fairseq_root=Path(parent(parent(parent(__file__)))),
                pre_dicts_root=Path(parent(parent(parent(__file__)))) / 'dict',
                max_len=1023)

        else:
            all_test_data = []
            for index, entry in enumerate(tqdm(subset_entries)):
                data = _process_one_data_entry(subset_name, index, *entry)
                all_test_data.append(data)
            _dump('test', all_test_data,
                fairseq_root=Path(parent(parent(parent(__file__)))),
                pre_dicts_root=Path(parent(parent(parent(__file__)))) / 'dict',
                max_len=1023)


if __name__ == '__main__':
    main()
    