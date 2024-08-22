#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Basic molecule utils."""

import logging
import re
from typing import Union, List

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdmolfiles
from rdkit.Chem.rdchem import Mol as MolType
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol

__all__ = [
    'SMI_PATTERN', 'tokenize_smiles', 'tokenize_smiles_list',
    'canonicalize_smiles',
    'smi2mol', 'inchi2mol',
    'smi2inchi', 'inchi2smi',
    'smi2pdb',
    'pdb_file_to_sdf_file',
    'molecular_weight', 'num_atoms', 'num_rings', 'get_scaffold',
    'num_H_acceptors',
    'num_H_donors',
    'num_rot_bonds',
    'TPSA',
    'disable_rdkit_log',
    '_to_mol',
]

SMI_PATTERN = re.compile(
    r"(\[[^]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>>?|\*|\$|%[0-9]{2}|[0-9])")


def tokenize_smiles(smiles: str) -> str:
    return ' '.join(SMI_PATTERN.findall(smiles))


def tokenize_smiles_list(smiles: str) -> List[str]:
    return SMI_PATTERN.findall(smiles)


def canonicalize_smiles(smiles: str, throw: bool = False) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        if throw:
            raise ValueError(f'Cannot convert SMILES {smiles}.')
        logging.warning(f'Cannot convert SMILES {smiles}.')
        return None
    return Chem.MolToSmiles(mol, canonical=True)


smi2mol = Chem.MolFromSmiles
inchi2mol = Chem.MolFromInchi


def _to_mol(mol: Union[str, MolType]) -> MolType:
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    return mol


def molecular_weight(mol: Union[str, MolType]) -> float:
    mol = _to_mol(mol)
    return Descriptors.MolWt(mol)


def num_atoms(mol: Union[str, MolType], only_heavy=True) -> int:
    mol = _to_mol(mol)
    return mol.GetNumAtoms(onlyExplicit=only_heavy)


def num_rings(mol: Union[str, MolType]) -> int:
    mol = _to_mol(mol)
    return mol.GetRingInfo().NumRings()

def num_H_acceptors(mol: Union[str, MolType]) -> int:
    mol = _to_mol(mol)
    return Descriptors.NumHAcceptors(mol)

def num_H_donors(mol: Union[str, MolType]) -> int:
    mol = _to_mol(mol)
    return Descriptors.NumHDonors(mol)

def num_rot_bonds(mol: Union[str, MolType]) -> int:
    mol = _to_mol(mol)
    return Descriptors.NumRotatableBonds(mol)

def TPSA(mol: Union[str, MolType], includeSandP=True) -> float:
    mol = _to_mol(mol)
    return Descriptors.TPSA(mol, includeSandP=includeSandP)

def num_H_acceptors(mol: Union[str, MolType]) -> int:
    mol = _to_mol(mol)
    return Descriptors.NumHAcceptors(mol)


def num_H_donors(mol: Union[str, MolType]) -> int:
    mol = _to_mol(mol)
    return Descriptors.NumHDonors(mol)


def num_rot_bonds(mol: Union[str, MolType]) -> int:
    mol = _to_mol(mol)
    return Descriptors.NumRotatableBonds(mol)


def TPSA(mol: Union[str, MolType], includeSandP=True) -> float:
    mol = _to_mol(mol)
    return Descriptors.TPSA(mol, includeSandP=includeSandP)


def get_scaffold(mol: Union[str, MolType]) -> str:
    mol = _to_mol(mol)
    return Chem.MolFromSmiles(GetScaffoldForMol(mol))


def inchi2smi(inchi: str) -> str:
    mol = Chem.MolFromInchi(inchi)
    if mol is None:
        mol = Chem.MolFromInchi(inchi, sanitize=False)
        if mol is None:
            raise ValueError(f'Cannot convert InChI {inchi}.')
    return Chem.MolToSmiles(mol, canonical=True)


def smi2inchi(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f'Cannot convert SMILES {smiles}.')
    return Chem.MolToInchi(mol)


def _smi2mol_structured(smiles, compute_coord, optimize):
    if optimize == 'UFF':
        optimize_func = AllChem.UFFOptimizeMolecule
    elif optimize == 'MMFF':
        optimize_func = AllChem.MMFFOptimizeMolecule
    elif optimize == 'none':
        optimize_func = lambda _: None
    else:
        raise ValueError(f'Unknown optimize type {optimize}.')

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f'Cannot convert SMILES {smiles}.')
    if compute_coord:
        emb_success = AllChem.EmbedMolecule(mol)
        if emb_success == -1:
            logging.warning(f'Cannot embed molecule {smiles}, use 2D coordinate as a fallback solution.')
            AllChem.Compute2DCoords(mol)
        try:
            optimize_func(mol)
        except RuntimeError as e:
            logging.warning(f'Failed to run {optimize} optimization on {smiles}, (error: {e}).')
    return mol


def smi2pdb(smiles: str, compute_coord=True, optimize='UFF') -> str:
    """Convert SMILES into PDB string."""
    mol = _smi2mol_structured(smiles, compute_coord, optimize)
    pdb_str = Chem.MolToPDBBlock(mol)
    return pdb_str


def pdb_file_to_sdf_file(pdb_filename, sdf_filename):
    """Convert PDB file to SDF file."""
    mol = rdmolfiles.MolFromPDBFile(str(pdb_filename))
    if mol is None:
        mol = rdmolfiles.MolFromPDBFile(str(pdb_filename), sanitize=False)
        if mol is None:
            raise ValueError(f'Cannot load PDB file {pdb_filename}.')
    writer = rdmolfiles.SDWriter(str(sdf_filename))
    try:
        writer.write(mol)
    finally:
        writer.close()



def is_same_mol(smi1, smi2):
    mol1 = Chem.MolFromSmiles(smi1)
    if mol1 is None:
        logging.error(f'Invalid input smiles: {smi1}.')
        return False
    mol2 = Chem.MolFromSmiles(smi2)
    if mol2 is None:
        logging.error(f'Invalid input smiles: {smi2}.')
        return False
    canonical_smi1 = Chem.MolToSmiles(mol1)
    canonical_smi2 = Chem.MolToSmiles(mol2)
    return canonical_smi1 == canonical_smi2

def smiles_variants(mol: Union[str, MolType], n: int = 10000000) -> List[str]:
    """Get SMILES variants (with different root atoms).

    Args:
        mol:
        n:

    Returns:

    """
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    num_atoms = mol.GetNumAtoms()
    if n > num_atoms:
        n = num_atoms
    atom_range = np.linspace(0, num_atoms - 1, n, dtype=int)

    variants = []
    for i in atom_range:
        try:
            smiles = Chem.MolToSmiles(mol, rootedAtAtom=int(i))
        except RuntimeError as e:
            print(f'| WARNING: Generate variant {i} of molecule failed, cause: {e}')
        else:
            variants.append(smiles)
    return variants


def disable_rdkit_log():
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
