#%%
from Bio.PDB import PDBParser, MMCIFParser, PDBIO, Select
from glob import glob
import numpy as np
import os
import sys
import pandas as pd
import argparse

class CleanPDBSelect(Select):
    def accept_atom(self, atom):
        # Exclude water molecules (HOH) and heteroatoms
        if atom.parent.resname == 'HOH' or atom.parent.id[0] != ' ':
            return False
        # Exclude alternate locations
        if atom.altloc not in (' ', 'A'):
            return False
        return True

def get_ligand_center(file_path, ligand_name, chain_id):
    if file_path.endswith('.pdb'):
        parser = PDBParser()
    elif file_path.endswith('.cif'):
        parser = MMCIFParser()
    else:
        raise ValueError("Unsupported file format. Please provide a PDB or CIF file.")
    
    structure = parser.get_structure('structure', file_path)
    
    atom_coords = []
    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                for residue in chain:
                    if residue.resname == ligand_name:
                        for atom in residue:
                            atom_coords.append((atom.element, atom.get_coord()))
    
    if atom_coords:
        coords = np.array([coord for _, coord in atom_coords])
        center = np.mean(coords, axis=0)
        return center, atom_coords
    return None, None


def clean_pdb(input_pdb, output_pdb):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('structure', input_pdb)
    
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_pdb, CleanPDBSelect())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract the pocket region of a given PDB.')
    parser.add_argument('--pdbid', type=str, default=None, help='PDB ID')
    parser.add_argument('--pdb_folder', type=str, default='pdb_storage', help='Path to processed pdb files')
    parser.add_argument('--ligand_name', type=str, default=None, help='Name of the ligand')
    parser.add_argument('--chain_id', type=str, default=None, help='Chain of the ligand. You should select the chain id with [auth chain_id]')

    args = parser.parse_args()
    args.pdbid = args.pdbid.upper()

    STORAGE_FOLDER = args.pdb_folder
    pdb = args.pdbid
    ligand_name = args.ligand_name
    chain_id = args.chain_id

    FF = glob(STORAGE_FOLDER + '/*')
    allnames = [e.lower() for e in FF]
    if any(pdb.lower() in e for e in allnames):
        sys.exit(f"File {pdb} already exists in the storage folder.")

    os.system(f"wget -P {STORAGE_FOLDER} https://files.rcsb.org/download/{pdb}.pdb")
    os.system(f"wget -P {STORAGE_FOLDER} https://files.rcsb.org/download/{pdb}.cif")
    center, pocket = get_ligand_center(f'{STORAGE_FOLDER}/{pdb}.pdb', ligand_name, chain_id)
    clean_pdb(f'{STORAGE_FOLDER}/{pdb}.pdb', f'{STORAGE_FOLDER}/{pdb}.receptor.pdb')

    datadict = {
        'pdb_id': [pdb],
        'chain_id': [chain_id],
        'ligand_name': [ligand_name],
        'center_x': center[0],
        'center_y': center[1],
        'center_z': center[2],
    }

    datadict2 = {
        'pdb_id': [pdb],
        'center_x': center[0],
        'center_y': center[1],
        'center_z': center[2],
    }

    with open(f'{STORAGE_FOLDER}/index_{pdb}.csv', 'w') as fw:
        df = pd.DataFrame.from_dict(datadict)   
        df.to_csv(fw, index=False)

    with open(f'{STORAGE_FOLDER}/index_simple_{pdb}.csv', 'w') as fw:
        df = pd.DataFrame.from_dict(datadict2)   
        df.to_csv(fw, index=False)