#%%
from rdkit import Chem
from rdkit.Chem import rdmolops  
import re  
import regex
import numpy as np
from rdkit.Chem import rdDepictor  
import random  

pdbid = r"7vh8"

# in all_lines, it is the smiles of the ligand of 7vh8
# I copy it from `https://pubchem.ncbi.nlm.nih.gov/compound/156621364`
all_lines = [
    'CC1([C@@H]2[C@H]1[C@H](N(C2)C(=O)[C@H](C(C)(C)C)NC(=O)C(F)(F)F)C(=O)N[C@@H](C[C@@H]3CCNC3=O)C=N)C'
]

mols = [Chem.MolFromSmiles(e) for e in all_lines]
newsmi = [Chem.MolToSmiles(e) for e in mols]

augmented_smiles = set()

def augment(smiles):
    mol = Chem.MolFromSmiles(smiles)
    smi_can = Chem.MolToSmiles(mol)
    augmented_smiles.add(smi_can)
    augmented_smiles.add(smiles)
    remapping = list(range(mol.GetNumAtoms()))  
    for i in range(20):
        random.shuffle(remapping) 
        new_mol = Chem.RenumberAtoms(mol, remapping) 
        new_smiles = Chem.MolToSmiles(new_mol,isomericSmiles=True, canonical=False)  
        m = Chem.MolFromSmiles(new_smiles)
        if m is None:
            continue
        x2 = Chem.MolToSmiles(m)
        if smi_can == x2:
            augmented_smiles.add(new_smiles)

for e in newsmi:
    augment(e)


with open(f'seed_cmpd_{pdbid}.txt', 'w', encoding='utf8') as fw:
    for e in augmented_smiles:
        print(e,file=fw)
# %%
