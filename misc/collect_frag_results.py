#%%
from rdkit import Chem
from collections import defaultdict
from glob import glob
import re
from tqdm import tqdm
from rdkit import RDLogger                                                                                                                                                               

RDLogger.DisableLog('rdApp.*')  

def reconstruct_molecule_from_fragments(fragments):  
    reconstructed_mol = Chem.RWMol()  
    combined_mol = fragments[0]  
    for fragment in fragments[1:]:  
        combined_mol = Chem.CombineMols(combined_mol, fragment)  
    
    reconstructed_mol = Chem.RWMol(combined_mol)  
    
    dummy_atoms = []  
    for atom_idx, atom in enumerate(reconstructed_mol.GetAtoms()):  
        if atom.GetAtomicNum() == 0:  
            neighbors = [x.GetIdx() for x in atom.GetNeighbors()]  
            dummy_atoms.append((atom_idx, atom.GetAtomMapNum(), neighbors[0] ))  
  
    # Sort dummy atoms by atom map number  
    dummy_atoms.sort(key=lambda x: x[1])  
  
    # Connect the fragments using the dummy atoms and remove them  
    for i in range(0, len(dummy_atoms), 2):  
        atom_idx1, map_num1, neighbor1 = dummy_atoms[i]  
        atom_idx2, map_num2, neighbor2 = dummy_atoms[i + 1]  
  
        if map_num1 == map_num2:  
            # Add a bond between the atoms connected to the dummy atoms  
            reconstructed_mol.AddBond(neighbor1, neighbor2, Chem.rdchem.BondType.SINGLE)  
  
    dummy_atoms = [e[0] for e in dummy_atoms]
    dummy_atoms = sorted(dummy_atoms, reverse=True)
    for atomidx in dummy_atoms: 
        reconstructed_mol.RemoveAtom(atomidx)  
  
    # Remove atom map numbers from the reconstructed molecule  
    for atom in reconstructed_mol.GetAtoms():  
        atom.SetAtomMapNum(0)  
  
    return reconstructed_mol

def check(line):
    s1 = line.count("[start-of-condition]")
    s2 = line.count("[cond-generation]")
    if s1 !=1 and s2 != 1:
        return -1
   
    line = line.split("\t")[-1].replace(" ", "")
    all_symbols = re.findall(r"\[\*\:[\d]+\]", line)
    cnt = defaultdict(int)
    for e in all_symbols:
        cnt[e] += 1
    
    for k,v in cnt.items():
        if v != 2:
            return 0
    
    return 1


def summarize(file_pattern):
    FF = glob(file_pattern)
    merged_mols_set = set()
    remaining_smiles = []

    for fn in FF:
        with open(fn) as fr:
            for e in fr:
                if e.startswith("H-") and check(e.strip()) == 1:
                    curr = e.strip().split("\t")[-1]
                    curr = curr.replace("[start-of-condition]", "").strip()
                    curr = curr.replace("[cond-generation]", ".")
                    curr = curr.replace(" ", "").strip()
                    remaining_smiles.append(curr)


    for smi in remaining_smiles:
        segs = smi.split(".")
        try:
            fragments = [Chem.MolFromSmiles(s) for s in segs]
            t = reconstruct_molecule_from_fragments(fragments)
            s = Chem.MolToSmiles(t)
            if "." in s:
                continue
            merged_mols_set.add(s)
        except:
            continue

    return merged_mols_set


FF = glob("GHDDI_frag_results_v0/*")

all_gen_mols = set()
for file_pattern in tqdm(FF,total=len(FF)):
    s = summarize(file_pattern)
    all_gen_mols |= s
    

#%%
desired_pattern_smarts = [
    "c1cccc2ccccc12",
    "C(=O)[NH]",
    "[N+]"
]

desired_pattern = []
for pat in desired_pattern_smarts:
    m = Chem.MolFromSmarts(pat)
    desired_pattern.append(m)

mols_with_matched_pattern = []
for mol in all_gen_mols:
    flag = 0
    mol = Chem.MolFromSmiles(mol)
    for pat in desired_pattern:
        t = mol.GetSubstructMatches(pat)
        flag += int(len(t) >= 1)
    mols_with_matched_pattern.append((mol, flag))

sorted_mols = sorted(mols_with_matched_pattern, key=lambda e: e[-1], reverse=True)

#%%
fn = r"GHDDI_fragBased_train_Oct31.txt"
with open(fn, "w") as fw:
    for e in sorted_mols:
        s = Chem.MolToSmiles(e[0])
        print(f"{s}\t{e[1]}", file=fw)