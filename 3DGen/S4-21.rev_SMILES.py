from rdkit import Chem
from rdkit.Chem import rdmolops
from itertools import product
import re
import regex
import numpy as np

import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--max_aug', type=int, default=20, help='How many attempts to choose to augment the SMILES')
parser.add_argument('--smiles_file', type=str, default=None, help='Path to the SMILES file')
parser.add_argument('--output_file', type=str, default=None, help='output file of the SMILES')

args = parser.parse_args()

if args.output_file is None:
    args.output_file = args.smiles_file + '.rev'

with open(args.smiles_file, 'r', encoding='utf8') as fr:
    smiles_list = [e.strip() for e in fr]
    assert len(smiles_list) >= 1

# This code reverses a SMILES representation so that it ends with the specified connection point for further expansion. It will be utilized to facilitate TamGen-based fragment design.
template_smiles = smiles_list[0]

for e in ['Br', 'Cl', 'I']:
    if e not in template_smiles:
        special_element = e
        break

source_smiles = template_smiles.replace('*', special_element)

def replace_brackets(text, placeholder):
    pattern = r'\((?:[^()]|(?R))*\)'
    matches = regex.findall(pattern, text)
    for i, match in enumerate(matches):
        text = text.replace(match, placeholder + str(i), 1)
    return text, {f'{placeholder}{i}': match for i, match in enumerate(matches)}

def smi_tokenizer(smi, placeholder):
    pattern = "(" + placeholder + "\d+|\[[^\]]+]|Br?|Cl?|[A-Za-z][0-9]*|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    return tokens

def reverse_smiles(smiles, end_atom_symbol):
    mol = Chem.MolFromSmiles(smiles)

    end_atom_index = max([atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == end_atom_symbol])
    editable = Chem.EditableMol(mol)
    editable.AddAtom(Chem.Atom(0))
    editable.AddBond(end_atom_index, mol.GetNumAtoms(), order=Chem.rdchem.BondType.SINGLE)
    new_mol = editable.GetMol()

    new_smiles = Chem.MolToSmiles(new_mol, rootedAtAtom=mol.GetNumAtoms(), doRandom=True, isomericSmiles=True) #, isomericSmiles=True, canonical=False)
    new_smiles = new_smiles[1:]

    placeholder = 'X'
    replaced_smi, mapper_tab = replace_brackets(new_smiles, placeholder)
    tok_smi_list = smi_tokenizer(replaced_smi, placeholder)

    fixed_list = []

    for e in tok_smi_list:
        if e not in mapper_tab:
            fixed_list.append([e])
        else:
            e2= mapper_tab[e]
            fixed_list[-1].append(e2)

    reversed_list = []
    for e in fixed_list[::-1]:
        reversed_list += e

    r = ''.join(reversed_list)
    r = r.replace('@@', 'XX')
    r = r.replace('@', '@@')
    r = r.replace('XX', '@')
    return r


def cmp_mol(s1, s2):
    m = Chem.MolFromSmiles(s1)
    if m is None:
        return False
    s11 = Chem.MolToSmiles(m)
    m = Chem.MolFromSmiles(s2)
    if m is None:
        return False
    s22 = Chem.MolToSmiles(m)
    return s11 == s22

def generate_smiles(smiles_string):
    parts = smiles_string.replace("[C@H]", "[C@@H]").split("[C@@H]")
    combinations = product(*[["[C@H]", "[C@@H]"]] * (len(parts) - 1))

    # Generate all possible SMILES strings
    smiles_strings = []
    for combination in combinations:
        smiles = parts[0]
        for i in range(1, len(parts)):
            smiles += combination[i - 1] + parts[i]
        smiles_strings.append(smiles)

    return smiles_strings

unique_set = set()
for e in range(args.max_aug):
    rx = reverse_smiles(source_smiles, special_element)
    unique_set.add(rx)

unique_correct_set = set()
for e in unique_set:
    S = generate_smiles(e)
    for s in S:
        if cmp_mol(s, source_smiles):
            unique_correct_set.add(s)
            break
unique_correct_list = list(unique_correct_set)

with open(args.output_file, 'w', encoding='utf8') as fw:
    for e in unique_correct_list:
        print(e.replace(special_element, ''), file=fw)
