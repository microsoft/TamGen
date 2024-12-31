from glob import glob
import pandas as pd
import os
import re
import argparse

parser = argparse.ArgumentParser(description='Process docking pose and SMILES file.')
parser.add_argument('--docking_pose_folder', type=str, default='/tmp/dockingpose/', help='Path to the docking pose folder')
parser.add_argument('--results_prefix', type=str, default='results', help='Prefix for the results folder')
parser.add_argument('--pdbid', type=str, default='7vh8', help='PDB ID')
parser.add_argument('--smiles_file', type=str, default='7vh8_vae_flatten.tsv', help='Path to the SMILES file')

args = parser.parse_args()

docking_pose_folder = args.docking_pose_folder
results_prefix = args.results_prefix
pdbid = args.pdbid
smiles_file = args.smiles_file


def get_ids(filename):
    suffix = filename.split('/')[-1]
    smi_id = int(suffix.split('_')[0])
    conf_id = int(suffix.split('_')[1].split('.')[0])
    return smi_id, conf_id

def get_vina_score(filename):
    with open(filename, 'r') as fr:
        all_lines = [e.strip() for e in fr]

    m = re.search(r"REMARK VINA RESULT:[\s]+(?P<vina>[\d\.\-]+)", all_lines[1])
    if m:
        vina = float(m.group('vina'))
    else:
        vina = None
    return vina

directory = results_prefix + '/' + pdbid    
if not os.path.exists(directory):
    os.makedirs(directory)

FF = glob(docking_pose_folder + '/*')

resultsdict = {}
for fn in FF:
    smi_id, conf_id = get_ids(fn)
    vina = get_vina_score(fn)
    if smi_id not in resultsdict:
        resultsdict[smi_id] = []
    resultsdict[smi_id].append((conf_id, vina))

for v in resultsdict.values():
    v.sort(key=lambda x: x[1])

with open(smiles_file, 'r') as fr:
    all_smiles = [e.strip() for e in fr]

all_records = []

for smi_id, line in enumerate(all_smiles):
    segs = line.strip().split('\t')
    smi = segs[0]   
    prob = segs[1]
    if smi_id not in resultsdict:
        continue
    conf_id, vina = resultsdict[smi_id][0]
    all_records.append((smi_id, smi, prob, vina, conf_id))

all_records.sort(key=lambda x: x[3])

datadict = {
    'idx': [],
    'SMILES': [],
    'prob': [],
    'vina': [],
}

for e in all_records:
    datadict['idx'].append(e[0])
    datadict['SMILES'].append(e[1])
    datadict['prob'].append(e[2])
    datadict['vina'].append(e[3])
    fn = f"{docking_pose_folder}/{e[0]}_{e[4]}.pdbqt"
    outfn = f"{directory}/{e[0]}.pdb"
    os.system(f"obabel -ipdbqt {fn} -opdb -O {outfn}")

df = pd.DataFrame.from_dict(datadict)
df.to_csv(directory + '/statistics.csv')