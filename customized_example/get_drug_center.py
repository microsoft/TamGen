#%%
from collections import defaultdict
import numpy as np
import pandas as pd

def process_line(line):
    segs = line.strip().split()
    idx = segs[6]
    ele = segs[2]
    x,y,z = segs[10], segs[11], segs[12]
    return idx, ele, x,y,z


def process_file(fname):
    ligand_lines = []
    with open(fname, "r") as fr:
        for line in fr:
            if line.strip().startswith("HETATM"):
                ligand_lines.append(line.strip())

    ligand_group = defaultdict(list)
    for line in ligand_lines:
        idx, ele, x,y,z = process_line(line)
        ligand_group[idx].append((ele, x,y,z))
    return ligand_group


def atom_filter(ligand_group, atom):
    pop_list = []
    for k, values in ligand_group.items():
        # if len(values) > 40:
        #     pop_list.append(k)
        #     continue
        B_in_ligand = False
        for v in values:
            if v[0] == atom:
                B_in_ligand = True
                break
        if not B_in_ligand:
            pop_list.append(k)

    for e in pop_list:
        ligand_group.pop(e)


def atom_filter_exactnum(ligand_group, atom, cnt):
    pop_list = []
    for k, values in ligand_group.items():
        if len(values) > 40:
            pop_list.append(k)
            continue
        if sum([v[0] == atom for v in values]) != cnt:
            pop_list.append(k)

    for e in pop_list:
        ligand_group.pop(e)


def get_center(ligand_group, excludeH=False):
    centers = []
    for k,values in ligand_group.items():
        x,y,z = [], [], []
        for v in values:
            if excludeH and v[0] == "H":
                continue
            x.append(float(v[1]))
            y.append(float(v[2]))
            z.append(float(v[3]))

        xmean = np.mean(x)
        ymean = np.mean(y)
        zmean = np.mean(z)

        centers.append((xmean, ymean, zmean))
    return centers

#%% 
# %%

pdbfn = r"7vh8"
ligand_group = process_file(f"{pdbfn}.cif")
atom_filter(ligand_group, "F")
centers = get_center(ligand_group)
centers_noH = get_center(ligand_group, excludeH=True)

# the no H version is very close to the H version, so i just use one

data_dict = {
    "pdb_id": [],
    "center_x": [],
    "center_y": [],
    "center_z": []
}

for ele in centers:
    data_dict['pdb_id'].append(pdbfn)
    data_dict['center_x'].append(ele[0])
    data_dict['center_y'].append(ele[1])
    data_dict['center_z'].append(ele[2])

df = pd.DataFrame.from_dict(data_dict)
df.to_csv(f"{pdbfn}_out.csv")

# %%
