#%%
from rdkit import Chem, RDLogger
from collections import defaultdict
import numpy as np
import pandas as pd
import sys

RDLogger.DisableLog('rdApp.*')

fn = sys.argv[1]
out_fn = sys.argv[2]

all_cmpds = defaultdict(lambda: defaultdict(list))

def canonicalize(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


with open(fn, 'r', encoding='utf8') as fr:
    for line in fr:
        if line.startswith('H-'):
            segs = line.split('\t')
            k = int(segs[0].replace('H-', ''))
            smi = segs[2].strip().replace(' ', '')
            smi2 = canonicalize(smi)
            score = float(segs[1])
            if smi2 is None:
                continue
            all_cmpds[k][smi2].append(score)


results = {
    "test_id": [],
    "smiles": [],
    "nlogP": []
}

for idx in range(100):
    X = []
    for smi, score in all_cmpds[idx].items():
        s = np.mean(score)
        X.append((smi, s))
    sortedX = sorted(X, key=lambda x: x[1], reverse=True)
    for smi, s in sortedX:
        results["test_id"].append(idx)
        results["smiles"].append(smi)
        results["nlogP"].append(s)

pd.DataFrame.from_dict(results).to_csv(out_fn, index=False)