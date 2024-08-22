from fy_common_ext.io import pickle_load, pickle_dump, csv_writer, csv_reader
from tqdm import tqdm
import numpy as np
from fairseq.data import indexed_dataset
import sys

binfolder = sys.argv[1]

def dumper(coord_fn, subset_name, out_folder):
    coord_data = pickle_load(coord_fn)
    coord_bin_fn = f'{out_folder}/{subset_name}.tg-m1.tg.coord'
    indexed_dataset.binarize_data(coord_data, str(coord_bin_fn), dtype=np.float32, dim=(3,))
    print(f'| Binarize coordinates {coord_fn} to {coord_bin_fn}.')

for subset in ['train', 'valid', 'test']:
    dumper(f'{binfolder}/{subset}-coordinates.pkl', subset, binfolder)