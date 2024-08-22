<img src="assets/TamGen-logo.png" width=100%> 

# TamGen

Target-aware Molecule Generation for Drug Design Using a Chemical Language Model

# Introduction
We present TamGen, a generative method for structure-based drug design. Given a protein binding site, TamGen generates small molecule compounds optimized to bind to that specific pocket.



The code is based on: [fairseq-v0.8.0](https://github.com/facebookresearch/fairseq)



# Installation

```bash
conda create -n TamGen python=3.9
conda activate TamGen

bash setup_env.sh
```

# Dataset

## Build training data for CrossDocked dataset
Please refer to the [README](data/README.md) in the folder `data`

## Build customized dataset

You can build your customized dataset through the following methods:

1. Build customized dataset based on pdb ids using the center coordinates of the binding site of each pdb.

   ```bash
   python scripts/build_data/prepare_pdb_ids_center.py ${PDB_ID_LIST} ${DATASET_NAME} -o ${OUTPUT_PATH} -t ${THRESHOLD}
   ```

   - `PDB_ID_LIST` format: CSV format with the following columns: `pdb_id,center_x,center_y,center_z,[uniprot_id]`. `[uniprot_id]` is optional.

   - `DATASET_NAME`: You could specify it by yourselv. The simplest way is to set it as `test`. 
   - `OUTPUT_PATH`:  The output path of the processed data.
   - `THRESHOLD`: The radius of the pocket region whose center is `center_x,center_y,center_z`.

2. Build customized dataset based on pdb ids, the script will automatically find the binding sites according to the ligands in the structure file.

   ```bash
   python scripts/build_data/prepare_pdb_ids.py ${PDB_ID_LIST} ${DATASET_NAME} -o ${OUTPUT_PATH} -t ${threshold}
   ```

   - `PDB_ID_LIST` format: CSV format with columns `pdb_id,[ligand_inchi,uniprot_id]`, where `[]` means optional.
   - `THRESHOLD`: A residue $r$ is considered part of the pocket region, if any atom in $r$ lies within THRESHOLD angstroms of a ligand atom. For a given `pdb_id`, its associated ligands can be found in [database/PdbCCD](database/PdbCCD).
   - The remaining parameters are the same as those in method 1.


3. Build customized dataset based on pdb ids using the center coordinates of the binding site of each pdb, and add the provided scaffold to each center

   ```bash
   python scripts/build_data/prepare_pdb_ids_center_scaffold.py ${PDB_ID_LIST} ${DATASET_NAME} -o ${OUTPUT_PATH} -t ${THRESHOLD} --scaffold-file ${SCAFFOLD_FILE}
   ```

   - `SCAFFOLD_FILE`:  It contains molecular scaffolds that will be incorporated into the processed database. These scaffolds serve as structural templates for subsequent conditional generation of new molecules.
   - The remaining parameters are the same as those in method 1.


   For customized pdb strcuture files, you can put your structure files to the `--pdb-path` folder, and in the `PDB_ID_LIST` csv file, put the filenames in the `pdb_id` column.

   We provide an example about how to build and use customized data in [customized_example](./customized_example).



# Model
The checkpoint can be found in the provided url from the paper, which is `TamGen.ckpt.tar.gz `. You should run `tar xzf TamGen.ckpt.tar.gz` to uncompress it. After uncompression, you will get two folders: `checkpoints` and `gpt_model`. Please place them under the folder `TamGen/`. The structures of the two folders are shown below:
```bash
checkpoints/
├── README.MD
├── crossdock_pdb_A10
│   └── checkpoint_best.pt
└── crossdocked_model
    └── checkpoint_best.pt

gpt_model/
├── checkpoint_best.pt
└── dict.txt
```


# Run scripts

## Training
```bash
# train a new model
bash scripts/train.sh -D ${DATA_PATH} --savedir ${SAVED_MODEL_PATH}
```

For example, one can run `bash scripts/train.sh -D data/crossdocked/bin/ --savedir crossdock_train --fp16` to train models.

## Inference
One can refer to `scripts/generate.sh` for running inference code.

We provide an example by running `bash scripts/example_inference.sh`

# Demo

We provide a demo at `interactive_decode.ipynb`

In the first cell of the demo 
```
from TamGen_Demo import TamGenDemo, prepare_pdb_data
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

worker = TamGenDemo(
    data="./TamGen_Demo_Data",
    ckpt="checkpoints/crossdock_pdb_A10/checkpoint_best.pt"
)
```

- Specify the GPU id
- Download the checkpoint and place it into "checkpoints/crossdock_pdb_A10/checkpoint_best.pt" or your specificied position
- Download the pre-trained GPT model and put it into the folder `gpt_model`
