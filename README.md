<img src="assets/TamGen-logo.png" width=100%> 

# TamGen

TamGen: Target-aware Molecule Generation for Drug Design Using a Chemical Language Model

# Introduction

This is the implementation of the paper `TamGen: Target-aware Molecule Generation for Drug Design Using a Chemical Language Model`. 

Our implementation is built on [fairseq-v0.8.0](https://github.com/facebookresearch/fairseq)



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
The checkpoint can be found in `https://doi.org/10.5281/zenodo.13751391`. Please download `checkpoints.zip` & `gpt_model.zip` and uncompress them. After that, you will get two folders: `checkpoints` and `gpt_model`. Please place them under the folder `TamGen/`. The structures of the two folders are shown below:
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


## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.

