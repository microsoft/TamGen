# Customized example [Upgraded]

This document provides a customized example of how to generate compounds for a specific binding pocket. We present two key examples:

- `Full Molecule Generation`: This is the primary focus of the Nature Communications paper.

- `Fragment Generation`: This process designs target-aware fragments, detailed in our technical blog. [Read more here.](https://www.microsoft.com/en-us/research/blog/accelerating-drug-discovery-with-tamgen-a-generative-ai-approach-to-target-aware-molecule-generation/)


## Full molecule generation

We choose the pdb `7vh8` as an example.

### Steps

1. Go to folder `3DGen`
2. `python S1.extract_pocket.py` to extract the pocket regions. 
3. If you plan to run conditiona generation, plesase kindly run `python S2.augment_smiles.py` to augment the SMILES. This is used for conditional optimization. For unconditional generation, you do not need to use it. We also prepare a common scaffold list from the training dataset, and you can consider using them. They are in `misc/scaffold_smiles_top100.txt`
4. Run `bash S3.build_bindata.sh` to create binary files.
5. Go back to the TamGen repo folder, and run `time bash 3DGen/S4-1.generator_full.sh`. You should get the results in `3DGen/7vh8-results`.
6. Go to `3DGen` folder;
7. Run `python S5.post_process.py` to summarize the output;
8. Dock the generated molecules using AutoDock Vina. Users could choose other software as well. Here, we will use Vina as an example.

   8.1 Installalation: 
   ```
    pip install meeko==0.1.dev3 scipy pdb2pqr vina==1.2.2
    python -m pip install git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3
    pip install openbabel-wheel 
   ```

   8.2 Docking using Autodock Vina: `python S6.dock_utils.py`

   8.3 Summarize the docking results by using `S7.aggregate_docking_pose.py`



## Fragment generation

### Steps:

1. Step 1 to 3 are the same as in `Full molecule generation`

2. Use `S4-21.rev_SMILES.py` to reverse the fragment you want to retain, ensuring it ends at the position for further growth of the SMILES.

3. Use `S4-22.generator_fragment.sh` to generate fragments.

The remaining steps are identical to those in `Full Molecule Generation`.