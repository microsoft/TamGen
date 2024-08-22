# Customized example

We provide a customized example about how to generate compounds for a given pocket.

We choose the pdb `7vh8` as an example.


## Steps

1. Go to folder `customized_example`

2. Download `7VH8.cif` file into this folder. You can use `curl https://files.rcsb.org/download/7VH8.cif > 7vh8.cif`

3. Use `get_drug_center.py` to get the ligand center. You can open this python script using vscode, and it will look like a jupyter notebook (due to the usage of `#%%`). You should get `7vh8_out.csv`.

4. Use `augment_smiles.py` to augment the SMILES of the reference ligand. This means that, given a SMILES string, we use different ways to represent it. Because we use sequence-based method, we can use it to increase diversity. The augmented SMILES strings are in `seed_cmpd_7vh8.txt`.

This is used for conditional optimization. For unconditional generation, you do not need to use it. We also prepare a common scaffold list from the training dataset, and you can consider using them. They are in `misc/scaffold_smiles_top100.txt`

5. Build binary file `build_bindata.sh`. You should get a folder `7vh8-bin/`

6. Go back to the TamGen repo folder, and run `time bash customized_example/generator-7vh8.sh`. You should get the results in `customized_example/7vh8-results`. It requires about 30 minutes, so using `tmux` is a better choice.

7. Go to `customized_example`. Run `post_process_7vh8.py` to post process the outputs. You should get one `pkl` and one `_flatten.tsv`. `pkl` file has all the information, while the `tsv` file has two columns: first column is the SMILES, and the second column is the average normalized log probability. 

For you reference, we put several intermediate files in the `reference_results` folder.
