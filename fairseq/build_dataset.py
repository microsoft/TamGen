import random
from pathlib import Path

import fairseq
from fairseq.molecule_utils.external.fairseq_dataset_build_utils import (
    process_one_pdb_given_center_coord,
    dump_center_data_ligand,
)

random.seed(144169)

def build_dataset(
	pdb_id: str,
	ligand_center: list,
	smiles: list,
	work_dir: str,
	output_dir: str,
):
	pdb_info = {
		'pdb_id': pdb_id,
		'center_x': ligand_center[0],
		'center_y': ligand_center[1],
		'center_z': ligand_center[2],
	}

	data = process_one_pdb_given_center_coord(
		0, pdb_info, threshold=10, pdb_mmcif_path=Path(work_dir),
	)

	dump_center_data_ligand(
		[data],
		"test",
		output_dir=Path(output_dir),
		fairseq_root=Path(fairseq.__path__[0]).parent,
		pre_dicts_root=Path(fairseq.__path__[0]).parent / "dict",
		max_len=1023,
		ligand_list=smiles,
	)
