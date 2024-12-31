import contextlib
import logging
import os
import os.path as op
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple
import threading

import AutoDockTools
from meeko import MoleculePreparation, obutils
from openbabel import openbabel as ob
from openbabel import pybel
from rdkit import Chem
from rdkit.Chem import AllChem
from vina import Vina
from tqdm import tqdm
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Process docking pose and SMILES file.')
parser.add_argument('--docking_pose_folder', type=str, default='/tmp/tmpdockpose/', help='Path to the docking pose folder')
parser.add_argument('--pdbid', type=str, default='7vh8', help='PDB ID')
parser.add_argument('--pdb_folder', type=str, default='pdb_storage', help='Path to processed pdb files')
parser.add_argument('--smiles_file', type=str, default='7vh8_vae_flatten.tsv', help='Path to the SMILES file')
parser.add_argument('--num_ligand_confs', type=int, default=5, help='Number of conformations for docking')

args = parser.parse_args()
args.pdbid = args.pdbid.upper()

@contextlib.contextmanager
def timing(msg: str):
    logging.info("Started %s", msg)
    tic = time.time()
    yield
    toc = time.time()
    logging.info("Finished %s in %.3f seconds", msg, toc - tic)
    
def supress_stdout(func):
    def wrapper(*a, **ka):
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull):
                return func(*a, **ka)

    return wrapper


class PrepProt(object):
    def __init__(self, prot_file):
        self.prot = prot_file
        self.file_format = self.detect_format(prot_file)

    def detect_format(self, prot_file):
        # Determine if the file is in PDB or CIF format based on the extension
        if prot_file.endswith('.pdb'):
            return 'pdb'
        elif prot_file.endswith('.cif'):
            return 'cif'
        else:
            raise ValueError("Unsupported file format. Only .pdb and .cif are supported.")

    def del_water(self, dry_file):  # optional
        if self.file_format == 'pdb':
            self.del_water_pdb(dry_file)
        elif self.file_format == 'cif':
            self.del_water_cif(dry_file)
        self.prot = dry_file

    def del_water_pdb(self, dry_pdb_file):
        with open(self.prot) as f:
            lines = [
                l
                for l in f.readlines()
                if l.startswith("ATOM") or l.startswith("HETATM")
            ]
            dry_lines = [l for l in lines if "HOH" not in l]

        with open(dry_pdb_file, "w") as f:
            f.write("".join(dry_lines))
    
    def del_water_cif(self, dry_cif_file):
        with open(self.prot) as f:
            lines = f.readlines()
            dry_lines = [l for l in lines if "HOH" not in l]
            
        with open(dry_cif_file, "w") as f:
            f.write("".join(dry_lines))

    def addH(self, prot_pqr):  # call pdb2pqr
        self.prot_pqr = prot_pqr
        subprocess.Popen(
            ["pdb2pqr30", "--ff=AMBER", self.prot, self.prot_pqr],
            stderr=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
        ).communicate()

    def get_pdbqt(self, prot_pdbqt):
        prepare_receptor = os.path.join(
            AutoDockTools.__path__[0], "Utilities24/prepare_receptor4.py"
        )
        subprocess.Popen(
            ["python3", prepare_receptor, "-r", self.prot_pqr if self.file_format == "pdb" else self.prot, "-o", prot_pdbqt],
            stderr=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
        ).communicate()

class PrepLig(object):
    def __init__(self, input_mol, mol_format):
        if mol_format == "smi":
            self.ob_mol = pybel.readstring("smi", input_mol)
        elif mol_format == "sdf":
            self.ob_mol = next(pybel.readfile(mol_format, input_mol))
        else:
            raise ValueError(f"mol_format {mol_format} not supported")

    def addH(self, polaronly=False, correctforph=True, PH=7):
        self.ob_mol.OBMol.AddHydrogens(polaronly, correctforph, PH)
        obutils.writeMolecule(self.ob_mol.OBMol, "tmp_h.sdf")

    def gen_conf(self, seed=0):
        sdf_block = self.ob_mol.write("sdf")
        rdkit_mol = Chem.MolFromMolBlock(sdf_block, removeHs=False)
        try:
            AllChem.EmbedMolecule(rdkit_mol, randomSeed=seed)
            AllChem.MMFFOptimizeMolecule(rdkit_mol, maxIters=500, confId=0)
        except:
            rdkit_mol.Compute2DCoords()
        self.ob_mol = pybel.readstring("sdf", Chem.MolToMolBlock(rdkit_mol))
        obutils.writeMolecule(self.ob_mol.OBMol, "conf_h.sdf")

    @supress_stdout
    def get_pdbqt(self, lig_pdbqt=None):
        preparator = MoleculePreparation()
        preparator.prepare(self.ob_mol.OBMol)
        if lig_pdbqt is not None:
            preparator.write_pdbqt_file(lig_pdbqt)
            return
        else:
            return preparator.write_pdbqt_string()

class AutoDockVina:
    def __init__(
        self,
        *,
        exhaustiveness: int = 32,
        seed: int = 1234,
    ):
        self.exhaustiveness = exhaustiveness
        self.seed = seed
        self.result = 500

    def query_box(
        self,
        receptor_path: Path,
        ligand_path: Path,
        center: Tuple[float, float, float],
        box: Tuple[float, float, float] = (30.0, 30.0, 30.0),
        output_complex_path: Optional[Path] = None,
        score_only: bool = False,
        timeout: int = 300,
    ) -> float:
        """Run query with box coordinate."""
        def run_docking():
            self.result = 500
            vina = Vina(sf_name="vina", seed=self.seed, verbosity=0)
            vina.set_receptor(str(receptor_path))
            vina.set_ligand_from_file(str(ligand_path))
            vina.compute_vina_maps(center=center, box_size=box)
            if score_only:
                self.result = vina.score()[0]
            else:
                vina.dock(exhaustiveness=self.exhaustiveness, n_poses=1)
                self.result = vina.energies(n_poses=1)[0][0]
            if output_complex_path is not None:
                vina.write_poses(str(output_complex_path), n_poses=1)
        
        t = threading.Thread(target=run_docking)
        t.start()
        t.join(timeout)
        if t.is_alive():
            raise TimeoutError("Docking process exceeded the time limit.")
        return self.result


def get_box(pdb_fname, buffer=0):
    """Get box size from pdb file."""
    with open(pdb_fname) as f:
        lines = [
            l for l in f.readlines() if l.startswith("ATOM") or l.startswith("HETATM")
        ]
        if pdb_fname.endswith(".cif"):
            xs = []
            ys = []
            zs = []
            for l in lines:
                l = l.split()
                xs.append(float(l[10]))
                ys.append(float(l[11]))
                zs.append(float(l[12]))
        elif pdb_fname.endswith(".pdb"):
            xs = [float(l[31:39]) for l in lines]
            ys = [float(l[39:46]) for l in lines]
            zs = [float(l[47:55]) for l in lines]
        else:
            raise ValueError("Unsupported file format. Only .pdb and .cif are supported.")
        pocket_center = [
            (max(xs) + min(xs)) / 2,
            (max(ys) + min(ys)) / 2,
            (max(zs) + min(zs)) / 2,
        ]
        box_size = [
            (max(xs) - min(xs)) + buffer,
            (max(ys) - min(ys)) + buffer,
            (max(zs) - min(zs)) + buffer,
        ]
        return pocket_center, box_size


def dock(
    receptor_path,
    ligand_smiles,
    center=None,
    output_complex_path=None,
    get_box_from_pdb=True,
    num_ligand_confs=5,
):
    """Dock a ligand to a receptor."""

    dock_software = AutoDockVina()
    with tempfile.TemporaryDirectory() as tmpdir:
        # prepare protein
        prot = PrepProt(receptor_path)
        prot.del_water(Path(tmpdir) / f"dry.{prot.file_format}")
        if prot.file_format != "cif":
            # althouth pdb2pqr can handle cif, it is not recommended
            prot.addH(Path(tmpdir) / "prot.pqr")
        prot.get_pdbqt(Path(tmpdir) / "prot.pdbqt")

        # prepare box
        if get_box_from_pdb:
            center, box = get_box(receptor_path)
        else:
            assert center is not None, "center must be specified if get_box_from_pdb=False"
            box = [25.0, 25.0, 25.0]

        # prepare ligand
        if num_ligand_confs > 1:
            best_affinity, best_id = 0.0, None
            for seed in range(num_ligand_confs):
                lig = PrepLig(ligand_smiles, "smi")
                lig.addH()
                lig.gen_conf(seed=seed)
                lig.get_pdbqt(Path(tmpdir) / f"ligand_{seed}.pdbqt")

                # dock
                affinity = dock_software.query_box(
                    Path(tmpdir) / "prot.pdbqt",
                    Path(tmpdir) / f"ligand_{seed}.pdbqt",
                    center,
                    box,
                    output_complex_path=output_complex_path.replace(
                        ".pdbqt", f"_{seed}.pdbqt"
                    ),
                    score_only=False,
                )
                if affinity < best_affinity:
                    best_affinity = affinity
                    best_id = seed
            affinity = best_affinity
        else:
            lig = PrepLig(ligand_smiles, "smi")
            lig.addH()
            lig.gen_conf()
            lig.get_pdbqt(Path(tmpdir) / "ligand.pdbqt")

            # dock
            affinity = dock_software.query_box(
                Path(tmpdir) / "prot.pdbqt",
                Path(tmpdir) / "ligand.pdbqt",
                center,
                box,
                output_complex_path=output_complex_path,
                score_only=False,
            )
    return affinity, best_id


with open(args.smiles_file, 'r') as fr:
    all_lines = [e.strip() for e in fr]

df = pd.read_csv(f'{args.pdb_folder}/index_{args.pdbid}.csv')
centers = [df.iloc[0]['center_x'], df.iloc[0]['center_y'], df.iloc[0]['center_z']]

if not os.path.exists(args.docking_pose_folder):
    os.makedirs(args.docking_pose_folder, exist_ok=True)


for idx, line in tqdm(enumerate(all_lines),total=len(all_lines)):
    try:
        segs = line.split('\t')
        smiles = segs[0]
        prob = segs[1]
        t = dock(
            f'{args.pdb_folder}/{args.pdbid}.receptor.pdb',
            smiles,
            get_box_from_pdb=False,
            center=centers,
            output_complex_path=args.docking_pose_folder + f'/{idx}.pdbqt',
            num_ligand_confs=args.num_ligand_confs
        )
    except:
        continue