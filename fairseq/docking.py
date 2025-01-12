import contextlib
import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Optional, Tuple

import AutoDockTools
import numpy as np
from meeko import MoleculePreparation, obutils
from openbabel import pybel
from rdkit import Chem
from rdkit.Chem import AllChem
from vina import Vina


np.int = int

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
            ["python", prepare_receptor, "-r", self.prot_pqr if self.file_format == "pdb" else self.prot, "-o", prot_pdbqt],
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

    def get_pdbqt(self, lig_pdbqt):
        with open("/dev/null", "w") as null:
            with contextlib.redirect_stdout(null):
                preparator = MoleculePreparation()
                preparator.prepare(self.ob_mol.OBMol)
                preparator.write_pdbqt_file(lig_pdbqt)

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
            line for line in f.readlines() if line.startswith("ATOM") or line.startswith("HETATM")
        ]
        if pdb_fname.endswith(".pdb"):
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
    center,
    output_complex_path,
    num_ligand_confs,
):
    """Dock a ligand to a receptor."""

    dock_software = AutoDockVina()
    with tempfile.TemporaryDirectory() as tmpdir:
        # prepare protein
        prot = PrepProt(receptor_path)
        prot.del_water(Path(tmpdir) / f"dry.{prot.file_format}")
        prot.addH(Path(tmpdir) / "prot.pqr")
        prot.get_pdbqt(Path(tmpdir) / "prot.pdbqt")

        box = [25, 25, 25]

        # prepare ligand
        best_affinity, best_idx = 0, None
        for i in range(num_ligand_confs):
            lig = PrepLig(ligand_smiles, "smi")
            lig.addH()
            lig.gen_conf(seed=i)
            lig.get_pdbqt(Path(tmpdir) / f"cand_{i}.pdbqt")

            # dock
            affinity = dock_software.query_box(
                Path(tmpdir) / "prot.pdbqt",
                Path(tmpdir) / f"cand_{i}.pdbqt",
                center,
                box,
                output_complex_path=Path(tmpdir) / f"ligand_{i}.pdbqt",
                score_only=False,
                timeout=30,
            )

            if affinity < best_affinity:
                best_affinity = affinity
                best_idx = i

        shutil.copyfile(
            str(Path(tmpdir) / f"ligand_{best_idx}.pdbqt"),
            output_complex_path,
        )

def attempt_docking(pdb_id, ligand_id, chain_id, smiles, center, work_dir, n_conf):
    output_pdbqt = work_dir + f'/{ligand_id}_{chain_id}.pdbqt'
    output_sdf = work_dir + f'/{ligand_id}_{chain_id}.sdf'
    dock(
        f'{work_dir}/{pdb_id}.receptor.pdb',
        smiles,
        center,
        output_complex_path=output_pdbqt,
        num_ligand_confs=n_conf,
    )

    subprocess.Popen(
        f"obabel -i pdbqt {output_pdbqt} -o sdf -O {output_sdf}",
        shell=True,
    )
