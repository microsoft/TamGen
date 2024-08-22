#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Target2Drug evaluator."""

import json
import logging
import math
import random
import pickle
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Tuple

from rdkit import Chem
from rdkit.Chem.rdchem import Mol as MolType
from tqdm import tqdm

from .basic import mol_scores as ms
from .basic import smiles_utils as smu
from .basic.run_docking import docking
from .basic import lipinski_pass


@dataclass
class EvaluateCase:
    index: int
    ref_smiles: str
    hyp_smiles_list: List[str]
    pdb_id: Optional[str]
    box_center: Tuple[float, float, float] = None

    @property
    def ref_mol(self) -> MolType:
        if not hasattr(self, '_ref_mol'):
            self._ref_mol = Chem.MolFromSmiles(self.ref_smiles)
        return self._ref_mol

    @property
    def hyp_mols_list(self) -> List[MolType]:
        if not hasattr(self, '_hyp_mols_list'):
            self._hyp_mols_list = [Chem.MolFromSmiles(smi) for smi in self.hyp_smiles_list]
        return self._hyp_mols_list


def _safe_mean(data, default=None):
    if not data:
        return default
    return statistics.mean(data)


def _safe_std(data, default=None):
    try:
        return statistics.stdev(data)
    except statistics.StatisticsError:
        return default

def _safe_median(data, default=None):
    if not data:
        return default
    return statistics.median(data)


def _stat_str(data):
    if not data:
        return '[NO DATA]'
    return f'{statistics.mean(data)} ± {statistics.stdev(data)}'


@dataclass
class EvaluateResult:
    index: List[int]
    testset_name: str
    test_size: int
    SA_score: List[Dict]
    similarity: List[Dict]
    logP: List[Dict]
    QED: List[Dict]
    diversity: List[float]
    SR: float
    lipinski_pass_rate: float
    affinity: List[Dict]
    docking_pdb_ids: List[str]
    BLEU: float
    unique_ligands: List[int]

    mol_weight: List[Dict]
    num_atoms: List[Dict]
    num_rings: List[Dict]
    num_H_acceptors: List[Dict]
    num_H_donors: List[Dict]
    num_rot_bonds: List[Dict]
    TPSAs: List[Dict]


class Tgt2DrugEvaluator:
    DOCKING_MODES = ['pdb', 'structure']
    CANDIDATE_STAT_METHODS = ['max', 'mean', 'median']

    def __init__(
            self, out_dir: Optional[Path], *,
            testset_name: str = 'test',
            top_n: int = 1,
            random: bool = False,
            run_bleu: bool = False,
            docking_mode: str = 'pdb', docking_first_n: int = 10,
            candidate_stat_method: str = 'max',
            disable_box_center: bool = False,
            smina_bin_path: Path = None, split_cache_path: Path = None,
            pdb_cache_path: Path = None, ccd_cache_path: Path = None,
            docking_result_cache_path: Path = None,
    ):
        self.out_dir = out_dir
        self.testset_name = testset_name
        self.top_n = top_n
        self.random = random
        self.run_bleu = run_bleu
        self.smina_bin_path = smina_bin_path
        self.split_cache_path = split_cache_path
        self.pdb_cache_path = pdb_cache_path
        self.ccd_cache_path = ccd_cache_path
        self.docking_result_cache_path = docking_result_cache_path

        assert candidate_stat_method in self.CANDIDATE_STAT_METHODS
        self.candidate_stat_method = candidate_stat_method
        if self.candidate_stat_method == 'max':
            self.candidate_stat_func = max
        elif self.candidate_stat_method == 'mean':
            self.candidate_stat_func = _safe_mean
        elif self.candidate_stat_method == 'median':
            self.candidate_stat_func = _safe_median
        self.disable_box_center = disable_box_center

        assert docking_mode in self.DOCKING_MODES
        self.docking_mode = docking_mode

        self.docking_first_n = docking_first_n
        if self.docking_first_n < 0:
            self.docking_first_n = 100000000

        if self.out_dir is not None:
            self.out_dir.mkdir(exist_ok=True)

    def report(self, result: EvaluateResult):
        if self.out_dir is not None:
            for filename, scores in [
                (f'SA_score-{self.testset_name}.json', result.SA_score),
                (f'similarity-{self.testset_name}.json', result.similarity),
                (f'logP-{self.testset_name}.json', result.logP),
                (f'QED-{self.testset_name}.json', result.QED),
                (f'mol_weight-{self.testset_name}.json', result.mol_weight),
                (f'num_atoms-{self.testset_name}.json', result.num_atoms),
                (f'num_rings-{self.testset_name}.json', result.num_rings),
                (f'num_H_acceptors-{self.testset_name}.json', result.num_H_acceptors),
                (f'num_H_donors-{self.testset_name}.json', result.num_H_donors),
                (f'num_rot_bonds-{self.testset_name}.json', result.num_rot_bonds),
                (f'TPSAs-{self.testset_name}.json', result.TPSAs),
                (f'Affinity-{self.testset_name}.json', result.affinity),
            ]:
                with open(self.out_dir / filename, 'w', encoding='utf-8') as f_score:
                    tmp = {}
                    for i, score in enumerate(scores):
                        tmp[result.index[i]] = score
                    json.dump(tmp, f_score, indent=4)

        # finite_affinity = [a for a in result.affinity if math.isfinite(a)]
        data = {
            'testset_name':
            result.testset_name,
            'test_size':
            result.test_size,
            'SA_score':
            _safe_mean([
                score[self.candidate_stat_method] for score in result.SA_score
            ],
                       default=math.nan),
            'SA_score_std':
            _safe_std([
                score[self.candidate_stat_method] for score in result.SA_score
            ],
                      default=math.nan),
            'similarity':
            _safe_mean([
                score[self.candidate_stat_method]
                for score in result.similarity
            ],
                       default=math.nan),
            'similarity_std':
            _safe_std([
                score[self.candidate_stat_method]
                for score in result.similarity
            ],
                      default=math.nan),
            'logP':
            _safe_mean(
                [score[self.candidate_stat_method] for score in result.logP],
                default=math.nan),
            'logP_std':
            _safe_std(
                [score[self.candidate_stat_method] for score in result.logP],
                default=math.nan),
            'QED':
            _safe_mean(
                [score[self.candidate_stat_method] for score in result.QED],
                default=math.nan),
            'QED_std':
            _safe_std(
                [score[self.candidate_stat_method] for score in result.QED],
                default=math.nan),
            'diversity':
            _safe_mean(result.diversity, default=math.nan),
            'diversity_std':
            _safe_std(result.diversity, default=math.nan),
            'SR':
            result.SR,
            'lipinski_pass_rate':
            result.lipinski_pass_rate,
            'affinity':
            _safe_mean([
                score[self.candidate_stat_method] for score in result.affinity
            ],
                       default=math.nan),
            'affinity_std':
            _safe_std([
                score[self.candidate_stat_method] for score in result.affinity
            ],
                      default=math.nan),
            'docking_pdb_ids':
            result.docking_pdb_ids,
            'BLEU':
            result.BLEU,
            'unique_ligands':
            result.unique_ligands,
            'mol_weight':
            _safe_mean([
                score[self.candidate_stat_method]
                for score in result.mol_weight
            ],
                       default=math.nan),
            'mol_weight_std':
            _safe_std([
                score[self.candidate_stat_method]
                for score in result.mol_weight
            ],
                      default=math.nan),
            'num_atoms':
            _safe_mean([
                score[self.candidate_stat_method] for score in result.num_atoms
            ],
                       default=math.nan),
            'num_atoms_std':
            _safe_std([
                score[self.candidate_stat_method] for score in result.num_atoms
            ],
                      default=math.nan),
            'num_rings':
            _safe_mean([
                score[self.candidate_stat_method] for score in result.num_rings
            ],
                       default=math.nan),
            'num_rings_std':
            _safe_std([
                score[self.candidate_stat_method] for score in result.num_rings
            ],
                      default=math.nan),
            'num_H_acceptors':
            _safe_mean([
                score[self.candidate_stat_method]
                for score in result.num_H_acceptors
            ],
                       default=math.nan),
            'num_H_acceptors_std':
            _safe_std([
                score[self.candidate_stat_method]
                for score in result.num_H_acceptors
            ],
                      default=math.nan),
            'num_H_donors':
            _safe_mean([
                score[self.candidate_stat_method]
                for score in result.num_H_donors
            ],
                       default=math.nan),
            'num_H_donors_std':
            _safe_std([
                score[self.candidate_stat_method]
                for score in result.num_H_donors
            ],
                      default=math.nan),
            'num_rot_bonds':
            _safe_mean([
                score[self.candidate_stat_method]
                for score in result.num_rot_bonds
            ],
                       default=math.nan),
            'num_rot_bonds_std':
            _safe_std([
                score[self.candidate_stat_method]
                for score in result.num_rot_bonds
            ],
                      default=math.nan),
            'TPSAs':
            _safe_mean(
                [score[self.candidate_stat_method] for score in result.TPSAs],
                default=math.nan),
            'TPSAs_std':
            _safe_std(
                [score[self.candidate_stat_method] for score in result.TPSAs],
                default=math.nan),
        }
        print(f'| Summary of {self.testset_name}:')
        for key, value in data.items():
            print(f'|   {key} = {value}')
        if self.out_dir is not None:
            summary_fn = self.out_dir / f'summary-{self.testset_name}.json'
            with summary_fn.open('w', encoding='utf-8') as f_summary:
                json.dump(data, f_summary, indent=4)
            print(f'| Dumped to {summary_fn}.')

    def evaluate_all(self, cases: List[EvaluateCase]) -> EvaluateResult:
        random.seed(1)
        result = EvaluateResult(
            index=[],
            testset_name=self.testset_name,
            test_size=len(cases),
            SA_score=[],
            similarity=[],
            logP=[],
            QED=[],
            diversity=[],
            SR=0.0,
            lipinski_pass_rate=0.0,
            affinity=[],
            docking_pdb_ids=[],
            BLEU=math.nan,
            unique_ligands=[],

            mol_weight=[],
            num_atoms=[],
            num_rings=[],
            num_H_acceptors=[],
            num_H_donors=[],
            num_rot_bonds=[],
            TPSAs=[],
        )

        successes = []
        lipinski_passes = []
        for case in tqdm(cases):
            n_hyp = len(case.hyp_smiles_list)
            if n_hyp < self.top_n:
                logging.warning(f'Case {case.index} only have {n_hyp} hypothesis, but top {self.top_n} required.')
            if self.random:
                mols = random.sample(case.hyp_mols_list, min(len(case.hyp_mols_list), self.top_n))
            else:
                mols = case.hyp_mols_list[:self.top_n]
            successes += [float(mol is not None) for mol in mols]
            lipinski_passes += [float(lipinski_pass(mol)) for mol in mols]

            SA_scores, logP_scores, QED_scores, sim_scores = [], [], [], []
            mol_weights, num_atoms, num_rings = [], [], []
            num_H_acceptors, num_H_donors, num_rot_bonds, TPSAs = [], [], [], []
            for mol in mols:
                if mol is not None:
                    SA_scores.append(ms.normalized_sa_score(mol))
                    logP_scores.append(ms.penalized_logp(mol, penalty=-100.0))
                    QED_scores.append(ms.qed_score(mol))
                    sim_scores.append(ms.similarity(mol, case.ref_mol))
                    mol_weights.append(smu.molecular_weight(mol))
                    num_atoms.append(smu.num_atoms(mol, only_heavy=True))
                    num_rings.append(smu.num_rings(mol))
                    num_H_acceptors.append(smu.num_H_acceptors(mol))
                    num_H_donors.append(smu.num_H_donors(mol))
                    num_rot_bonds.append(smu.num_rot_bonds(mol))
                    TPSAs.append(smu.TPSA(mol, includeSandP=True))
                else:
                    SA_scores.append(0.0)
                    logP_scores.append(-100.0)
                    QED_scores.append(0.0)
                    sim_scores.append(0.0)
                    mol_weights.append(0.0)
                    num_atoms.append(0)
                    num_rings.append(0)
                    num_H_acceptors.append(0)
                    num_H_donors.append(0)
                    num_rot_bonds.append(0)
                    TPSAs.append(0.0)
            result.index.append(case.index)
            result.SA_score.append({
                'max': max(SA_scores, default=0.0),
                'mean': _safe_mean(SA_scores, default=0.0),
                'median': _safe_median(SA_scores, default=0.0),
                'all': SA_scores,
            })
            result.logP.append({
                'max': max(logP_scores, default=-100.),
                'mean': _safe_mean(logP_scores, default=-100.),
                'median': _safe_median(logP_scores, default=-100.),
                'all': logP_scores,
            })
            result.QED.append({
                'max': max(QED_scores, default=0.0),
                'mean': _safe_mean(QED_scores, default=0.0),
                'median': _safe_median(QED_scores, default=0.0),
                'all': QED_scores,
            })
            result.similarity.append({
                'max': max(sim_scores, default=0.0),
                'mean': _safe_mean(sim_scores, default=0.0),
                'median': _safe_median(sim_scores, default=0.0),
                'all': sim_scores,
            })
            result.mol_weight.append({
                'max': max(mol_weights, default=0.0),
                'mean': _safe_mean(mol_weights, default=0.0),
                'median': _safe_median(mol_weights, default=0.0),
                'all': mol_weights,
            })
            result.num_atoms.append({
                'max': max(num_atoms, default=0),
                'mean': _safe_mean(num_atoms, default=0),
                'median': _safe_median(num_atoms, default=0),
                'all': num_atoms,
            })
            result.num_rings.append({
                'max': max(num_rings, default=0),
                'mean': _safe_mean(num_rings, default=0),
                'median': _safe_median(num_rings, default=0),
                'all': num_rings,
            })
            result.diversity.append(ms.diversity(mols))
            result.num_H_acceptors.append({
                'max': max(num_H_acceptors, default=0),
                'mean': _safe_mean(num_H_acceptors, default=0),
                'median': _safe_median(num_H_acceptors, default=0),
                'all': num_H_acceptors,
            })
            result.num_H_donors.append({
                'max': max(num_H_donors, default=0),
                'mean': _safe_mean(num_H_donors, default=0),
                'median': _safe_median(num_H_donors, default=0),
                'all': num_H_donors,
            })
            result.num_rot_bonds.append({
                'max': max(num_rot_bonds, default=0),
                'mean': _safe_mean(num_rot_bonds, default=0),
                'median': _safe_median(num_rot_bonds, default=0),
                'all': num_rot_bonds,
            })
            result.TPSAs.append({
                'max': max(TPSAs, default=0.0),
                'mean': _safe_mean(TPSAs, default=0.0),
                'median': _safe_median(TPSAs, default=0.0),
                'all': TPSAs,
            })

        if self.run_bleu:
            ref_smi_list = [case.ref_smiles for case in cases]
            hyp_smi_list = list(zip(*(case.hyp_smiles_list[:self.top_n] for case in cases)))
            result.BLEU = ms.smiles_bleu(hyp_smi_list, ref_smi_list, tokenize='char')
        result.SR = _safe_mean(successes)
        result.lipinski_pass_rate = _safe_mean(lipinski_passes)

        # Unique ligands (top 1 to top N)
        for i in range(1, self.top_n + 1):
            top_n_ligands = []
            for case in cases:
                top_n_ligands.extend(case.hyp_smiles_list[:i])
            unique_ligands = set(top_n_ligands)
            result.unique_ligands.append(len(unique_ligands))

        if self.docking_first_n > 0:
            print('| Running docking ...')
            if self.docking_mode == 'pdb':
                self._pdb_docking(cases, result)
            else:   # self.docking_mode == 'structure'
                self._structure_docking(cases, result)

        self.report(result)

        return result

    def _pdb_docking(self, cases, result):
        if self.docking_result_cache_path is not None:
            if self.docking_result_cache_path.exists():
                logging.info(f'Loading docking result cache from {self.docking_result_cache_path}')
                with open(self.docking_result_cache_path, 'rb') as f_docking_result_cache:
                    docking_result_cache = pickle.load(f_docking_result_cache)
            else:
                docking_result_cache = {}
        else:
            docking_result_cache = None

        docking_pdb_ids = set()
        for case in cases:
            docking_pdb_ids.add(case.pdb_id)
            if len(docking_pdb_ids) >= self.docking_first_n:
                break

        result.docking_pdb_ids = sorted(docking_pdb_ids)
        pdb_affinities = {pdb_id: [] for pdb_id in docking_pdb_ids}
        docking_cases = [case for case in cases if case.pdb_id in docking_pdb_ids]
        for case in tqdm(docking_cases):    # type: EvaluateCase
            affinity = docking(
                pdb_id=case.pdb_id,
                ligand_smiles=case.hyp_smiles_list[0],
                smina_bin_path=self.smina_bin_path,
                split_cache_path=self.split_cache_path,
                pdb_cache_path=self.pdb_cache_path,
                ccd_cache_path=self.ccd_cache_path,
                docking_result_cache=docking_result_cache,
                box_center=None if self.disable_box_center else case.box_center,
            )
            if affinity is None:
                affinity = math.inf
            pdb_affinities[case.pdb_id].append(affinity)
        best_pdb_affinities = []
        for key, candidates in pdb_affinities.items():
            neg_finite_candidates = [-affinity for affinity in candidates if math.isfinite(affinity)]
            best_pdb_affinities.append({
                'pdb_id':
                key,
                'max':
                min(neg_finite_candidates, default=0.0),
                'mean':
                _safe_mean(neg_finite_candidates, default=0.0),
                'median':
                _safe_median(neg_finite_candidates, default=0.0),
                'all':
                neg_finite_candidates,
            })
        result.affinity = best_pdb_affinities

        if self.docking_result_cache_path is not None:
            with open(self.docking_result_cache_path, 'wb') as f_docking_result_cache:
                pickle.dump(docking_result_cache, f_docking_result_cache)
            logging.info(
                f'Dump docking cache into {self.docking_result_cache_path} ({len(docking_result_cache)} items)')

    def _structure_docking(self, cases, result):
        raise NotImplementedError()
