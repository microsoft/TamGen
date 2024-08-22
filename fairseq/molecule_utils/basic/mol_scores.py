#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Calculate molecule scores."""


import gzip
import math
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, QED, Descriptors, rdMolDescriptors

__all__ = [
    'similarity', 'diversity', 'similarity_fp',
    'drd2_score', 'qed_score', 'penalized_logp',
    'smiles_bleu', 'calculate_sa_score', 'RDKit_LogP',
]


def similarity(mol_a, mol_b):
    if mol_a is None or mol_b is None:
        return 0.0
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol_a, 2, nBits=2048, useChirality=False)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol_b, 2, nBits=2048, useChirality=False)
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def similarity_fp(fp1, fp2):
    if fp1 is None or fp2 is None:
        return 0.0
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def similarity_smiles(smi1, smi2):
    return similarity(Chem.MolFromSmiles(smi1), Chem.MolFromSmiles(smi2))


def diversity_backup(succ_cand: list):
    if not succ_cand:
        return None
    if len(succ_cand) == 1:
        return 0.0
    div = 0.0
    tot = 0
    for i in range(len(succ_cand)):
        for j in range(i + 1, len(succ_cand)):
            div += 1 - similarity(succ_cand[i], succ_cand[j])
            tot += 1
    return div / tot


def diversity(succ_cand: list):
    if not succ_cand:
        return None
    if len(succ_cand) == 1:
        return 0.0
    div = 0.0
    tot = 0
    succ_cand_fps = [
        AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, useChirality=False) if mol is not None else None
        for mol in succ_cand
    ]
    for i in range(len(succ_cand_fps)):
        for j in range(i + 1, len(succ_cand_fps)):
            div += 1 - similarity_fp(succ_cand_fps[i], succ_cand_fps[j])
            tot += 1
    return div / tot


# DRD2 Scorer: Scores based on an ECFP classifier for activity.
def _fingerprints_from_mol(mol):
    fp = AllChem.GetMorganFingerprint(mol, 3, useCounts=True, useFeatures=True)
    size = 2048
    nfp = np.zeros((1, size), np.int32)
    for idx, v in fp.GetNonzeroElements().items():
        nidx = idx % size
        nfp[0, nidx] += int(v)
    return nfp


CLF_MODEL = None


# [NOTE]: clf_py36.pkl requires scikit-learn==0.18.1, we install 0.19.0.
def drd2_score(mol):
    if mol is None:
        return 0.0

    global CLF_MODEL
    if CLF_MODEL is None:
        try:
            import sklearn
        except ModuleNotFoundError:
            print('ERROR: scikit-learn not installed. Please install scikit-learn.')
            exit()
        name = Path(__file__).absolute().parent / 'clf_py36.pkl'
        with open(name, 'rb') as f:
            CLF_MODEL = pickle.load(f)

    fp = _fingerprints_from_mol(mol)
    score = CLF_MODEL.predict_proba(fp)[:, 1]
    return float(score)


def qed_score(mol):
    """QED scorer."""
    if mol is None:
        return 0.0
    try:
        return QED.qed(mol)
    except ValueError:
        return 0.0


def qed_score_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return qed_score(mol)


# calculation of synthetic accessibility score as described in:
#
# Estimation of Synthetic Accessibility Score of Drug-like Molecules based on Molecular Complexity and Fragment Contributions
# Peter Ertl and Ansgar Schuffenhauer
# Journal of Cheminformatics 1:8 (2009)
# http://www.jcheminf.com/content/1/1/8
#
# several small modifications to the original paper are included
# particularly slightly different formula for marocyclic penalty
# and taking into account also molecule symmetry (fingerprint density)
#
# for a set of 10k diverse molecules the agreement between the original method
# as implemented in PipelinePilot and this implementation is r2 = 0.97
#
# peter ertl & greg landrum, september 2013
_F_SCORES = None


def _read_fragment_scores():
    global _F_SCORES
    _F_SCORES = pickle.load(gzip.open(Path(__file__).absolute().parent / 'fpscores.pkl.gz'))
    out_dict = {}
    for i in _F_SCORES:
        for j in range(1, len(i)):
            out_dict[i[j]] = float(i[0])
    _F_SCORES = out_dict


def _num_bridgeheads_and_spiro(mol, ri=None):
    n_spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    n_bridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    return n_bridgehead, n_spiro


def calculate_sa_score(mol):
    if _F_SCORES is None:
        _read_fragment_scores()

    # fragment score
    fp = rdMolDescriptors.GetMorganFingerprint(mol, 2)  # <- 2 is the *radius* of the circular fingerprint
    fps = fp.GetNonzeroElements()
    score1 = 0.
    nf = 0
    for bitId, v in fps.items():
        nf += v
        sfp = bitId
        score1 += _F_SCORES.get(sfp, -4) * v
    score1 /= nf

    # features score
    n_atoms = mol.GetNumAtoms()
    n_chiral_centers = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    ri = mol.GetRingInfo()
    n_bridgeheads, n_spiro = _num_bridgeheads_and_spiro(mol, ri)
    n_macrocycles = 0
    for x in ri.AtomRings():
        if len(x) > 8:
            n_macrocycles += 1

    size_penalty = n_atoms ** 1.005 - n_atoms
    stereo_penalty = math.log10(n_chiral_centers + 1)
    spiro_penalty = math.log10(n_spiro + 1)
    bridge_penalty = math.log10(n_bridgeheads + 1)
    macrocycle_penalty = 0.
    # ---------------------------------------
    # This differs from the paper, which defines:
    #  macrocycle_penalty = math.log10(n_macrocycles+1)
    # This form generates better results when 2 or more macrocycles are present
    if n_macrocycles > 0:
        macrocycle_penalty = math.log10(2)

    score2 = 0. - size_penalty - stereo_penalty - spiro_penalty - bridge_penalty - macrocycle_penalty

    # correction for the fingerprint density
    # not in the original publication, added in version 1.1
    # to make highly symmetrical molecules easier to synthetise
    score3 = 0.
    if n_atoms > len(fps):
        score3 = math.log(float(n_atoms) / len(fps)) * .5

    sascore = score1 + score2 + score3

    # need to transform "raw" value into scale between 1 and 10
    _min = -4.0
    _max = 2.5
    sascore = 11. - (sascore - _min + 1) / (_max - _min) * 9.
    # smooth the 10-end
    if sascore > 8.:
        sascore = 8. + math.log(sascore + 1. - 9.)
    if sascore > 10.:
        sascore = 10.0
    elif sascore < 1.:
        sascore = 1.0

    return sascore


def normalized_sa_score(mol):
    return (10.0 - calculate_sa_score(mol)) / 9.0


def penalized_logp(mol, penalty=-100):
    """Modified from https://github.com/bowenliu16/rl_graph_generation."""
    if mol is None:
        return penalty

    logP_mean = 2.4570953396190123
    logP_std = 1.434324401111988
    SA_mean = -3.0525811293166134
    SA_std = 0.8335207024513095
    cycle_mean = -0.0485696876403053
    cycle_std = 0.2860212110245455

    try:
        log_p = Descriptors.MolLogP(mol)
    except (ValueError, RuntimeError):
        return penalty
    try:
        SA = -calculate_sa_score(mol)
    except ZeroDivisionError:
        return penalty

    # cycle score
    cycle_list = nx.cycle_basis(nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol)))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    cycle_score = -cycle_length

    normalized_log_p = (log_p - logP_mean) / logP_std
    normalized_SA = (SA - SA_mean) / SA_std
    normalized_cycle = (cycle_score - cycle_mean) / cycle_std
    return normalized_log_p + normalized_SA + normalized_cycle

def RDKit_LogP(mol):
    return Descriptors.MolLogP(mol)

def calculate_score(dataset, hyp_mols, ref_mols):
    if dataset == 'drd2':
        score_list = [[drd2_score(mol) for mol in hyp_cand] for hyp_cand in hyp_mols]
    elif dataset == 'qed':
        score_list = [[qed_score(mol) for mol in hyp_cand] for hyp_cand in hyp_mols]
    elif dataset in {'logp04', 'logp06'}:
        score_list = []
        for ref, hyp_cand in zip(ref_mols, hyp_mols):
            ref_logp = penalized_logp(ref)
            score_cand_list = [penalized_logp(hyp) - ref_logp for hyp in hyp_cand]
            score_list.append(score_cand_list)
    elif dataset == 'drug':
        score_list = []
    elif dataset == 'zinc-mono':
        score_list = []
    else:
        raise RuntimeError('unknown dataset {!r}'.format(dataset))
    return score_list


def smiles_bleu(smi_hyp_list, smi_ref_list, tokenize='char') -> float:
    import sacrebleu

    if isinstance(smi_hyp_list[0], str):
        smi_hyp_list = [smi_hyp_list]

    score = sacrebleu.corpus_bleu(smi_ref_list, smi_hyp_list, tokenize=tokenize)
    return score.score
