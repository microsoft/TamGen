#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Data transforms on template features.

Copied from AlphaFold2 project `alphafold/model/tf/data_transforms.py`."""

import functools
from typing import Dict, List

import numpy as np

from .common import residue_constants
from . import config

# Type aliases:
ProteinDatasetDict = Dict[str, np.ndarray]

NUMPY_MAP_HHBLITS_AATYPE_TO_OUR_AATYPE = np.asarray(residue_constants.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE, dtype=np.int32)


def curry1(f):
    """Supply all arguments but the first.

    >>> @curry1
    >>> def func(x, y, z=1):
    >>>     return x * y + z
    >>>
    >>> curried = func(2, z=3)
    >>> curried(4)
    >>> 11  # 4 * 2 + 3
    """

    @functools.wraps(f)
    def func(*args, **kwargs):
        return lambda x: f(x, *args, **kwargs)

    return func


def squeeze_features(protein: ProteinDatasetDict) -> ProteinDatasetDict:
    """Remove singleton and repeated dimensions in protein features."""
    protein['aatype'] = np.argmax(
        protein['aatype'], axis=-1).astype(np.int32)
    for k in [
            'domain_name', 'msa', 'num_alignments', 'seq_length', 'sequence',
            'superfamily', 'deletion_matrix', 'resolution',
            'between_segment_residues', 'residue_index', 'template_all_atom_masks']:
        if k in protein:
            final_dim = np.shape(protein[k])[-1]
            if isinstance(final_dim, int) and final_dim == 1:
                protein[k] = np.squeeze(protein[k], axis=-1)

    for k in ['seq_length', 'num_alignments']:
        if k in protein:
            protein[k] = protein[k][0]  # Remove fake sequence dimension
    return protein


def fix_templates_aatype(protein: ProteinDatasetDict) -> ProteinDatasetDict:
    """Fixes aatype encoding of templates."""

    protein['template_aatype'] = np.argmax(protein['template_aatype'], axis=-1).astype(np.int32)
    new_order = NUMPY_MAP_HHBLITS_AATYPE_TO_OUR_AATYPE
    protein['template_aatype'] = new_order[protein['template_aatype']]

    return protein


def make_template_mask(protein: ProteinDatasetDict) -> ProteinDatasetDict:
    protein['template_mask'] = np.ones(protein['template_domain_names'].shape, dtype=np.float32)
    return protein


def pseudo_beta_fn(aatype, all_atom_positions, all_atom_masks):
    """Create pseudo beta features."""
    is_gly = np.equal(aatype, residue_constants.restype_order['G'])
    ca_idx = residue_constants.atom_order['CA']
    cb_idx = residue_constants.atom_order['CB']
    pseudo_beta = np.where(
        np.tile(is_gly[..., None], [1] * len(is_gly.shape) + [3]),
        all_atom_positions[..., ca_idx, :],
        all_atom_positions[..., cb_idx, :])

    if all_atom_masks is not None:
        pseudo_beta_mask = np.where(
            is_gly, all_atom_masks[..., ca_idx], all_atom_masks[..., cb_idx])
        pseudo_beta_mask = pseudo_beta_mask.astype(np.float32)
        return pseudo_beta, pseudo_beta_mask
    else:
        return pseudo_beta


@curry1
def make_pseudo_beta(protein: ProteinDatasetDict, prefix='') -> ProteinDatasetDict:
    """Create pseudo-beta (alpha for glycine) position and mask."""
    assert prefix in ['', 'template_']
    protein[prefix + 'pseudo_beta'], protein[prefix + 'pseudo_beta_mask'] = pseudo_beta_fn(
        protein['template_aatype' if prefix else 'all_atom_aatype'],
        protein[prefix + 'all_atom_positions'],
        protein['template_all_atom_masks' if prefix else 'all_atom_mask'])
    return protein


def nonensembled_transforms():
    """Input pipeline functions which are not ensembled.

    Derived from AlphaFold2 project `alphafold/model/tf/input_pipeline.py:nonensembled_map_fns`.

    [NOTE]: Only implement template transforms now."""
    map_fns = [
        squeeze_features,
        fix_templates_aatype,
        make_template_mask,
        make_pseudo_beta('template_'),
    ]

    return map_fns


@curry1
def select_feat(protein: ProteinDatasetDict, feature_list: List[str]) -> ProteinDatasetDict:
    return {k: v for k, v in protein.items() if k in feature_list}


@curry1
def random_crop_to_size(protein: ProteinDatasetDict) -> ProteinDatasetDict:
    """Crop randomly to `crop_size`, or keep as is if shorter than that.

    Derived from AlphaFold2 project `alphafold/model/tf/data_transforms.py:random_crop_to_size`."""
    # [NOTE]: We do not apply random crop on templates now.
    return protein


@curry1
def crop_templates(protein: ProteinDatasetDict, max_templates: int) -> ProteinDatasetDict:
    for k, v in protein.items():
        if k.startswith('template_'):
            protein[k] = v[:max_templates]
    return protein


def ensembled_transforms():
    """Input pipeline functions that can be ensembled and averaged.

    Derived from AlphaFold2 project `alphafold/model/tf/input_pipeline.py:ensembled_map_fns`.

    [NOTE]: Only implement template transforms now."""

    map_fns = [
        select_feat(list(config.EVAL_FEATURES)),
        random_crop_to_size(),
        crop_templates(config.MAX_TEMPLATES),
    ]

    return map_fns


def duplicate_features(protein: ProteinDatasetDict, n_ensembles: int) -> ProteinDatasetDict:
    for key, value in protein.items():
        if not key.startswith('template_'):
            continue
        protein[key] = np.repeat(value[None], n_ensembles, axis=0)
    return protein


def apply_transforms(protein: ProteinDatasetDict) -> ProteinDatasetDict:
    """Apply filters and maps to an existing dataset.

    Derived from AlphaFold2 project `alphafold/model/tf/input_pipeline.py:process_tensors_from_config`.
    """

    nonensembled_map_fns = nonensembled_transforms()
    for fn in nonensembled_map_fns:
        protein = fn(protein)

    ensembled_map_fns = ensembled_transforms()
    for fn in ensembled_map_fns:
        protein = fn(protein)

    # [NOTE]
    #  Since AlphaFold2 config data.eval.subsample_templates == False,
    #  we always get top templates and do not resample on template features.
    protein = duplicate_features(protein, n_ensembles=config.N_ENSEMBLES)

    return protein
