#! /usr/bin/python
# -*- coding: utf-8 -*-

"""AlphaFold2 model config."""

N_ENSEMBLES = 4
MAX_TEMPLATES = 4

NUM_RES = 'num residues placeholder'
NUM_MSA_SEQ = 'msa placeholder'
NUM_EXTRA_SEQ = 'extra msa placeholder'
NUM_TEMPLATES = 'num templates placeholder'

# Final features in model input.
EVAL_FEATURES = {
    'aatype': [NUM_RES],
    'all_atom_mask': [NUM_RES, None],
    'all_atom_positions': [NUM_RES, None, None],
    'alt_chi_angles': [NUM_RES, None],
    'atom14_alt_gt_exists': [NUM_RES, None],
    'atom14_alt_gt_positions': [NUM_RES, None, None],
    'atom14_atom_exists': [NUM_RES, None],
    'atom14_atom_is_ambiguous': [NUM_RES, None],
    'atom14_gt_exists': [NUM_RES, None],
    'atom14_gt_positions': [NUM_RES, None, None],
    'atom37_atom_exists': [NUM_RES, None],
    'backbone_affine_mask': [NUM_RES],
    'backbone_affine_tensor': [NUM_RES, None],
    'bert_mask': [NUM_MSA_SEQ, NUM_RES],
    'chi_angles': [NUM_RES, None],
    'chi_mask': [NUM_RES, None],
    'extra_deletion_value': [NUM_EXTRA_SEQ, NUM_RES],
    'extra_has_deletion': [NUM_EXTRA_SEQ, NUM_RES],
    'extra_msa': [NUM_EXTRA_SEQ, NUM_RES],
    'extra_msa_mask': [NUM_EXTRA_SEQ, NUM_RES],
    'extra_msa_row_mask': [NUM_EXTRA_SEQ],
    'is_distillation': [],
    'msa_feat': [NUM_MSA_SEQ, NUM_RES, None],
    'msa_mask': [NUM_MSA_SEQ, NUM_RES],
    'msa_row_mask': [NUM_MSA_SEQ],
    'pseudo_beta': [NUM_RES, None],
    'pseudo_beta_mask': [NUM_RES],
    'random_crop_to_size_seed': [None],
    'residue_index': [NUM_RES],
    'residx_atom14_to_atom37': [NUM_RES, None],
    'residx_atom37_to_atom14': [NUM_RES, None],
    'resolution': [],
    'rigidgroups_alt_gt_frames': [NUM_RES, None, None],
    'rigidgroups_group_exists': [NUM_RES, None],
    'rigidgroups_group_is_ambiguous': [NUM_RES, None],
    'rigidgroups_gt_exists': [NUM_RES, None],
    'rigidgroups_gt_frames': [NUM_RES, None, None],
    'seq_length': [],
    'seq_mask': [NUM_RES],
    'target_feat': [NUM_RES, None],
    'template_aatype': [NUM_TEMPLATES, NUM_RES],
    'template_all_atom_masks': [NUM_TEMPLATES, NUM_RES, None],
    'template_all_atom_positions': [
        NUM_TEMPLATES, NUM_RES, None, None],
    'template_backbone_affine_mask': [NUM_TEMPLATES, NUM_RES],
    'template_backbone_affine_tensor': [
        NUM_TEMPLATES, NUM_RES, None],
    'template_mask': [NUM_TEMPLATES],
    'template_pseudo_beta': [NUM_TEMPLATES, NUM_RES, None],
    'template_pseudo_beta_mask': [NUM_TEMPLATES, NUM_RES],
    'template_sum_probs': [NUM_TEMPLATES, None],
    'true_msa': [NUM_MSA_SEQ, NUM_RES]
}
