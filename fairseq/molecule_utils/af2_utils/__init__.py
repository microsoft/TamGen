#! /usr/bin/python
# -*- coding: utf-8 -*-

"""AlphaFold2-related functions.

AlphaFold2 Template-related Feature Processing steps:


HHSearch Results / Templates / mmCIF files: Atom positions

== (alphafold/data/templates.py:_extract_template_features) ==>

features.pkl: Dict[str, np.ndarray]
    aatype (N, 21)
    between_segment_residues (N,)
    domain_name (1,)
    residue_index (N,)
    seq_length (N,)
    sequence (1,)
    deletion_matrix_int (N_msa, N)
    msa (N_msa, N)
    num_alignments (N,)
    * template_aatype (N_tpl, N, 22)
    * template_all_atom_masks (N_tpl, N, 37)
    * template_all_atom_positions (N_tpl, N, 37, 3)
    * template_domain_names (N_tpl,)
    * template_sequence (N_tpl,)
    * template_sum_probs (N_tpl, 1)

== (alphafold/model/features.py:np_example_to_features) ==>

AF2 model input: Dict[str, np.ndarray]
(N_maxtpl = 4, N_ens = (N_recycle + 1) * N_ensemble = 4, N_extra_msa = 5120)
    aatype (N_ens, N)
    residue_index (N_ens, N)
    seq_length (N_ens,)
    atom14_atom_exists (N_ens, N, 14)
    residx_atom14_to_atom37 (N_ens, N, 14)
    residx_atom37_to_atom14 (N_ens, N, 37)
    atom37_atom_exists (N_ens, N, 37)
    bert_mask (N_ens, N_msa, N)
    true_msa (N_ens, N_msa, N)
    extra_msa (N_ens, N_extra_msa, N)
    extra_msa_mask (N_ens, N_extra_msa, N)
    extra_msa_row_mask (N_ens, N_extra_msa)
    extra_has_deletion (N_ens, N_extra_msa, N)
    extra_deletion_value (N_ens, N_extra_msa, N)
    msa_feat (N_ens, 508, N, 49)
    target_feat (N_ens, N, 22)
    is_distillation (N_ens,)
    seq_mask (N_ens, N)
    msa_mask (N_ens, N_msa, N)
    msa_row_mask (N_ens, N_msa)
    random_crop_to_size_seed (N_ens, 2)
    * template_aatype (N_ens, N_maxtpl, N)
    * template_all_atom_masks (N_ens, N_maxtpl, N, 37)
    * template_all_atom_positions (N_ens, N_maxtpl, N, 37, 3)
    * template_sum_probs (N_ens, N_maxtpl, 1)
    * template_mask (N_ens, N_maxtpl)
    * template_pseudo_beta (N_ens, N_maxtpl, N, 3)
    * template_pseudo_beta_mask (N_ens, N_maxtpl, N)
"""
