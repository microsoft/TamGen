#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Datasets consisting of proteins (numpy version)."""
from typing import Mapping, Sequence, Dict, Optional

import numpy as np

from . import protein_features

ProteinDatasetDict = Dict[str, np.ndarray]


def parse_reshape_logic(
        parsed_features: ProteinDatasetDict,
        features: protein_features.FeaturesMetadata,
        key: Optional[str] = None) -> ProteinDatasetDict:
    """Transforms parsed serial features to the correct shape.

    TODO: This function can be simplified (by define `num_xxx` directly instead of parsing from features).
    """

    num_residues = int(parsed_features['seq_length'].flat[0])

    if "num_alignments" in parsed_features:
        num_msa = int(parsed_features['num_alignments'].flat[0])
    else:
        num_msa = 0

    if "template_domain_names" in parsed_features:
        num_templates = parsed_features['template_domain_names'].shape[0]
    else:
        num_templates = 0

    if key is not None and "key" in features:
        parsed_features["key"] = [key]  # Expand dims from () to (1,).

    # Reshape the tensors according to the sequence length and num alignments.
    for k, v in parsed_features.items():
        new_shape = protein_features.shape(
            feature_name=k,
            num_residues=num_residues,
            msa_length=num_msa,
            num_templates=num_templates,
            features=features)
        new_shape_size = 1
        for dim in new_shape:
            new_shape_size *= dim

        if "template" not in k:
            parsed_features[k] = np.reshape(v, new_shape)
            # Make sure the feature we are reshaping is not empty.
            assert np.size(v) > 0, (
                f"The feature {k} is not set in the example. Either do not request the feature or use an example that "
                "has the feature set. ")
            assert np.size(v) == new_shape_size, \
                f"The size of feature {k} ({np.size(v)}) could not be reshaped into {new_shape}"
        else:
            parsed_features[k] = np.reshape(v, new_shape)
            assert np.size(v) == new_shape_size, \
                f"The size of feature {k} ({np.size(v)}) could not be reshaped into {new_shape}"
    return parsed_features


def _make_features_metadata(
        feature_names: Sequence[str]) -> protein_features.FeaturesMetadata:
    """Makes a feature name to type and shape mapping from a list of names."""
    # Make sure these features are always read.
    required_features = ["aatype", "sequence", "seq_length"]
    feature_names = list(set(feature_names) | set(required_features))

    features_metadata = {name: protein_features.FEATURES[name]
                         for name in feature_names}
    return features_metadata


def make_proteins_dataset_dict(
        np_example: Mapping[str, np.ndarray],
        features: Sequence[str] = None,
) -> ProteinDatasetDict:
    """Creates protein dataset dict from a dict of NumPy arrays.

    Args:
      np_example: A dict of NumPy feature arrays.
      features: A list of strings of feature names to be returned in the dataset.

    Returns:
      A dictionary of features mapping feature names to features. Only the given
      features are returned, all other ones are filtered out.
    """
    if features is None:
        features = list(protein_features.FEATURES.keys())
    features_metadata = _make_features_metadata(features)
    protein = {k: v for k, v in np_example.items() if k in features_metadata}

    # Ensures shapes are as expected. Needed for setting size of empty features
    # e.g. when no template hits were found.
    protein = parse_reshape_logic(protein, features_metadata)
    return protein
