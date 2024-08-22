#! /usr/bin/python
# -*- coding: utf-8 -*-

import copy
import pickle as pkl
import unittest
from pathlib import Path

import numpy as np

from fairseq.molecule_utils.af2_utils import (
    proteins_dataset,
    data_transforms,
)


def _load_pkl(path: Path):
    if path.exists():
        with open(path, 'rb') as f:
            return pkl.load(f)
    return None


def _print_template_features(protein: dict, title: str = ''):
    print(f'[Template Features] {title}')
    for key, value in protein.items():
        if key.startswith('template_'):
            if value.dtype == 'object':
                example_value = value.flat[0]
            else:
                example_value = np.mean(value)
            print(f'{key}: {value.shape}, {value.dtype}, {example_value}')
    print(f'[Template Features End] {title}')


def _assert_templates_equal(protein, ref_protein):
    for key in protein:
        if key.startswith('template_'):
            np.testing.assert_equal(protein[key], ref_protein[key], err_msg=f'Value of key {key} mismatch')


class TestAF2TemplateProcessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # [NOTE] Download from blob path: /blob/v-yaf/off-AF2/output/drugbank-test/1h00/*.pkl
        here = Path(__file__).absolute().parent
        cls.features = _load_pkl(here / 'features.pkl')
        cls.ne_features = _load_pkl(here / 'nonensembled_processed_features.pkl')
        cls.processed_features = _load_pkl(here / 'processed_features.pkl')

    def setUp(self) -> None:
        if self.features is None or self.processed_features is None or self.ne_features is None:
            self.skipTest('Input features file not found.')
        self.protein = copy.deepcopy(self.features)

    def testExtractTemplateFeatures(self):
        # TODO
        pass

    @unittest.skip('Only run once, already tested')
    def testEnsemble(self):
        """Check that the final feature is a duplicate of N_ens ensembles."""
        n_ensemble = 4
        for key, value in self.processed_features.items():
            if not key.startswith('template_'):
                continue
            self.assertEqual(value.shape[0], n_ensemble)
            for i in range(4):
                np.testing.assert_equal(value[i], value[0])

    def testDataTransforms(self):
        protein = self.protein
        protein = proteins_dataset.make_proteins_dataset_dict(protein)
        protein = data_transforms.apply_transforms(protein)

        _assert_templates_equal(protein, self.processed_features)


if __name__ == '__main__':
    unittest.main()
