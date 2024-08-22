#! /usr/bin/python
# -*- coding: utf-8 -*-

import unittest
from pathlib import Path

import numpy as np

from fairseq.molecule_utils.af2_utils import parse_templates
from fairseq.molecule_utils.database import get_af2_mmcif_object

HERE = Path(__file__).absolute().parent
TEST_DATA_PATH = HERE.parent / 'test_data'


class TestParseTemplates(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mmcif_object = get_af2_mmcif_object('2rbg', pdb_cache_path=TEST_DATA_PATH)
        here = Path(__file__).absolute().parent
        path = here / 'template_outputs_2rbg_A_features.npz'
        if not path.exists():
            cls.features = None
        else:
            cls.features: dict = dict(np.load(str(path)).items())

    def setUp(self) -> None:
        if self.features is None:
            self.skipTest('Input features file not found.')

    def testGetAtomPositions(self):
        all_positions, all_positions_mask = parse_templates.get_atom_positions(
            self.mmcif_object, auth_chain_id='A', max_ca_ca_distance=150.0,
        )
        np.testing.assert_equal(all_positions, self.features['template_all_atom_positions'])
        np.testing.assert_equal(all_positions_mask, self.features['template_all_atom_masks'])
        print(self.mmcif_object.chain_to_seqres)
        print(all_positions.shape)
        print(all_positions_mask.shape)
        for key, value in self.features.items():
            print(key, value.shape, value.dtype)


if __name__ == '__main__':
    unittest.main()
