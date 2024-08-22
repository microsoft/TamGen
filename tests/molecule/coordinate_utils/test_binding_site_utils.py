#! /usr/bin/python
# -*- coding: utf-8 -*-

import unittest

import numpy as np

from fairseq.molecule_utils.coordinate_utils import binding_site_utils as bsu
from tests.molecule import helper


class TestBindingSiteUtils(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mmcif_object = helper.get_3ny8_af2_mmcif()

    def test_ligand_nearest_residues(self):
        near_ligand_0 = bsu.residues_near_ligand(
            self.mmcif_object, auth_chain_id='A', ligand_res_id=1201, ligand_id='CLR', threshold=0.0)
        self.assertEqual(near_ligand_0.size, 490)
        self.assertEqual(near_ligand_0.sum(), 0)
        near_ligand_5 = bsu.residues_near_ligand(
            self.mmcif_object, auth_chain_id='A', ligand_res_id=1201, ligand_id='CLR', threshold=5.0)
        self.assertEqual(near_ligand_5.sum(), 13)
        near_ligand_10 = bsu.residues_near_ligand(
            self.mmcif_object, auth_chain_id='A', ligand_res_id=1201, ligand_id='CLR', threshold=10.0)
        self.assertEqual(near_ligand_10.sum(), 58)

        # High threshold should include low threshold.
        self.assertTrue(np.all(near_ligand_0 <= near_ligand_5))
        self.assertTrue(np.all(near_ligand_5 <= near_ligand_10))

    def test_site_nearest_residues(self):
        near_ac1_0 = bsu.residues_near_site(self.mmcif_object, auth_chain_id='A', site_ids=['AC1'], threshold=0.0)
        self.assertEqual(near_ac1_0.size, 490)
        self.assertEqual(near_ac1_0.sum(), 4)
        near_ac1_5 = bsu.residues_near_site(self.mmcif_object, auth_chain_id='A', site_ids=['AC1'], threshold=5.0)
        self.assertEqual(near_ac1_5.sum(), 41)
        near_ac1_10 = bsu.residues_near_site(self.mmcif_object, auth_chain_id='A', site_ids=['AC1'], threshold=10.0)
        self.assertEqual(near_ac1_10.sum(), 91)

        # High threshold should include low threshold.
        self.assertTrue(np.all(near_ac1_0 <= near_ac1_5))
        self.assertTrue(np.all(near_ac1_5 <= near_ac1_10))

        near_all_10 = bsu.residues_near_site(self.mmcif_object, auth_chain_id='A', site_ids=None, threshold=10.0)
        self.assertEqual(near_all_10.sum(), 268)


if __name__ == '__main__':
    unittest.main()
