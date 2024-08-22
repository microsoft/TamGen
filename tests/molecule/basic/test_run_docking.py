# -*- coding: utf-8 -*-

import unittest
from unittest.mock import patch
from pathlib import Path

from fairseq.molecule_utils.basic import run_docking
from fairseq.molecule_utils.external_tools.autodock_smina import AutoDockSmina

from tests.molecule import helper

HERE = Path(__file__).absolute().parent
TEST_DATA_PATH = HERE.parent / 'test_data'


class TestRunDocking(unittest.TestCase):
    @patch('fairseq.molecule_utils.database.split_complex.get_pdb_ccd_info', helper.get_sample_pdb_ccd_info)
    def test_basic(self):
        if AutoDockSmina.find_binary() is None:
            self.skipTest('AutoDock-smina not found, skip this test.')

        affinity = run_docking.docking(pdb_id='3ny8',
                                       ligand_smiles='CC(C)CCCC(C)C1CCC2C3CCC4CC(O)CC[C@]4(C)C3CC[C@]12C',
                                       smina_bin_path=None, split_cache_path=TEST_DATA_PATH / 'SplitPdb',
                                       pdb_cache_path=TEST_DATA_PATH, ccd_cache_path=None)
        self.assertAlmostEqual(affinity, -8.7, places=5)


if __name__ == '__main__':
    unittest.main()
