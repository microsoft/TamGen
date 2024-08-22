#! /usr/bin/python
# -*- coding: utf-8 -*-

import unittest
from pathlib import Path

from fairseq.molecule_utils.external_tools import autodock_smina

HERE = Path(__file__).absolute().parent
TEST_DATA_PATH = HERE.parent / 'test_data'


class TestAutoDockSmina(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        smina_path = autodock_smina.AutoDockSmina.find_binary()
        if smina_path is not None:
            cls.smina = autodock_smina.AutoDockSmina(binary_path=smina_path, exhaustiveness=32, seed=1234)
        else:
            cls.smina = None
        cls.receptor_path = TEST_DATA_PATH / 'SplitPdb' / 'stub-3ny8-no-ligand.cif'
        cls.ligand_path = TEST_DATA_PATH / 'SplitPdb' / 'stub-3ny8-ligand0.cif'

    def setUp(self) -> None:
        if self.smina is None:
            self.skipTest('Smina binary not found, skip this test.')

    def test_basic(self):
        affinity = self.smina.query(
            receptor_path=self.receptor_path,
            ligand_path=self.ligand_path,
            autobox_ligand_path=self.receptor_path,
            output_complex_path=None,
        )
        self.assertAlmostEqual(affinity, -9.0, places=5)


if __name__ == '__main__':
    unittest.main()
