#! /usr/bin/python
# -*- coding: utf-8 -*-

import unittest
from pathlib import Path

import numpy as np

from fairseq.molecule_utils.database import af2_mmcif_parsing

HERE = Path(__file__).absolute().parent
TEST_DATA_PATH = HERE.parent / 'test_data'


class TestAF2MmcifParsing(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with (TEST_DATA_PATH / '2rbg.cif').open('r', encoding='utf-8') as cif_f:
            cls.mmcif_string = cif_f.read()

    def testAF2MmcifParsing(self):
        parsing_result = af2_mmcif_parsing.parse_string(file_id='2rbg', mmcif_string=self.mmcif_string)

        mmcif_object = parsing_result.mmcif_object
        self.assertEqual(mmcif_object.file_id, '2rbg')
        self.assertEqual(mmcif_object.header['resolution'], 1.75)

        structure = mmcif_object.structure
        atom0 = next(structure.get_atoms())
        self.assertEqual(atom0.name, 'N')
        self.assertEqual(atom0.parent.resname, 'TYR')
        np.testing.assert_almost_equal(atom0.coord, [33.471, 9.062, 24.101], decimal=5)


if __name__ == '__main__':
    unittest.main()
