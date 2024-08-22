#! /usr/bin/python
# -*- coding: utf-8 -*-
#
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for protein."""

import unittest
from pathlib import Path

import numpy as np

from fairseq.molecule_utils.af2_utils.common import (
    protein,
    residue_constants,
)

HERE = Path(__file__).absolute().parent.parent
TEST_DATA_DIR = HERE.parent / 'test_data'


def _get_pdb_string():
    with open(TEST_DATA_DIR / '2rbg.pdb', 'r', encoding='utf-8') as f:
        pdb_string = f.read()
    return pdb_string


class TestProtein(unittest.TestCase):
    def _check_shapes(self, prot, num_res):
        """Check that the processed shapes are correct."""
        num_atoms = residue_constants.atom_type_num
        self.assertEqual((num_res, num_atoms, 3), prot.atom_positions.shape)
        self.assertEqual((num_res,), prot.aatype.shape)
        self.assertEqual((num_res, num_atoms), prot.atom_mask.shape)
        self.assertEqual((num_res,), prot.residue_index.shape)
        self.assertEqual((num_res, num_atoms), prot.b_factors.shape)

    def _testFromPdbStr(self, pdb_file, chain_id, num_res):
        pdb_file = TEST_DATA_DIR / pdb_file
        with open(pdb_file, 'r', encoding='utf-8') as f:
            pdb_string = f.read()
        prot = protein.from_pdb_string(pdb_string, chain_id)
        self._check_shapes(prot, num_res)
        self.assertGreaterEqual(prot.aatype.min(), 0)
        # Allow equal since unknown restypes have index equal to restype_num.
        self.assertLessEqual(prot.aatype.max(), residue_constants.restype_num)

    def testFromPdbStr(self):
        self._testFromPdbStr('2rbg.pdb', 'A', 282)
        self._testFromPdbStr('2rbg.pdb', 'B', 282)

    def testToPdb(self):
        pdb_string = _get_pdb_string()
        prot = protein.from_pdb_string(pdb_string, chain_id='A')
        pdb_string_reconstr = protein.to_pdb(prot)
        prot_reconstr = protein.from_pdb_string(pdb_string_reconstr)

        np.testing.assert_array_equal(prot_reconstr.aatype, prot.aatype)
        np.testing.assert_array_almost_equal(
            prot_reconstr.atom_positions, prot.atom_positions)
        np.testing.assert_array_almost_equal(
            prot_reconstr.atom_mask, prot.atom_mask)
        np.testing.assert_array_equal(
            prot_reconstr.residue_index, prot.residue_index)
        np.testing.assert_array_almost_equal(
            prot_reconstr.b_factors, prot.b_factors)

    def testIdealFromMask(self):
        pdb_string = _get_pdb_string()
        prot = protein.from_pdb_string(pdb_string, chain_id='A')
        ideal_mask = protein.ideal_atom_mask(prot)
        non_ideal_residues = set([102] + list(range(127, 285)))
        for i, (res, atom_mask) in enumerate(
                zip(prot.residue_index, prot.atom_mask)):
            if res in non_ideal_residues:
                self.assertFalse(np.all(atom_mask == ideal_mask[i]), msg=f'{res}')
            else:
                self.assertTrue(np.all(atom_mask == ideal_mask[i]), msg=f'{res}')


if __name__ == '__main__':
    unittest.main()
