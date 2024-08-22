# -*- coding: utf-8 -*-

import unittest

import numpy as np

from fairseq.molecule_utils.coordinate_utils import atom_positions as apos
from tests.molecule import helper


class TestAtomPositions(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mmcif_object = helper.get_3ny8_af2_mmcif()

    def test_get_residue_atom_coordinates(self):
        res = self.mmcif_object.structure['A'][78]
        coord = apos.get_residue_atom_coordinates(res)
        for i, atom in enumerate(res.get_atoms()):
            np.testing.assert_array_equal(coord[i], atom.get_coord())

    def test_get_atom_weights(self):
        res = self.mmcif_object.structure['A'][78]
        atom_weights = apos.get_atom_weights(res)
        self.assertEqual(len(res), len(atom_weights))
        np.testing.assert_array_equal(
            atom_weights,
            np.asarray([14.007, 12.011, 12.011, 15.999, 12.011], dtype=np.float32))


if __name__ == '__main__':
    unittest.main()
