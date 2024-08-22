#! /usr/bin/python
# -*- coding: utf-8 -*-

import unittest
from pathlib import Path

import numpy as np

from fairseq.molecule_utils import database

HERE = Path(__file__).absolute().parent
TEST_DATA_PATH = HERE.parent / 'test_data'


class TestGetFiles(unittest.TestCase):
    def testGetPdb(self):
        structure = database.get_pdb_structure('2rbg', pdb_cache_path=TEST_DATA_PATH)

        atom0 = next(structure.get_atoms())
        self.assertEqual(atom0.name, 'N')
        self.assertEqual(atom0.parent.resname, 'TYR')
        np.testing.assert_almost_equal(atom0.coord, [33.471, 9.062, 24.101], decimal=5)

    def testGetMmcifObject(self):
        mmcif_object = database.get_af2_mmcif_object('2rbg', pdb_cache_path=TEST_DATA_PATH)
        self.assertEqual(mmcif_object.header['structure_method'], 'x-ray diffraction')
        self.assertEqual(mmcif_object.header['release_date'], '2008-09-30')
        self.assertAlmostEqual(mmcif_object.header['resolution'], 1.75, delta=1e-5)

    def testGetFastaFromUniProt(self):
        fasta_str = database.get_fasta_from_uniprot('q975b5', uniprot_fasta_cache_path=TEST_DATA_PATH, get_str=True)
        fasta_record = database.get_fasta_from_uniprot('q975b5', uniprot_fasta_cache_path=TEST_DATA_PATH, get_str=False)
        self.assertEqual(
            fasta_str,
            'MPYKNILTLISVNNDNFENYFRKIFLDVRSSGSKKTTINVFTEIQYQELVTLIREALLENIDIGYELFLWKKNEVDIFLKNLEKSEVDGLLVYCDDENKVFMSKI'
            'VDNLPTAIKRNLIKDFCRKLS')
        self.assertEqual(str(fasta_record.seq), fasta_str)

    @unittest.skip('Only run once, already tested')
    def testGetInvalid(self):
        with self.assertRaises(FileNotFoundError):
            database.get_pdb_structure('non-exist', pdb_cache_path=TEST_DATA_PATH)
        with self.assertRaises(FileNotFoundError):
            database.get_fasta_from_uniprot('non-exist', uniprot_fasta_cache_path=TEST_DATA_PATH)

    def testWrongPdb(self):
        with self.assertRaises(KeyError) as cm:
            database.get_pdb_structure('wrong', pdb_cache_path=TEST_DATA_PATH)
        error = cm.exception
        self.assertEqual(str(error), "'_atom_site.id'")


if __name__ == '__main__':
    unittest.main()
