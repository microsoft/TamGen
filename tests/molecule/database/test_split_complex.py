#! /usr/bin/python
# -*- coding: utf-8 -*-

import unittest
from pathlib import Path
from unittest.mock import patch

from fairseq.molecule_utils.database import split_complex
from Bio.Data.SCOPData import protein_letters_3to1

from tests.molecule import helper

HERE = Path(__file__).absolute().parent
TEST_DATA_PATH = HERE.parent / 'test_data'
SPLIT_PDB_PATH = TEST_DATA_PATH / 'SplitPdb'


class TestSplitComplex(unittest.TestCase):
    @patch('fairseq.molecule_utils.database.split_complex.get_pdb_ccd_info', helper.get_sample_pdb_ccd_info)
    def testBasic(self):
        split_result = split_complex.split_pdb_complex_paths(
            '3ny8', split_ext='.cif', split_cache_path=SPLIT_PDB_PATH, pdb_cache_path=TEST_DATA_PATH,
        )
        self.assertEqual(str(split_result.target_filename.name), '3ny8-no-ligand.cif')
        self.assertEqual(str(split_result.ligand_info_filename.name), '3ny8-ligand-info.json')
        self.assertEqual(len(split_result.ligand_filenames), 7)
        self.assertEqual(str(split_result.ligand_filenames[0].name), '3ny8-ligand0.cif')
        example_info = split_complex.LigandInfo(
            pdb_id='3ny8', ligand_id=0, ccd_id='CLR', model_id=0, chain_id='A', res_id=1201, insertion_code=' ',
        )
        self.assertEqual(split_result.ligand_info[0], example_info)

    @unittest.skip('Only run once, already tested')
    def testGetInvalid(self):
        split_result = split_complex.split_pdb_complex_paths(
            'non-exist', split_ext='.cif', split_cache_path=SPLIT_PDB_PATH, pdb_cache_path=TEST_DATA_PATH,
        )
        self.assertIsNone(split_result.target_filename)
        self.assertEqual(split_result.ligand_filenames, [])
        self.assertIsNone(split_result.ligand_info_filename)
        self.assertEqual(split_result.ligand_info, {})

    @patch('fairseq.molecule_utils.database.split_complex.get_pdb_ccd_info', helper.get_sample_pdb_ccd_info)
    def testGetPairs(self):
        pairs = split_complex.get_target_ligand_pairs_from_pdb(
            '3ny8', pdb_cache_path=TEST_DATA_PATH)
        self.assertEqual(len(pairs), 7)

        self.assertEqual(pairs[0].chain_id, 'A')
        seq_chain_a = pairs[0].chain_fasta
        self.assertEqual(seq_chain_a[85 - 1], protein_letters_3to1['CYS'])
        self.assertEqual(seq_chain_a[89 - 1], protein_letters_3to1['VAL'])
        self.assertEqual(seq_chain_a[159 - 1], protein_letters_3to1['ARG'])
        self.assertEqual(seq_chain_a[166 - 1], protein_letters_3to1['TRP'])

        # pairs_3 = split_complex.get_target_ligand_pairs_from_pdb(
        #     '2zk5')
        # print(pairs_3)
        # from fairseq.molecule_utils.database import get_af2_mmcif_object
        # m = get_af2_mmcif_object('2zk5')
        # print(m.chain_to_seqres.keys())
        # print(m.mmcif_to_author_chain_id)


if __name__ == '__main__':
    unittest.main()
