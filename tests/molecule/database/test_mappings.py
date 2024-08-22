#! /usr/bin/python
# -*- coding: utf-8 -*-

import unittest

from fairseq.molecule_utils.database import mappings

from tests.molecule import helper


class TestMappings(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pdb_sws_u2p_mapping = helper.get_sample_pdb_sws_u2p(None)

    def testUniprot2Pdb_PdbSws(self):
        pdb_chain_id_list = mappings.uniprot_to_pdb('q975b5', mapping=self.pdb_sws_u2p_mapping)
        self.assertEqual(pdb_chain_id_list, [('2rbg', 'A'), ('2rbg', 'B')])

        first_pdb_chain_id = mappings.uniprot_to_best_pdb(
            'q975b5', mapping=self.pdb_sws_u2p_mapping, policy=mappings.UniProt2PdbChoosePolicy.FIRST)
        self.assertEqual(first_pdb_chain_id, ('2rbg', 'A'))


if __name__ == '__main__':
    unittest.main()
