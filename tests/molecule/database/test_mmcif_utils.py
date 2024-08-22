#! /usr/bin/python
# -*- coding: utf-8 -*-

import unittest

from fairseq.molecule_utils.database import mmcif_utils

from tests.molecule import helper


class TestMmcifUtils(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mmcif_object = helper.get_3ny8_af2_mmcif()

    def testUniProtRefs(self):
        uniprot_ref_data = mmcif_utils.get_uniprot_ref(self.mmcif_object)
        self.assertEqual(uniprot_ref_data['1'].uniprot_id, 'p07550')
        self.assertEqual(uniprot_ref_data['1'].chain_id, 'A')
        self.assertEqual(uniprot_ref_data['1'].seq_begin, 9)
        self.assertEqual(uniprot_ref_data['1'].seq_end, 238)

    def testBindingSites(self):
        # binding_sites = mmcif_utils.get_binding_sites(self.mmcif_object)
        #
        # self.assertEqual(binding_sites['AC1'].site_id, 'AC1')
        # self.assertEqual(binding_sites['AC1'].pdb_id, '3ny8')
        # self.assertEqual(len(binding_sites['AC1'].binding_positions), 4)
        # self.assertEqual(binding_sites['AC1'].binding_positions[0].chain_id, 'A')
        # self.assertEqual(binding_sites['AC1'].binding_positions[0].res_id, 85)
        # self.assertEqual(binding_sites['AC1'].binding_positions[0].res_name, 'CYS')

        from fairseq.molecule_utils.database import get_af2_mmcif_object
        from fairseq.molecule_utils.basic import aligned_print
        m = get_af2_mmcif_object('6uap')
        print(m.mmcif_to_author_chain_id)
        print(m.chain_to_seqres.keys())
        aligned_print(m.chain_to_seqres['A'])
        sites = mmcif_utils.get_binding_sites(m, auth_id=False)
        for site in sites.values():
            site.pretty_print()


if __name__ == '__main__':
    unittest.main()
