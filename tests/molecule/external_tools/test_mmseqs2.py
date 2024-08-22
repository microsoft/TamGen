# -*- coding: utf-8 -*-

import unittest

from fairseq.molecule_utils.external_tools import mmseqs2


class TestMMSeqs2(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        mmseqs_path = mmseqs2.MMSeqs2.find_binary()
        if mmseqs_path is not None:
            cls.mmseqs2 = mmseqs2.MMSeqs2(binary_path=mmseqs_path)
        else:
            cls.mmseqs2 = None
        cls.entries = [
            ('PDB', 0, 'AAAAABBBBBCCCCCDDDDDEEEEE'),
            ('PDB', 1, 'AAAAABBBBBCCCCCDDDDDEEEEF'),
            ('CrossDocked', 0, 'AAAAABBBBBCCCCCCDDDDDEEEE'),
            ('CrossDocked', 1, 'RUNNINGTESTMMSEQSTWO'),
            ('CrossDocked', 2, 'THISISSOMERANDOMSTRING'),
            ('DrugBank', 0, 'RUNNINGTESTMMSEQSTHREE'),
            ('DrugBank', 1, 'THATISSOMERANDOMSTRING'),
            ('DrugBank', 2, 'THISISALONELYSTRING'),
        ]

    def setUp(self) -> None:
        if self.mmseqs2 is None:
            self.skipTest('MMseqs2 binary not found, skip this test.')

    def test_basic(self):
        cluster_result = self.mmseqs2.cluster(self.entries, dataset_priority=['DrugBank', 'CrossDocked', 'PDB'])
        self.assertEqual(cluster_result, {'PDB': [], 'CrossDocked': [0], 'DrugBank': [0, 1, 2]})

        cluster_result2 = self.mmseqs2.cluster(self.entries, dataset_priority=['PDB', 'CrossDocked', 'DrugBank'])
        self.assertEqual(cluster_result2, {'PDB': [0], 'CrossDocked': [1, 2], 'DrugBank': [2]})


if __name__ == '__main__':
    unittest.main()
