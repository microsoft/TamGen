#! /usr/bin/python
# -*- coding: utf-8 -*-

import unittest

from fairseq.molecule_utils.basic import smiles_utils


class TestSmilesUtils(unittest.TestCase):
    def test_tokenize_smiles(self):
        smiles = r'C=Cc1c(C)c2cc3nc(c4c5nc(cc6[n-]c(cc1[nH]2)c(C)c6CC)C(C)=C5C(=O)[C@@H]4C(=O)OC)[C@@H](CCC(' \
                 r'=O)OC/C=C(\C)CCC[C@H](C)CCC[C@H](C)CCCC(C)C)[C@@H]3C.[Mg+2] '
        tokenized_smiles = r'C = C c 1 c ( C ) c 2 c c 3 n c ( c 4 c 5 n c ( c c 6 [n-] c ( c c 1 [nH] 2 ) c ( C ) c ' \
                           r'6 C C ) C ( C ) = C 5 C ( = O ) [C@@H] 4 C ( = O ) O C ) [C@@H] ( C C C ( = O ) O C / C ' \
                           r'= C ( \ C ) C C C [C@H] ( C ) C C C [C@H] ( C ) C C C C ( C ) C ) [C@@H] 3 C . [Mg+2]'
        self.assertEqual(smiles_utils.tokenize_smiles(smiles), tokenized_smiles)

    def test_smi_canonical(self):
        smiles = 'C(O)CO'
        smiles_can = smiles_utils.canonicalize_smiles(smiles)
        self.assertEqual(smiles_can, 'OCCO')

    def test_smi_inchi(self):
        smiles = 'OCCO'
        inchi = smiles_utils.smi2inchi(smiles)
        self.assertEqual(inchi, 'InChI=1S/C2H6O2/c3-1-2-4/h3-4H,1-2H2')
        smiles_back = smiles_utils.inchi2smi(inchi)
        self.assertEqual(smiles, smiles_back)

    def test_smi_pdb(self):
        smiles = 'OCCO'
        pdb_str = smiles_utils.smi2pdb(smiles)
        expected_pdb_str = '''\
HETATM    1  O1  UNL     1      -1.421  -0.532  -0.198  1.00  0.00           O  
HETATM    2  C1  UNL     1      -0.693   0.538   0.324  1.00  0.00           C  
HETATM    3  C2  UNL     1       0.699   0.556  -0.277  1.00  0.00           C  
HETATM    4  O2  UNL     1       1.415  -0.561   0.151  1.00  0.00           O  
CONECT    1    2
CONECT    2    3
CONECT    3    4
END
'''
        self.assertEqual(pdb_str, expected_pdb_str)

        pdb_str_0d = smiles_utils.smi2pdb(smiles, compute_coord=False)
        self.assertEqual(pdb_str_0d, '''\
HETATM    1  O1  UNL     1       0.000   0.000   0.000  1.00  0.00           O  
HETATM    2  C1  UNL     1       0.000   0.000   0.000  1.00  0.00           C  
HETATM    3  C2  UNL     1       0.000   0.000   0.000  1.00  0.00           C  
HETATM    4  O2  UNL     1       0.000   0.000   0.000  1.00  0.00           O  
CONECT    1    2
CONECT    2    3
CONECT    3    4
END
''')

    def test_mol_weight(self):
        smiles = 'OCCO'
        weight = smiles_utils.molecular_weight(smiles)
        self.assertAlmostEqual(weight, 62.068, places=6)


if __name__ == '__main__':
    unittest.main()
