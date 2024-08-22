#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Test helpers."""

from pathlib import Path
from unittest.mock import patch

from fairseq.molecule_utils.database import get_af2_mmcif_object
from fairseq.molecule_utils.database.split_complex import CcdInfo

HERE = Path(__file__).absolute().parent
TEST_DATA_PATH = HERE / 'test_data'


def get_3ny8_af2_mmcif():
    return get_af2_mmcif_object('3ny8', pdb_cache_path=TEST_DATA_PATH)


def get_sample_pdb_ccd_info(_):
    return {
        'CLR': CcdInfo(
            'CLR', 'CHOLESTEROL', 'CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CC=C4C[C@@H](O)CC[C@]4(C)[C@H]3CC[C@]12C',
            'C27 H46 O', 386.654, 'InChI=1S/C27H46O/c1-18(2)7-6-8-19(3)23-11-12-24-22-10-9-20-17-21(28)13-15-26(20,'
                                  '4)25(22)14-16-27(23,24)5/h9,18-19,21-25,28H,6-8,10-17H2,1-5H3/t19-,21+,22+,23-,'
                                  '24+,25+,26+,27-/m1/s1'),
        'JRZ': CcdInfo('JRZ', '(2S,3S)-1-[(7-methyl-2,3-dihydro-1H-inden-4-yl)oxy]-3-[(1-methylethyl)amino]butan-2-ol',
                       'CC(C)N[C@@H](C)[C@H](O)COc1ccc(C)c2CCCc12', 'C17 H27 N O2', 277.402,
                       'InChI=1S/C17H27NO2/c1-11(2)18-13(4)16(19)10-20-17-9-8-12(3)14-6-5-7-15(14)17/h8-9,11,13,16,'
                       '18-19H,5-7,10H2,1-4H3/t13-,16+/m0/s1'),
        'OLA': CcdInfo('OLA', 'OLEIC ACID', r'CCCCCCCC\C=C/CCCCCCCC(O)=O', 'C18 H34 O2', 132.158,
                       'InChI=1S/C18H34O2/c1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18(19)20/h9-10H,2-8,11-17H2,1H3,'
                       '(H,19,20)/b10-9-'),
        'OLC': CcdInfo('OLC', '(2R)-2,3-dihydroxypropyl (9Z)-octadec-9-enoate',
                       r'CCCCCCCC\C=C/CCCCCCCC(=O)OC[C@H](O)CO', 'C21 H40 O4', 356.54,
                       'InChI=1S/C21H40O4/c1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-21(24)25-19-20(23)18-22/h9-10,'
                       '20,22-23H,2-8,11-19H2,1H3/b10-9-/t20-/m1/s1'),
        'PGE': CcdInfo('PGE', 'TRIETHYLENE GLYCOL', 'OCCOCCOCCO', 'C9 H20 O4', 192.253,
                       'InChI=1S/C6H14O4/c7-1-3-9-5-6-10-4-2-8/h7-8H,1-6H2'),
    }


def get_sample_pdb_sws_u2p(_):
    return {
        'q975b5': [('2rbg', 'A'), ('2rbg', 'B')],
    }
