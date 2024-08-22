# -*- coding: utf-8 -*-

import unittest

import torch

import tests.utils as test_utils
from fairseq.data.dictionary import Dictionary
from fairseq.data.molecule.translation_protein_dataset import TranslationProteinDataset


def _make_dataset(lines):
    dictionary = Dictionary()
    tokens = [dictionary.encode_line(line) for line in lines]
    dataset = test_utils.TestDataset(tokens)
    sizes = [len(s) for s in tokens]

    return dictionary, dataset, sizes


class TestTranslationProteinDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        src_lines = ['a b c d', 'b', 'd e f']
        src_dict, src_dataset, src_sizes = _make_dataset(src_lines)

        tgt_lines = ['e c d', 'b c', 'a f e']
        tgt_dict, tgt_dataset, tgt_sizes = _make_dataset(tgt_lines)

        sites = [
            torch.tensor([0, 1, 1, 0], dtype=torch.int64),
            torch.tensor([1], dtype=torch.int64),
            torch.tensor([1, 0, 1], dtype=torch.int64),
        ]
        sites_dataset = test_utils.TestDataset(sites)

        coordinates = [
            torch.tensor([[.1, .1, .1], [.2, .2, .2], [.3, .3, .3], [.4, .4, .4]], dtype=torch.float32),
            torch.tensor([[.5, .5, .5]], dtype=torch.float32),
            torch.tensor([[.6, .6, .6], [.7, .7, .7], [.8, .8, .8]], dtype=torch.float32),
        ]
        coord_dataset = test_utils.TestDataset(coordinates)

        # TODO: src_feature_dataset

        cls.dataset = TranslationProteinDataset(
            src_dataset, src_dataset.sizes, src_dict,
            tgt_dataset, tgt_dataset.sizes, tgt_dict,
            left_pad_source=True,
            left_pad_target=True,
            max_source_positions=1024,
            max_target_positions=1024,
            sites_dataset=sites_dataset, coord_dataset=coord_dataset,
            src_feature_dataset=None,
        )

    def assertTensorEqual(self, t1, t2, msg=None):
        self.assertIsInstance(t1, torch.Tensor, 'First argument is not a torch.Tensor')
        self.assertIsInstance(t2, torch.Tensor, 'Second argument is not a torch.Tensor')
        if not torch.equal(t1, t2):
            standard_msg = 'Tensor not equal'
            self.fail(self._formatMessage(msg, standard_msg))

    def setUp(self) -> None:
        self.addTypeEqualityFunc(torch.Tensor, self.assertTensorEqual)

    def testBasic(self):
        sample = self.dataset[0]
        self.assertEqual(sample['id'], 0)
        self.assertEqual(sample['source'].tolist(), [4, 5, 6, 7, 2])
        self.assertEqual(sample['target'].tolist(), [4, 5, 6, 2])
        self.assertEqual(sample['src_sites'].tolist(), [0, 1, 1, 0])
        self.assertEqual(sample['src_coord'].shape, torch.Size([4, 3]))
        self.assertAlmostEqual(sample['src_coord'].sum().item(), 3.0, places=5)

    def testBasicCollate(self):
        # TODO
        pass


if __name__ == '__main__':
    unittest.main()
