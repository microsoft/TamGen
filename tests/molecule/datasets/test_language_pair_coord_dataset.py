#! /usr/bin/python
# -*- coding: utf-8 -*-

import unittest

import torch

from fairseq.data.molecule.language_pair_coord_dataset import LanguagePairCoordinatesDataset
from fairseq.data.dictionary import Dictionary
import tests.utils as test_utils


class TestLanguagePairCoordinateDataset(unittest.TestCase):
    @staticmethod
    def _make_dataset(lines):
        dictionary = Dictionary()
        tokens = [dictionary.encode_line(line) for line in lines]
        dataset = test_utils.TestDataset(tokens)
        sizes = [len(s) for s in tokens]

        return dictionary, dataset, sizes

    def assertTensorEqual(self, t1, t2, msg=None):
        self.assertIsInstance(t1, torch.Tensor, 'First argument is not a torch.Tensor')
        self.assertIsInstance(t2, torch.Tensor, 'Second argument is not a torch.Tensor')
        if not torch.equal(t1, t2):
            standard_msg = 'Tensor not equal'
            self.fail(self._formatMessage(msg, standard_msg))

    def setUp(self) -> None:
        self.addTypeEqualityFunc(torch.Tensor, self.assertTensorEqual)

        src_lines = ['a b c d', 'b', 'd e f']
        src_dict, src_dataset, src_sizes = self._make_dataset(src_lines)

        tgt_lines = ['e c d', 'b c', 'a f e']
        tgt_dict, tgt_dataset, tgt_sizes = self._make_dataset(tgt_lines)

        coordinates = [
            torch.tensor([[.1, .1, .1], [.2, .2, .2], [.3, .3, .3], [.4, .4, .4]], dtype=torch.float32),
            torch.tensor([[.5, .5, .5]], dtype=torch.float32),
            torch.tensor([[.6, .6, .6], [.7, .7, .7], [.8, .8, .8]], dtype=torch.float32),
        ]
        coord_dataset = test_utils.TestDataset(coordinates)

        self.dataset = LanguagePairCoordinatesDataset(
            src_dataset, src_dataset.sizes, src_dict,
            tgt_dataset, tgt_dataset.sizes, tgt_dict,
            left_pad_source=True,
            left_pad_target=True,
            max_source_positions=1024,
            max_target_positions=1024,
            src_coord=coord_dataset,
            coord_mode='raw',
        )

    def testBasic(self):
        sample = self.dataset[0]
        self.assertEqual(sample['id'], 0)
        self.assertEqual(sample['source'].tolist(), [4, 5, 6, 7, 2])
        self.assertEqual(sample['target'].tolist(), [4, 5, 6, 2])
        self.assertEqual(sample['src_coord'].shape, torch.Size([4, 3]))
        self.assertAlmostEqual(sample['src_coord'].sum().item(), 3.0, places=5)

    def testCollate(self):
        samples = [self.dataset[0], self.dataset[1], self.dataset[2]]
        batch = self.dataset.collater(samples)

        expected_batch = {
            'id': torch.LongTensor([0, 2, 1]),
            'net_input': {
                'src_coord': torch.FloatTensor([
                   [[.1, .1, .1], [.2, .2, .2], [.3, .3, .3], [.4, .4, .4]],
                   [[.0, .0, .0], [.6, .6, .6], [.7, .7, .7], [.8, .8, .8]],
                   [[.0, .0, .0], [.0, .0, .0], [.0, .0, .0], [.5, .5, .5]],
                ]),
                'prev_output_tokens': torch.IntTensor([
                    [2, 4, 5, 6],
                    [2, 8, 9, 4],
                    [1, 2, 7, 5],
                ]),
                'src_lengths': torch.LongTensor([5, 4, 2]),
                'src_tokens': torch.IntTensor([
                    [4, 5, 6, 7, 2],
                    [1, 7, 8, 9, 2],
                    [1, 1, 1, 5, 2],
                ])
            },
            'nsentences': 3,
            'ntokens': 11,
            'target': torch.IntTensor([
                [4, 5, 6, 2],
                [8, 9, 4, 2],
                [1, 7, 5, 2],
            ]),
        }

        for k, v in expected_batch.items():
            self.assertIn(k, batch)
            bk = batch[k]
            if isinstance(v, dict):
                self.assertEqual(k, 'net_input')
                for k2, v2 in v.items():
                    self.assertIn(k2, bk)
                    self.assertEqual(bk[k2], v2, msg=f'Key {k}.{k2} mismatch')
            else:
                self.assertEqual(bk, v, msg=f'Key {k} mismatch')


if __name__ == '__main__':
    unittest.main()
