#! /usr/bin/python
# -*- coding: utf-8 -*-

import unittest

import torch

from fairseq.data.language_pair_feature_dataset import LanguagePairFeatureDataset
from fairseq.data.dictionary import Dictionary
import tests.utils as test_utils


class TestLanguagePairFeatureDataset(unittest.TestCase):
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

        feature = [
            torch.tensor([[.1, .1], [.1, .2], [.3, .4], [.5, .6], [.7, .8], [.9, .0]], dtype=torch.float32),
            torch.tensor([[.1, .1], [.3, .7], [.5, .5]], dtype=torch.float32),
            torch.tensor([[.1, .1], [.2, .22], [.4, .44], [.6, .66], [.8, .88]], dtype=torch.float32),
        ]
        feature_dataset = test_utils.TestDataset(feature)
        length = [t.shape[0] for t in feature]
        length_dataset = test_utils.TestDataset(torch.LongTensor(length))

        self.dataset = LanguagePairFeatureDataset(
            src_dataset, src_dataset.sizes, src_dict,
            tgt_dataset, tgt_dataset.sizes, tgt_dict,
            left_pad_source=True,
            left_pad_target=True,
            max_source_positions=1024,
            max_target_positions=1024,
            feature=feature_dataset,
            length=length_dataset,
        )

    def testBasic(self):
        sample = self.dataset[0]
        self.assertEqual(sample['id'], 0)
        self.assertEqual(sample['source'].tolist(), [4, 5, 6, 7, 2])
        self.assertEqual(sample['target'].tolist(), [4, 5, 6, 2])
        self.assertEqual(sample['feature'].shape, torch.Size([6, 2]))
        self.assertAlmostEqual(sample['feature'].sum().item(), 4.7, places=5)

        if sample['length'] is not None:
            self.assertEqual(sample['length'].item(), 6)

    def testCollate(self):
        samples = [self.dataset[0], self.dataset[1], self.dataset[2]]
        batch = self.dataset.collater(samples)

        expected_batch = {
            'id': torch.LongTensor([0, 2, 1]),
            'net_input': {
                'feature': torch.FloatTensor([
                   [[.1, .1], [.1, .2], [.3, .4], [.5, .6], [.7, .8], [.9, .0]],
                   [[.0, .0], [.1, .1], [.2, .22], [.4, .44], [.6, .66], [.8, .88]],
                   [[.0, .0], [.0, .0], [.0, .0], [.1, .1], [.3, .7], [.5, .5]],
                ]),
                'length': torch.LongTensor([6, 5, 3]),
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
