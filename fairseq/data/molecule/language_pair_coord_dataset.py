#! /usr/bin/python
# -*- coding: utf-8 -*-

from ..language_pair_dataset import LanguagePairDataset, collate
from ..data_utils import collate_tokens_multidim, collate_tokens

import torch

def collate_coordinates(
        samples, batch, sort_order, coord_mode,
        left_pad_source=True):
    if len(samples) == 0:
        return batch

    if coord_mode != 'raw':
        raise NotImplementedError

    def merge(key, left_pad):
        return collate_tokens_multidim(
            [s[key] for s in samples],
            pad_value=0.0, left_pad=left_pad, move_eos_to_beginning=False,
        )

    if samples[0].get('src_coord', None) is not None:
        feature = merge('src_coord', left_pad=left_pad_source)
        feature = feature.index_select(0, sort_order)
        batch['net_input']['src_coord'] = feature

    return batch

def collate_target(samples, batch, input_target=False, 
                   remove_prefix_fragment=False, tgt_dict=None):
    """ prepare the target sequence for VAE einput
    split_tokens 
    """
    if len(samples) == 0:
        return batch
    if not input_target:
        return batch
    if not remove_prefix_fragment:
        batch['net_input']['tgt_tokens'] = batch['target']
    else:
        trunc_tgt_tokens = []
        trunc_tgt_tokens.append(tgt_dict.indices['[cond-generation]'])
        trunc_tgt_tokens.append(tgt_dict.indices['[generation]'])
        mask = torch.isin(batch['target'], torch.tensor(trunc_tgt_tokens))
        indices = torch.nonzero(mask)
        new_tgt_tokens = []
        for row in indices:
            rid = row[0]
            cid = row[1] + 1
            new_tgt_tokens.append(batch['target'][rid][cid:])
        
        batch['net_input']['tgt_tokens'] = collate_tokens(new_tgt_tokens, tgt_dict.pad())
        
    return batch



class LanguagePairCoordinatesDataset(LanguagePairDataset):
    """LanguagePairDataset + 3D coordinates.

    Args:
        src_coord (torch.utils.data.Dataset): source coordinates dataset to wrap
            Each item is a float tensor in shape `(src_len, 3)`.
    """

    AVAILABLE_COORD_MODES = ('flatten', 'raw')

    def __init__(self, *args, **kwargs):
        src_coord = kwargs.pop('src_coord', None)
        coord_mode = kwargs.pop('coord_mode', 'raw')
        input_target = kwargs.pop('input_target', False)
        remove_prefix_fragment = kwargs.pop('remove_prefix_fragment', False)

        if coord_mode not in self.AVAILABLE_COORD_MODES:
            raise ValueError(f'Unknown coordinate mode {coord_mode}')

        super().__init__(*args, **kwargs)

        self.src_coord = src_coord
        self.coord_mode = coord_mode
        self.input_target = input_target
        self.remove_prefix_fragment = remove_prefix_fragment

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        src_coord_item = self.src_coord[index] if self.src_coord is not None else None
        sample['src_coord'] = src_coord_item
        return sample

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        See `LanguagePairDataset.collater` for more details.

        Returns:
            dict: a mini-batch with the keys in `LanguagePairDataset.collater` and following *extra* keys:

                - `src_coord` (FloatTensor): an 3D Tensor of coordinates of source tokens.
        """
        try:
            batch, sort_order = collate(
                samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
                left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
                input_feeding=self.input_feeding, return_sort_order=True,
            )
        except ValueError:
            return {}
            
        collate_coordinates(
            samples, batch, sort_order, self.coord_mode,
            left_pad_source=self.left_pad_source,
        )
        collate_target(samples, batch, input_target=self.input_target, 
                       tgt_dict=self.tgt_dict, remove_prefix_fragment=self.remove_prefix_fragment)
        return batch

    @property
    def supports_prefetch(self):
        return (
            super().supports_prefetch
            and (getattr(self.src_coord, 'supports_prefetch', False) or self.src_coord is None)
        )

    def prefetch(self, indices):
        super().prefetch(indices)
        if self.src_coord is not None:
            self.src_coord.prefetch(indices)

    @classmethod
    def from_base_dataset(cls,
                          base,
                          src_coord=None,
                          coord_mode='raw',
                          input_target=False,
                          remove_prefix_fragment=False):
        """Create dataset from base dataset.

        Args:
            base (LanguagePairDataset): the original dataset
            src_coord (torch.utils.data.Dataset): source coordinates dataset to wrap
            coord_mode (str): coordinates representation mode

        Returns:
            LanguagePairCoordinatesDataset:
        """

        return cls(base.src,
                   base.src_sizes,
                   base.src_dict,
                   tgt=base.tgt,
                   tgt_sizes=base.tgt_sizes,
                   tgt_dict=base.tgt_dict,
                   left_pad_source=base.left_pad_source,
                   left_pad_target=base.left_pad_target,
                   max_source_positions=base.max_source_positions,
                   max_target_positions=base.max_target_positions,
                   shuffle=base.shuffle,
                   input_feeding=base.input_feeding,
                   remove_eos_from_source=base.remove_eos_from_source,
                   append_eos_to_target=base.append_eos_to_target,
                   src_coord=src_coord,
                   coord_mode=coord_mode,
                   input_target=input_target,
                   remove_prefix_fragment=remove_prefix_fragment)
