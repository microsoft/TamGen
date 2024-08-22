#! /usr/bin/python
# -*- coding: utf-8 -*-

import os

import numpy as np
import pickle
import torch

from fairseq.data import (
    LanguagePairCoordinatesDataset,
    data_utils,
    FairseqDataset,
    iterators
)
from fairseq.tasks.translation import TranslationTask, load_langpair_dataset
from .. import register_task


def load_langpair_coord_dataset(data_path,
                                split,
                                src,
                                src_dict,
                                tgt,
                                tgt_dict,
                                combine,
                                dataset_impl,
                                upsample_primary,
                                left_pad_source,
                                left_pad_target,
                                max_source_positions,
                                max_target_positions,
                                coord_mode,
                                load_coord=True,
                                input_target=False,
                                remove_prefix_fragment=False):
    base_dataset = load_langpair_dataset(
        data_path, split,
        src, src_dict, tgt, tgt_dict,
        combine, dataset_impl, upsample_primary,
        left_pad_source, left_pad_target, max_source_positions, max_target_positions,
        remove_eos_from_source=True
    )
    if load_coord:
        coord_filename = os.path.join(data_path, '{}.{}-{}.{}.coord'.format(split, src, tgt, src))
        src_coord = data_utils.load_indexed_dataset(
                    coord_filename,
                    None,
                    dataset_impl='mmap',
                    default='mmap',
                )
    else:
        src_coord = None

    return LanguagePairCoordinatesDataset.from_base_dataset(
        base_dataset, src_coord, coord_mode, input_target, remove_prefix_fragment)


@register_task("translation_coord")
class TranslationCoordinatesPointsTask(TranslationTask):
    """Same as TranslationTask, but use extra 3D coordinates source input.

    Extra input format:
        Source coordinates filename: <split>.<src>-<tgt>.<src>.coord.pickle
        pickleFile key: '0', '1', ..., 'N', in the same order of data
        Expected source coordinates shape: floats of `(src_len, 3)`
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        TranslationTask.add_args(parser)

        parser.add_argument('--coord-mode', default='raw', metavar='MODE',
                            help='coordinates representation mode')
        parser.add_argument('--use-src-coord', default=False, action='store_true',
                            help='use src coord')
        parser.add_argument('--use-tgt-coord', default=False, action='store_true',
                            help='use tgt coord')
        parser.add_argument('--shuffle-input', default=False, action='store_true',
                            help='shuffle input')
        parser.add_argument('--vae', default=False, action='store_true',
                            help='use vae encoder')
        parser.add_argument('--sample-beta', type=float, default=1.0, metavar='D',
                            help='sample beta')
        parser.add_argument('--hint', default=False, action='store_true',
                            help='generate with hint')
        parser.add_argument('--hint-rate', type=float, default=0.5, metavar='D',
                            help='the rate of hint')
        parser.add_argument('--gen-coord-noise', default=False, action='store_true', 
                            help='generate with coord noise')
        parser.add_argument('--gen-rot', default=False, action='store_true', 
                            help='add random rotation to coord when generating')
        parser.add_argument('--gen-vae', default=False, action='store_true', 
                            help='use vae encoder to get z when generating')
        parser.add_argument('--remove-prefix-fragment', default=False, action='store_true', 
                            help='whether remove fragments from input')

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = self.args.data.split(':')
        assert len(paths) > 0
        data_path = paths[epoch % len(paths)]

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang

        self.datasets[split] = load_langpair_coord_dataset(
            data_path,
            split,
            src,
            self.src_dict,
            tgt,
            self.tgt_dict,
            combine=combine,
            dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            coord_mode=self.args.coord_mode,
            load_coord=self.args.use_src_coord,
            input_target=self.args.vae or self.args.gen_vae,
            remove_prefix_fragment=self.args.remove_prefix_fragment
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, src_coord):
        # TODO: Change the function signature, and change the caller in interactive.py to add feature/length.
        raise NotImplementedError()

    def build_generator(self, args):
        if getattr(args, 'score_reference', False):
            from fairseq.sequence_scorer import SequenceScorer
            return SequenceScorer(self.target_dictionary)
        else:
            if args.hint:
                from fairseq.sequence_generator_hint import SequenceGeneratorWithHint
                return SequenceGeneratorWithHint(
                    self.target_dictionary,
                    beam_size=getattr(args, 'beam', 5),
                    max_len_a=getattr(args, 'max_len_a', 0),
                    max_len_b=getattr(args, 'max_len_b', 200),
                    min_len=getattr(args, 'min_len', 1),
                    normalize_scores=(not getattr(args, 'unnormalized', False)),
                    len_penalty=getattr(args, 'lenpen', 1),
                    unk_penalty=getattr(args, 'unkpen', 0),
                    sampling=getattr(args, 'sampling', False),
                    sampling_topk=getattr(args, 'sampling_topk', -1),
                    sampling_topp=getattr(args, 'sampling_topp', -1.0),
                    temperature=getattr(args, 'temperature', 1.),
                    diverse_beam_groups=getattr(args, 'diverse_beam_groups', -1),
                    diverse_beam_strength=getattr(args, 'diverse_beam_strength', 0.5),
                    match_source_len=getattr(args, 'match_source_len', False),
                    no_repeat_ngram_size=getattr(args, 'no_repeat_ngram_size', 0),
                    hint=True,
                    hint_rate=args.hint_rate,
                    left_pad_target=args.left_pad_target,
                )
            from fairseq.sequence_generator import SequenceGenerator
            return SequenceGenerator(
                self.target_dictionary,
                beam_size=getattr(args, 'beam', 5),
                max_len_a=getattr(args, 'max_len_a', 0),
                max_len_b=getattr(args, 'max_len_b', 200),
                min_len=getattr(args, 'min_len', 1),
                normalize_scores=(not getattr(args, 'unnormalized', False)),
                len_penalty=getattr(args, 'lenpen', 1),
                unk_penalty=getattr(args, 'unkpen', 0),
                sampling=getattr(args, 'sampling', False),
                sampling_topk=getattr(args, 'sampling_topk', -1),
                sampling_topp=getattr(args, 'sampling_topp', -1.0),
                temperature=getattr(args, 'temperature', 1.),
                diverse_beam_groups=getattr(args, 'diverse_beam_groups', -1),
                diverse_beam_strength=getattr(args, 'diverse_beam_strength', 0.5),
                match_source_len=getattr(args, 'match_source_len', False),
                no_repeat_ngram_size=getattr(args, 'no_repeat_ngram_size', 0),
            )

    def get_batch_iterator(
        self, dataset, max_tokens=None, max_sentences=None, max_positions=None,
        ignore_invalid_inputs=False, required_batch_size_multiple=1,
        seed=1, num_shards=1, shard_id=0, num_workers=0, epoch=0,
    ):
        assert isinstance(dataset, FairseqDataset)

        with data_utils.numpy_seed(seed):
            if self.args.shuffle_input:
                # get random indices
                indices = np.random.permutation(len(dataset))
            else:
                # get indices ordered by example size
                indices = dataset.ordered_indices()


        # filter examples that are too large
        if max_positions is not None:
            indices = data_utils.filter_by_size(
                indices, dataset.size, max_positions, raise_exception=(not ignore_invalid_inputs),
            )

        # create mini-batches with given size constraints
        batch_sampler = data_utils.batch_by_size(
            indices, dataset.num_tokens, max_tokens=max_tokens, max_sentences=max_sentences,
            required_batch_size_multiple=required_batch_size_multiple,
        )

        # return a reusable, sharded iterator
        return iterators.EpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_sampler=batch_sampler,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
        )
