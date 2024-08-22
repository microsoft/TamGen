#! /usr/bin/python
# -*- coding: utf-8 -*-

from pathlib import PurePosixPath


class CodeGenerationScorer:
    _WEIGHTS = [
        None,
        (1., 0., 0., 0.),
        (1 / 2, 1 / 2, 0., 0.),
        (1 / 3, 1 / 3, 1 / 3, 0),
        (1 / 4, 1 / 4, 1 / 4, 1 / 4),
    ]
    _UNK_REPLACE_STR = 'MASK'

    def __init__(self, task):
        args = task.args
        self.pov_replace_unk = args.codegen_pov_replace_unk

        import nltk
        nltk.download('wordnet', quiet=True)
        from nltk.translate.bleu_score import SmoothingFunction
        self._smooth_fn = SmoothingFunction().method4
        from nltk.translate.bleu_score import sentence_bleu
        self._sentence_bleu_fn = sentence_bleu

        dataset_name = PurePosixPath(task.args.data.split(':')[0]).name
        if dataset_name.startswith('java.'):
            self._language = 'java'
            if dataset_name.endswith('.orig'):
                self._split_method = 'original'
            elif '.imp' in dataset_name:
                self._split_method = 'improved'
            else:
                self._split_method = 'original'

            from . import java_utils
            self._split_string_func = java_utils.split_string
            self._parse_func = java_utils.parse_java
        elif dataset_name.startswith('python.'):
            self._language = 'python'
            # TODO
            raise NotImplementedError('Code generation not implemented for dataset {!r}'.format(dataset_name))
        else:
            raise NotImplementedError('Code generation not implemented for dataset {!r}'.format(dataset_name))

        self.reset()

    def reset(self, one_init=False):
        if one_init:
            raise NotImplementedError
        self.ref = []
        self.sys = []

    def add_string(self, ref, pred):
        self.ref.append(self._split_string_func(ref, method=self._split_method))
        self.sys.append(self._split_string_func(pred, method=self._split_method))

    def score(self, order=4):
        assert order <= 4, "BLEU scores for order > 4 aren't supported"
        weights = self._WEIGHTS[order]
        cum_bleu_score = 0.0
        for ref, sys in zip(self.ref, self.sys):
            if len(sys) == 1:
                continue
            cum_bleu_score += self._sentence_bleu_fn([ref], sys, weights=weights, smoothing_function=self._smooth_fn)
        bleu_score = cum_bleu_score / len(self.sys) * 100.0
        return bleu_score

    def pov(self):
        """Percent of valid code."""
        cum_pov = 0.0
        for sys in self.sys:
            cum_pov += self._parse_func(
                sys, pov_replace_unk=self.pov_replace_unk, unk_replace_str=self._UNK_REPLACE_STR)
        pov = cum_pov / len(self.sys) * 100
        return pov

    def result_string(self, order=4):
        return 'BLEU4 = {:.2f}, PoV = {:.2f}'.format(self.score(order), self.pov())
