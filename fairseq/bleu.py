# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import ctypes
import math
import torch

from .codegen_utils.codegen_scorer import CodeGenerationScorer

try:
    from fairseq import libbleu
except ImportError as e:
    import sys
    sys.stderr.write('ERROR: missing libbleu.so. run `pip install --editable .`\n')
    raise e


C = ctypes.cdll.LoadLibrary(libbleu.__file__)


class BleuStat(ctypes.Structure):
    _fields_ = [
        ('reflen', ctypes.c_size_t),
        ('predlen', ctypes.c_size_t),
        ('match1', ctypes.c_size_t),
        ('count1', ctypes.c_size_t),
        ('match2', ctypes.c_size_t),
        ('count2', ctypes.c_size_t),
        ('match3', ctypes.c_size_t),
        ('count3', ctypes.c_size_t),
        ('match4', ctypes.c_size_t),
        ('count4', ctypes.c_size_t),
    ]


class SacrebleuScorer(object):
    def __init__(self):
        import sacrebleu
        self.sacrebleu = sacrebleu
        self.reset()

    def reset(self, one_init=False):
        if one_init:
            raise NotImplementedError
        self.ref = []
        self.sys = []

    def add_string(self, ref, pred):
        self.ref.append(ref)
        self.sys.append(pred)

    def score(self, order=4):
        return self.result_string(order).score

    def result_string(self, order=4):
        if order != 4:
            raise NotImplementedError
        return self.sacrebleu.corpus_bleu(self.sys, [self.ref])


class SummarizationScorer(object):
    _WEIGHTS = [
        None,
        (1., 0., 0., 0.),
        (.5, .5, 0., 0.),
        (1 / 3, 1 / 3, 1 / 3, 0),
        (.25, .25, .25, .25),
    ]

    def __init__(self):
        import nltk
        from nltk.translate.bleu_score import SmoothingFunction
        self._smooth_fn = SmoothingFunction().method4
        self.reset()

        nltk.download('wordnet', quiet=True)
    
    def reset(self, one_init=False):
        if one_init:
            raise NotImplementedError
        self.ref = []
        self.sys = []
        self.ref_strings = []
        self.sys_strings = []
    
    def add_string(self, ref, pred):
        self.ref_strings.append(ref)
        self.sys_strings.append(pred)
        self.ref.append(ref.split())
        self.sys.append(pred.split())
    
    def score(self, order=4):
        from nltk.translate.bleu_score import sentence_bleu

        assert order <= 4, "BLEU scores for order > 4 aren't supported"
        weights = self._WEIGHTS[order]
        cum_bleu_score = 0.0
        for ref, sys in zip(self.ref, self.sys):
            if len(sys) == 1:
                continue
            cum_bleu_score += sentence_bleu([ref], sys, weights=weights, smoothing_function=self._smooth_fn)
        bleu_score = cum_bleu_score / len(self.sys) * 100.0
        return bleu_score
    
    def word_accuracy(self):
        cum_acc = 0
        for ref, sys in zip(self.ref, self.sys):
            cum_acc += sum(1 for ref_w, sys_w in zip(ref, sys) if ref_w == sys_w) / float(len(sys) + 1e-6)
        acc = cum_acc / len(self.sys) * 100.0
        return acc

    def sentence_accuracy(self):
        cum_acc = 0
        for ref, sys in zip(self.ref, self.sys):
            cum_acc += 1. if all(ref_w == sys_w for ref_w, sys_w in zip(ref, sys)) else 0.
        acc = cum_acc / len(self.sys) * 100.0
        return acc
    
    def meteor_score(self):
        from nltk.translate.meteor_score import single_meteor_score
        cum_meteor = 0

        for ref_string, sys_string in zip(self.ref_strings, self.sys_strings):
            cum_meteor += single_meteor_score(ref_string, sys_string)
        meteor = cum_meteor / len(self.sys_strings) * 100.0
        return meteor
    
    def rouge_l_recall_score(self):
        # NOTE: Use recall, follow previous works.
        from rouge import Rouge

        scorer = Rouge()
        scores = scorer.get_scores(self.sys_strings, self.ref_strings, avg=True)
        return scores['rouge-l']['r'] * 100.0
    
    def result_string(self, order=4):
        return 'BLEU4 = {:.2f}, WordAcc = {:.2f}, SentAcc = {:.2f}, Meteor = {:.2f}, Rouge-L = {:.2f}'.format(
            self.score(order), self.word_accuracy(), self.sentence_accuracy(), self.meteor_score(), self.rouge_l_recall_score())


# class CodeGenerationScorer(object):
#     _WEIGHTS = [
#         None,
#         (1., 0., 0., 0.),
#         (.5, .5, 0., 0.),
#         (1 / 3, 1 / 3, 1 / 3, 0),
#         (.25, .25, .25, .25),
#     ]
#     _UNK_REPLACE_STR = 'MASK'
#
#     def __init__(self, task, pov_replace_unk):
#         import nltk
#         from nltk.translate.bleu_score import SmoothingFunction
#         self._smooth_fn = SmoothingFunction().method4
#         nltk.download('wordnet', quiet=True)
#
#         self.pov_replace_unk = pov_replace_unk
#
#         dataset_name = PurePosixPath(task.args.data.split(':')[0]).name
#         if dataset_name.startswith('java'):
#             import javalang
#             self._javalang = javalang
#             self._parse_method = self._parse_java
#         elif dataset_name.startswith('python'):
#             self._javalang = None
#             self._parse_method = self._parse_python
#
#             # To fix the bug of original dataset (a temporarily solution).
#             self._fix_python_orig_bug = dataset_name.endswith('orig')
#         else:
#             raise RuntimeError('code generation scorer requires only support Java and Python now, '
#                                'dataset name must be started with "java" or "python"')
#
#         self.reset()
#
#     def reset(self, one_init=False):
#         if one_init:
#             raise NotImplementedError
#         self.ref = []
#         self.sys = []
#
#     def add_string(self, ref, pred):
#         self.ref.append(ref.split())
#         self.sys.append(pred.split())
#
#     def score(self, order=4):
#         from nltk.translate.bleu_score import sentence_bleu
#
#         assert order <= 4, "BLEU scores for order > 4 aren't supported"
#         weights = self._WEIGHTS[order]
#         cum_bleu_score = 0.0
#         for ref, sys in zip(self.ref, self.sys):
#             if len(sys) == 1:
#                 continue
#             cum_bleu_score += sentence_bleu([ref], sys, weights=weights, smoothing_function=self._smooth_fn)
#         bleu_score = cum_bleu_score / len(self.sys) * 100.0
#         return bleu_score
#
#     def _parse_java(self, sys_tokens):
#         _replace_unk = self.pov_replace_unk
#         _unk_replace_s = self._UNK_REPLACE_STR
#         normalized_tokens = []
#         for token in sys_tokens:
#             if token == '_NUM':
#                 new_token = '0'
#             elif token == '_STR':
#                 new_token = '""'
#             elif token == '_BOOL':
#                 new_token = 'true'
#             elif _replace_unk and token == '<unk>':
#                 new_token = _unk_replace_s
#             else:
#                 new_token = token
#             normalized_tokens.append(new_token)
#         try:
#             self._javalang.parse.parse_member_signature(' '.join(normalized_tokens))
#         except self._javalang.parser.JavaParserBaseException:
#             return 0
#         else:
#             return 1
#
#     def _parse_python(self, sys_tokens):
#         _replace_unk = self.pov_replace_unk
#         _unk_replace_s = self._UNK_REPLACE_STR
#         _fix_python_orig_bug = self._fix_python_orig_bug
#
#         normalized_tokens = []
#         for token in sys_tokens:
#             token = unescape(token)
#             if token == 'DCNL':
#                 new_token = '\n'
#                 space_after = False
#             elif token == 'DCSP':
#                 new_token = '\t'
#                 space_after = False
#             elif _replace_unk and token == '<unk>':
#                 new_token = _unk_replace_s
#                 space_after = True
#             elif _fix_python_orig_bug and token == '_':
#                 # If token is '_', concat two tokens (remove spaces).
#                 new_token = '_'
#                 if normalized_tokens and normalized_tokens[-1] == ' ':
#                     # Do not concat 'def _', because 'def _private_name' is a common pattern.
#                     # Do concat 'class _', because 'class _private_name' is uncommon, and '__class__' is common.
#                     if len(normalized_tokens) <= 1 or normalized_tokens[-2] != 'def':
#                         normalized_tokens.pop()
#                 space_after = False
#             else:
#                 new_token = token
#                 if _fix_python_orig_bug:
#                     # Concat '= =' to '=='.
#                     space_after = new_token.isidentifier()
#                 else:
#                     space_after = True
#             normalized_tokens.append(new_token)
#             if space_after:
#                 normalized_tokens.append(' ')
#         try:
#             ast_parse(''.join(normalized_tokens))
#         except SyntaxError:
#             return 0
#         else:
#             return 1
#
#     def pov(self):
#         """Percent of valid code."""
#         cum_pov = 0.0
#         for sys in self.sys:
#             cum_pov += self._parse_method(sys)
#         pov = cum_pov / len(self.sys) * 100
#         return pov
#
#     def result_string(self, order=4):
#         return 'BLEU4 = {:.2f}, PoV = {:.2f}'.format(self.score(order), self.pov())


class Scorer(object):
    def __init__(self, pad, eos, unk):
        self.stat = BleuStat()
        self.pad = pad
        self.eos = eos
        self.unk = unk
        self.reset()

    def reset(self, one_init=False):
        if one_init:
            C.bleu_one_init(ctypes.byref(self.stat))
        else:
            C.bleu_zero_init(ctypes.byref(self.stat))

    def add(self, ref, pred):
        if not isinstance(ref, torch.IntTensor):
            raise TypeError('ref must be a torch.IntTensor (got {})'
                            .format(type(ref)))
        if not isinstance(pred, torch.IntTensor):
            raise TypeError('pred must be a torch.IntTensor(got {})'
                            .format(type(pred)))

        # don't match unknown words
        rref = ref.clone()
        assert not rref.lt(0).any()
        rref[rref.eq(self.unk)] = -999

        rref = rref.contiguous().view(-1)
        pred = pred.contiguous().view(-1)

        C.bleu_add(
            ctypes.byref(self.stat),
            ctypes.c_size_t(rref.size(0)),
            ctypes.c_void_p(rref.data_ptr()),
            ctypes.c_size_t(pred.size(0)),
            ctypes.c_void_p(pred.data_ptr()),
            ctypes.c_int(self.pad),
            ctypes.c_int(self.eos))

    def score(self, order=4):
        psum = sum(math.log(p) if p > 0 else float('-Inf')
                   for p in self.precision()[:order])
        return self.brevity() * math.exp(psum / order) * 100

    def precision(self):
        def ratio(a, b):
            return a / b if b > 0 else 0

        return [
            ratio(self.stat.match1, self.stat.count1),
            ratio(self.stat.match2, self.stat.count2),
            ratio(self.stat.match3, self.stat.count3),
            ratio(self.stat.match4, self.stat.count4),
        ]

    def brevity(self):
        r = self.stat.reflen / self.stat.predlen
        return min(1, math.exp(1 - r))

    def result_string(self, order=4):
        assert order <= 4, "BLEU scores for order > 4 aren't supported"
        fmt = 'BLEU{} = {:2.2f}, {:2.1f}'
        for _ in range(1, order):
            fmt += '/{:2.1f}'
        fmt += ' (BP={:.3f}, ratio={:.3f}, syslen={}, reflen={})'
        bleup = [p * 100 for p in self.precision()[:order]]
        return fmt.format(order, self.score(order=order), *bleup,
                          self.brevity(), self.stat.predlen/self.stat.reflen,
                          self.stat.predlen, self.stat.reflen)
