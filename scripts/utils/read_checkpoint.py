#! /usr/bin/python
# -*- coding: utf-8 -*-

import argparse
from collections import defaultdict
from pprint import pprint
from typing import Dict, Union, Tuple

import torch


KeyWords = Tuple[Union[str, int]]


def parse_args():
    parser = argparse.ArgumentParser(description='Checkpoint reader.')
    parser.add_argument('path', help='Checkpoint path')
    parser.add_argument('-k', '--key', action='store_true', default=False, help='Print all keys')

    args = parser.parse_args()
    return args


def _key2words(key: str) -> KeyWords:
    words = key.split('.')
    result = []
    for word in words:
        try:
            int_word = int(word)
            result.append(int_word)
        except ValueError:
            result.append(word)
    return tuple(result)


def _word_same(words1: KeyWords, words2: KeyWords):
    if len(words1) != len(words2):
        return False
    for w1, w2 in zip(words1, words2):
        if w1 != w2 and (isinstance(w1, str) or isinstance(w2, str)):
            return False
    return True


def _words2str(words: KeyWords, max_len: int = None, top_n_empty: int = 0):
    def _gen(_words):
        for i, w in enumerate(words):
            ws = str(w)
            if i < top_n_empty:
                yield ''.ljust(len(ws))
            else:
                yield ws
    key = '.'.join(_gen(words))
    if max_len:
        key = key.ljust(max_len)
    return key


def _common_prefix_idx(s1_words: KeyWords, s2_words: KeyWords):
    idx = 0
    while idx < len(s1_words) and idx < len(s2_words) and s1_words[idx] == s2_words[idx]:
        idx += 1
    return idx


def _print_params(model: Dict[str, torch.Tensor]):
    key_max_len = max(len(key) for key in model)

    # Merge duplicate keys.
    duplicate_layer_words = defaultdict(list)
    for key in model.keys():
        key_words = _key2words(key)
        for cur_words in duplicate_layer_words:
            if _word_same(cur_words, key_words):
                duplicate_layer_words[cur_words].append(key_words)
                break
        else:
            duplicate_layer_words[key_words] = []

    print('===== Model params =====')
    prev_key_words = []
    for key_words in duplicate_layer_words:
        value = model[_words2str(key_words)]
        p_idx = _common_prefix_idx(key_words, prev_key_words)
        abbr_key = _words2str(key_words, max_len=key_max_len, top_n_empty=p_idx)
        print(f'{abbr_key}: {str(value.dtype):<13}, {list(value.shape)}')
        prev_key_words = key_words
    print('===== Duplicates =====')
    for words, dup in duplicate_layer_words.items():
        if not dup:
            continue
        print(_words2str(words), '{')
        for dup_words in dup:
            print(_words2str(dup_words))
        print('}')


def main():
    args = parse_args()

    checkpoint = torch.load(args.path, map_location='cpu')
    model = checkpoint['model']     # type: dict

    print('Checkpoint path:', args.path)
    print('===== Checkpoint arguments =====')
    print(checkpoint['args'])
    print('Keys:', *checkpoint.keys())
    print('===== Extra state =====')
    pprint(checkpoint['extra_state'])

    if args.key:
        _print_params(model)


if __name__ == '__main__':
    main()
