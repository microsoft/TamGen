#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Java utils.

Also see ``run/transformer-mb-supp/code-improver.py``.
"""

from javalang.parse import parse_member_signature
from javalang.parser import JavaParserBaseException

_UNK_TOKEN = '<unk>'
_SPECIAL_TOKENS = {
    _UNK_TOKEN: _UNK_TOKEN,
    '<NUM>': '0',
    '<BOOL>': 'true',
    '<STR>': '""',
}


def split_string(string: str, method='original'):
    """Split reference or predicted strings, and optionally merge '@'-started code fragments.

    >>> split_string('m @_list @model', method='original')
    ['m', '@_list', '@model']
    >>> split_string('m @_list @model', method='improved')
    ['m_listModel']
    """
    tokens = string.split()
    if method == 'original':
        return tokens
    elif method == 'improved':
        new_tokens = []
        for token in tokens:
            if token.startswith('@'):
                real_token = '{}{}'.format(token[1].upper(), token[2:])
                if not new_tokens:
                    new_tokens.append(real_token)
                else:
                    last_new_token = new_tokens[-1]
                    if last_new_token[0] == '<' and last_new_token[-1] == '>':
                        new_tokens.append(real_token)
                    else:
                        new_tokens[-1] = last_new_token + real_token
            else:
                new_tokens.append(token)
        # Unwrap special tokens.
        unwrap_tokens = []
        for token in new_tokens:
            if token[0] == '<' and token[-1] == '>':
                new_token = _SPECIAL_TOKENS.get(token, None)
                if new_token is None:
                    new_token = token[1:-1]
                unwrap_tokens.append(new_token)
            else:
                unwrap_tokens.append(token)
        return unwrap_tokens
    else:
        raise RuntimeError('Unknown split method {!r}'.format(method))


def parse_java(sys_tokens, pov_replace_unk=True, unk_replace_str='MASK'):
    if pov_replace_unk:
        normalized_tokens = []
        for token in sys_tokens:
            if token == _UNK_TOKEN:
                normalized_tokens.append(unk_replace_str)
            else:
                normalized_tokens.append(token)
    else:
        normalized_tokens = sys_tokens

    try:
        parse_member_signature(' '.join(normalized_tokens))
    except JavaParserBaseException:
        return 0
    else:
        return 1
