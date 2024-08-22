"""Decompose RoBERTa pretrained model into low-rank matrices."""

import argparse
import collections
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Parameter

__author__ = 'v-yaf'

DEBUG_SVD = False
FIRST_FLAG = True
ONLY_FIRST = False


def _debug_svd(mat, L, R):
    global FIRST_FLAG

    if not DEBUG_SVD:
        return
    if ONLY_FIRST and not FIRST_FLAG:
        return
    FIRST_FLAG = False

    mat_32 = mat.to(torch.float32)
    L_32 = L.to(torch.float32)
    R_32 = R.to(torch.float32)

    mat_32_2 = mat_32 ** 2
    LR_32 = torch.mm(L_32, R_32)
    LR_32_2 = LR_32 ** 2
    delta_2 = (mat_32 - LR_32) ** 2
    print('Means: A^2 = {:.6f}, (LR)^2 = {:.6f}, L2-norm = {:.6f}'.format(
        mat_32_2.mean().item(), LR_32_2.mean().item(), delta_2.mean().item()))


def decompose_matrix(mat, k, dtype=None):
    assert mat.ndimension() == 2, 'input matrix must be 2-dim'

    if mat.dtype is torch.float16:
        mat_32 = mat.to(torch.float32)
    else:
        mat_32 = mat

    m, n = mat_32.size()
    U, s, V = torch.svd(mat_32)

    if DEBUG_SVD and not (ONLY_FIRST and not FIRST_FLAG):
        print('Singular values: max = {:.6f}, min = {:.6f}'.format(s.max().item(), s.min().item()))

    Uk = U[:, :k]
    sk = s[:k]
    Vk = V[:, :k]
    Rk = torch.mm(torch.diag(sk), Vk.transpose(0, 1))

    if dtype is torch.float32:
        _debug_svd(mat_32, Uk, Rk)
        return Uk, Rk

    if dtype is torch.float16 or mat.dtype is torch.float16:
        Uk = Uk.to(torch.float16)
        Rk = Rk.to(torch.float16)
    
    _debug_svd(mat_32, Uk, Rk)
    
    return Uk, Rk


def _assign_new_value(new_model, new_key, new_value):
    new_model[new_key] = new_value
    if not (DEBUG_SVD and ONLY_FIRST and not FIRST_FLAG):
        print('->', new_key, new_value.dtype, new_value.shape)


def decompose_roberta(args, model):
    """Decompose the RoBERTa model."""
    new_model = collections.OrderedDict()
    for key, value in model.items():
        # PFFN.
        if args.k_ffn > 0:
            if key.endswith('fc1.weight') or key.endswith('fc2.weight'):
                weight_l, weight_r = decompose_matrix(value, args.k_ffn, args.dtype)
                _assign_new_value(new_model, '{}_l'.format(key), weight_l)
                _assign_new_value(new_model, '{}_r'.format(key), weight_r)
                continue

        # Input embedding.
        if args.k_emb > 0:
            if 'embed_tokens' in key:
                weight_l, weight_r = decompose_matrix(value, args.k_emb, args.dtype)
                _assign_new_value(new_model, '{}_l'.format(key), weight_l)
                _assign_new_value(new_model, '{}_r'.format(key), weight_r)
                continue
        
        # LM head.
        if args.k_emb > 0:
            if 'lm_head.weight' in key:
                weight_l, weight_r = decompose_matrix(value, args.k_emb, args.dtype)
                _assign_new_value(new_model, '{}_l'.format(key), weight_l)
                _assign_new_value(new_model, '{}_r'.format(key), weight_r)
                continue

        # Self Attention: in projection.

        # Default: copy.
        if not DEBUG_SVD:
            print('== {} [{} to {}] {}'.format(key, value.dtype, (value.dtype if args.dtype is None else args.dtype), value.shape))
        if args.dtype is None:
            new_model[key] = value
        else:
            new_model[key] = value.to(args.dtype)
    
    print('#parameters: {} -> {}'.format(
        sum(p.numel() for p in model.values()),
        sum(p.numel() for p in new_model.values()),
    ))
    
    return new_model


def main(args=None):
    global DEBUG_SVD, ONLY_FIRST, FIRST_FLAG

    parser = argparse.ArgumentParser(description='Decompose RoBERTa model.')
    parser.add_argument('path', nargs='?', default=r'\\msralpa\Users\v-yaf\public\DataTransfer\GitProjects\LightBERT\checkpoints\roberta.large\model.pt',
        help='Path of the checkpoint, default is %(default)r')
    parser.add_argument('-e', '--path-ext', default='', help='Extra postfix of dumped checkpoint filename, default is %(default)r')
    parser.add_argument('--k-emb', type=int, default=128, help='K of embedding, (<= 0 means disabled), default is %(default)r')
    parser.add_argument('--k-ffn', type=int, default=128, help='K of PFFN, (<= 0 means disabled), default is %(default)r')
    parser.add_argument('--dtype', default=None, help='Target dtype (fp16, fp32 or keep), default is %(default)r')
    parser.add_argument('-D', '--debug', action='store_true', default=False, help='Debug mode, default is %(default)r')
    parser.add_argument('--D1', '--only-first', dest='only_first', action='store_true', default=False, help='Only print first item in debug mode, default is %(default)r')
    args = parser.parse_args(args=args)

    # Process args.
    assert args.dtype in (None, 'fp16', 'fp32'), '--dtype must be "fp16" or "fp32"'
    if args.dtype == 'fp16':
        args.dtype = torch.float16
    elif args.dtype =='fp32':
        args.dtype = torch.float32
    
    FIRST_FLAG = True
    ONLY_FIRST = args.only_first
    DEBUG_SVD = args.debug
    if DEBUG_SVD:
        print('DEBUG'.center(70, '='))

    print(args)

    checkpoint = torch.load(args.path, map_location='cpu')
    model = checkpoint['model']
    new_model = decompose_roberta(args, model)

    # Fix other data.
    pt_args = checkpoint['args']
    pt_args.arch = 'd_{}'.format(pt_args.arch)
    pt_args.k_emb = args.k_emb
    pt_args.k_ffn = args.k_ffn
    checkpoint['model'] = new_model
    
    # Dump new checkpoint.
    if not args.debug:
        _base, _ext = os.path.splitext(args.path)
        new_pt_name = _base + '_' + args.path_ext + _ext
        torch.save(checkpoint, new_pt_name)
        print('Save decomposed checkpoint to {}.'.format(new_pt_name))


if __name__ == "__main__":
    # main()
    # main('-e 1024 --k-emb 1024 --k-ffn 1024'.split())
    # main('-e 1024-fp32 --k-emb 1024 --k-ffn 1024 --dtype fp32'.split())
    # main('-e 128-fp32 --k-emb 128 --k-ffn 128 --dtype fp32'.split())
    # main('-e 128-f --k-emb 0 --k-ffn 128'.split())
    # main('-e 256-f --k-emb 0 --k-ffn 256'.split())
    # main('-e 512-f --k-emb 0 --k-ffn 512'.split())
    # main('-e 768-f --k-emb 0 --k-ffn 768'.split())
    # main('-e 1000-f --k-emb 0 --k-ffn 1000'.split())
    # main('-e 1024-f --k-emb 0 --k-ffn 1024'.split())
    # main('-e 0 --k-emb 0 --k-ffn 0'.split())

    # Debug jobs.
    # main('-e 1024-f --k-emb 0 --k-ffn 1024 -D --D1'.split())
    # main('-e 1000-f --k-emb 0 --k-ffn 1000 -D --D1'.split())
    # main('-e 768-f --k-emb 0 --k-ffn 768 -D --D1'.split())
    # main('-e 512-f --k-emb 0 --k-ffn 512 -D --D1'.split())
    # main('-e 256-f --k-emb 0 --k-ffn 256 -D --D1'.split())
    # main('-e 128-f --k-emb 0 --k-ffn 128 -D --D1'.split())
    pass
