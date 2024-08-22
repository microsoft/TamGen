#! /usr/bin/python
# -*- coding: utf-8 -*-

"""My debug utils."""

import sys
import os
import inspect
from collections import defaultdict
from typing import List

import numpy as np
import torch


def _bool(val: str) -> bool:
    val = val.lower()
    if val in {None, '', '0', 'false', 'f', 'no', 'n'}:
        return False
    if val in {'1', 'true', 't', 'yes', 'y'}:
        return True
    raise ValueError(f'Cannot convert value {val} to bool')


def debug_enabled(enabled=True) -> bool:
    return enabled and _bool(os.environ.get('FYDEBUG', ''))


_LOG_TENSORS_ACCESS_CNT = defaultdict(int)


def get_caller_frame(caller_depth=2):
    if hasattr(sys, '_getframe'):
        return sys._getframe(caller_depth)
    else:
        raise RuntimeError('cannot get caller frame')


def get_caller_env(frame):
    return frame.f_globals, frame.f_locals


def get_caller_info(frame_info: 'inspect.Traceback'):
    return frame_info.filename, frame_info.lineno


def log_tensor_attr(env, tensor_name: str, attr: str, skip_not_exist: bool = True):
    try:
        attr_value = eval(f'{tensor_name}.{attr}', *env)
    except AttributeError:
        if skip_not_exist:
            return
        else:
            raise

    if isinstance(attr_value, torch.Size):
        attr_value = list(attr_value)
    _access_cnt = _LOG_TENSORS_ACCESS_CNT[tensor_name]
    print(f'[{_access_cnt}] {tensor_name}.{attr} = {attr_value}', file=sys.stderr)


def log_tensor(tensor_names: List[str], extra_attr_list=None, log_range=None):
    if not debug_enabled():
        return

    real_tensor_names = [
        tensor_name for tensor_name in tensor_names
        if log_range is None or _LOG_TENSORS_ACCESS_CNT[tensor_name] in log_range
    ]

    # Mark an access
    for tensor_name in tensor_names:
        _LOG_TENSORS_ACCESS_CNT[tensor_name] += 1

    if not real_tensor_names:
        return

    frame = get_caller_frame()
    frame_info = inspect.getframeinfo(frame)
    env = get_caller_env(frame)
    caller_fn, caller_lineno = get_caller_info(frame_info)

    print(f'======= [DEBUG] =======', file=sys.stderr)
    print(f'Called at {caller_fn}:{caller_lineno}', file=sys.stderr)

    attr_list = ['size()', 'dtype', 'device']
    np_attr_list = ['shape', 'dtype']

    if extra_attr_list is None:
        extra_attr_list = []

    for tensor_name in real_tensor_names:
        tensor = eval(tensor_name, *env)
        if torch.is_tensor(tensor):
            this_attr_list = attr_list + extra_attr_list
        elif isinstance(tensor, np.ndarray):
            this_attr_list = np_attr_list + extra_attr_list
        else:
            # Not a tensor
            print(f'--- {type(tensor).__name__} {tensor_name}={tensor}', file=sys.stderr)
            continue

        print(f'--- {type(tensor).__name__} {tensor_name} ---', file=sys.stderr)
        for attr in this_attr_list:
            log_tensor_attr(env, tensor_name, attr)

    print('===== [END DEBUG] =====', file=sys.stderr)
    sys.stderr.flush()
