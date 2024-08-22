#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Caching utils."""

from collections import OrderedDict
from types import MappingProxyType
from typing import Callable, Dict, Any

# Type aliases.
CacheType = 'Union[Cache, dict]'

_sentinel = object()
_ALL_CACHES: 'Dict[str, CacheType]' = {}


class Cache:
    def __init__(self, get_fn: Callable[[Any], Any], max_size: int, description: str):
        """ABC for caches.

        Args:
            get_fn:
            max_size: Max cache size, None means no limit, <= 0 means no cache.
            description:
        """
        self.get_fn = get_fn
        self.max_size = max_size
        self.description = description

    def __bool__(self):
        raise NotImplementedError()

    def set_max_size(self, max_size):
        self.max_size = max_size

    @property
    def data(self):
        raise NotImplementedError()


class SimpleCache(Cache):
    """Simple cache. Will call `popitem()` method when full."""
    def __init__(self, get_fn: Callable[[Any], Any] = None, max_size: int = None, description: str = ''):
        super().__init__(get_fn, max_size, description)
        self._data = {}

    def __bool__(self):
        return bool(self._data)

    def _internal_get(self, key, get_fn):
        if self.max_size is not None and self.max_size <= 0:
            return get_fn(key)
        value = self._data.get(key, _sentinel)
        if value is _sentinel:
            value = get_fn(key)
            self.__setitem__(key, value)
        return value

    def __getitem__(self, key):
        return self._internal_get(key, self.get_fn)

    def __setitem__(self, key, value):
        """Add item that not in current dict."""
        self._data[key] = value
        if self.max_size is not None and len(self._data) > self.max_size:
            self._data.popitem()

    def clear(self):
        self._data.clear()

    def get_by_fn(self, key, get_fn=None):
        """Get cache[key], call get_fn if not exists."""
        if get_fn is None:
            get_fn = self.get_fn
        return self._internal_get(key, get_fn)

    @property
    def data(self):
        return MappingProxyType(self._data)


class LRUCache(Cache):
    """LRU cache."""

    def __init__(self, get_fn: Callable[[Any], Any] = None, max_size: int = None, description: str = ''):
        super().__init__(get_fn, max_size, description)
        self._data = OrderedDict()

    def __bool__(self):
        return bool(self._data)

    def _internal_get(self, key, get_fn):
        if self.max_size is not None and self.max_size <= 0:
            return get_fn(key)
        value = self._data.get(key, _sentinel)
        if value is _sentinel:
            value = get_fn(key)
            self.__setitem__(key, value)
        else:
            # Refresh the newest access.
            self._data.move_to_end(key)
        return value

    def __getitem__(self, key):
        return self._internal_get(key, self.get_fn)

    def __setitem__(self, key, value):
        """Add item that not in current dict."""
        self._data[key] = value
        self._data.move_to_end(key)
        if self.max_size is not None and len(self._data) > self.max_size:
            self._data.popitem(last=False)

    def clear(self):
        self._data.clear()

    def get_by_fn(self, key, get_fn=None):
        """Get cache[key], call get_fn if not exists."""
        if get_fn is None:
            get_fn = self.get_fn
        return self._internal_get(key, get_fn)

    @property
    def data(self):
        return MappingProxyType(self._data)


def has_cache(category):
    return category in _ALL_CACHES


def add_cache(category, cache: CacheType):
    if category in _ALL_CACHES:
        raise KeyError(f'{category} already cached')
    _ALL_CACHES[category] = cache


def get_cache(category, default=None) -> CacheType:
    return _ALL_CACHES.get(category, default)
