#! /usr/bin/python
# -*- coding: utf-8 -*-

import unittest

from fairseq.molecule_utils.database import caching_utils


class TestCachingUtils(unittest.TestCase):
    def testLRUCache(self):
        cache = caching_utils.LRUCache(max_size=2)

        def get_func(a):
            return a + 100

        self.assertEqual(bool(cache), False)

        cache.get_by_fn(1, get_fn=get_func)
        self.assertEqual(list(cache.data.items()), [(1, 101)])

        cache.get_by_fn(2, get_fn=get_func)
        self.assertEqual(list(cache.data.items()), [(1, 101), (2, 102)])

        cache.get_by_fn(1, get_fn=get_func)
        self.assertEqual(list(cache.data.items()), [(2, 102), (1, 101)])

        cache.get_by_fn(3, get_fn=get_func)
        self.assertEqual(list(cache.data.items()), [(1, 101), (3, 103)])


if __name__ == '__main__':
    unittest.main()
