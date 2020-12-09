#   Copyright 2017-2020 typed_python Authors
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import numpy
import time
import typed_python.lib.sorting as sorting
import unittest

from flaky import flaky
from typed_python import ListOf, Tuple


class TestSorting(unittest.TestCase):
    @flaky(max_runs=3, min_passes=1)
    def test_perf_not_quadratic(self):
        length = 100000

        x = ListOf(int)(i for i in range(length))
        y = ListOf(int)(length - 1 - i for i in range(length))

        z = ListOf(int)(numpy.random.choice(length, length, replace=False))

        allEqual = ListOf(int)(0 for i in range(length))

        # prime the compiler
        sorting.sorted(x)

        t0 = time.time()
        sorting.sort(x)
        t1 = time.time()
        sorting.sort(y)
        t2 = time.time()
        sorting.sort(z)
        t3 = time.time()
        sorting.sort(allEqual)
        t4 = time.time()

        self.assertEqual(x, y)
        self.assertEqual(y, z)

        inOrder = t1 - t0
        inReverseOrder = t2 - t1
        inRandomOrder = t3 - t2
        allEqualTiming = t4 - t3

        self.assertLess(inOrder, inRandomOrder * 10)
        self.assertLess(inReverseOrder, inRandomOrder * 10)
        self.assertLess(allEqualTiming, inRandomOrder * 10)

    def test_quicksort_float_correct(self):
        x = ListOf(float)(numpy.random.uniform(size=1000))

        self.assertEqual(ListOf(float)(sorted(x)), sorting.sorted(x))

    def test_quicksort_int_correct(self):
        x = ListOf(int)(numpy.random.choice(1000, 1000, replace=False))

        self.assertEqual(ListOf(int)(sorted(x)), sorting.sorted(x))

    def test_quicksort_int_correct_with_repeated_values(self):
        x = ListOf(int)(numpy.random.choice(100, size=1000, replace=True))

        self.assertEqual(ListOf(int)(sorted(x)), sorting.sorted(x))

    @flaky(max_runs=3, min_passes=1)
    def test_sort_perf_simple(self):
        x = ListOf(float)(numpy.random.uniform(size=1000000))

        sorting.sorted(x[:10])

        t0 = time.time()
        sorting.sorted(x)
        t1 = time.time()
        sorted(x)
        t2 = time.time()

        speedup = (t2 - t1) / (t1 - t0)

        # I get about 3
        self.assertGreater(speedup, 1.5)

    @flaky(max_runs=3, min_passes=1)
    def test_sort_perf_tuples(self):
        x = ListOf(Tuple(float, float))()

        while len(x) < 100000:
            x.append((numpy.random.uniform(), numpy.random.uniform()))

        sorting.sorted(x[:10])

        t0 = time.time()
        sorting.sorted(x)
        t1 = time.time()
        sorted(x)
        t2 = time.time()

        speedup = (t2 - t1) / (t1 - t0)

        # I get about 9
        self.assertGreater(speedup, 2)

    def test_sort_with_key(self):
        x = ListOf(int)(range(100))

        self.assertEqual(
            sorting.sorted(x, key=lambda x: Tuple(int, int)((x % 10, x))),
            sorted(x, key=lambda x: (x % 10, x))
        )
