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

import unittest
import time

from flaky import flaky
from typed_python import Tuple, NamedTuple, Entrypoint

from typed_python import reduce


class TestReduce(unittest.TestCase):
    def test_reduce(self):
        def f(x, y=0.0):
            return x + y

        @Entrypoint
        def compiledReduce(f, tup):
            return reduce(f, tup)

        aTup = Tuple(int, int, float)((1, 2, 3.0))

        self.assertEqual(reduce(f, aTup), 6.0)
        self.assertEqual(compiledReduce(f, aTup), 6.0)

        aNamedTup = NamedTuple(x=int, y=float)((1, 2.5))
        self.assertEqual(reduce(f, aNamedTup), 3.5)
        self.assertEqual(compiledReduce(f, aNamedTup), 3.5)

    @flaky(max_runs=3, min_passes=1)
    def test_reduce_perf(self):
        def doit(tup, times):
            res = 0.0
            for i in range(times):
                res += reduce(lambda x, y=0.0: x + y, tup)
            return res

        compiledDoit = Entrypoint(doit)

        aTup = Tuple(int, int, float)((1, 2, 3.0))

        compiledDoit(aTup, 1)

        t0 = time.time()
        compiledDoit(aTup, 100000)
        t1 = time.time()
        doit(aTup, 100000)
        t2 = time.time()

        speedup = (t2 - t1) / (t1 - t0)

        print(f"tp is {speedup} times faster.")

        # I get about 2000, mostly because the python implementation,
        # which has to recreate the type object each time, is very slow.
        self.assertGreater(speedup, 100)
