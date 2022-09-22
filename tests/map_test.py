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
from typed_python import Tuple, NamedTuple, Entrypoint, makeNamedTuple, ListOf, Function, OneOf
from typed_python import map


class TestMap(unittest.TestCase):
    def test_map(self):
        def f(x):
            return x + 1

        def toStr(x):
            return str(x)

        @Entrypoint
        def compiledMap(f, tup):
            return map(f, tup)

        aTup = Tuple(int, int, float)((1, 2, 3.0))

        self.assertEqual(type(map(f, aTup)), Tuple(int, int, float))
        self.assertEqual(type(map(toStr, aTup)), Tuple(str, str, str))
        self.assertEqual(type(compiledMap(f, aTup)), Tuple(int, int, float))
        self.assertEqual(type(compiledMap(toStr, aTup)), Tuple(str, str, str))

        aNamedTup = NamedTuple(x=int, y=float)((1, 2.5))

        self.assertEqual(type(map(f, aNamedTup)), NamedTuple(x=int, y=float))
        self.assertEqual(type(compiledMap(f, aNamedTup)), NamedTuple(x=int, y=float))

    @flaky(max_runs=3, min_passes=1)
    def test_map_perf(self):
        @Function
        def addOne(x):
            return x + 1

        def doit(tup, times):
            res = 0.0
            for i in range(times):
                res += map(addOne, tup)[2]
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

    def test_transpose_lists(self):
        @Entrypoint
        def transposeLists(tupOfLists):
            outList = ListOf(type(map(lambda l: type(l).ElementType(), tupOfLists)))()

            for i in range(len(tupOfLists[0])):
                outList.append(map(lambda l: l[i], tupOfLists))

            return outList

        listOfInts = ListOf(int)(range(1000000))
        listOfFloats = ListOf(float)(range(1000000))

        res = transposeLists(makeNamedTuple(x=listOfInts, y=listOfFloats))

        t0 = time.time()
        res = transposeLists(makeNamedTuple(x=listOfInts, y=listOfFloats))
        print(time.time() - t0, " to transpose.")

        self.assertEqual(res[0], makeNamedTuple(x=0, y=0.0))

    def test_map_with_multiple_outputs(self):
        @Entrypoint
        def f(x) -> OneOf(int, float):
            if x % 2:
                return int(x)
            return float(x)

        aTup = Tuple(int, int, float)((1, 2, 3.0))

        self.assertEqual(type(map(f, aTup)), Tuple(OneOf(int, float), OneOf(int, float), OneOf(int, float)))
