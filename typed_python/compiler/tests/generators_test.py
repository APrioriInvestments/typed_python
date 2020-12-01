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
import gc
import pytest
from flaky import flaky

from typed_python import Entrypoint, ListOf, TupleOf
from typed_python.test_util import currentMemUsageMb


def timeIt(f):
    t0 = time.time()
    f()
    return time.time() - t0


class TestGeneratorsAndComprehensions(unittest.TestCase):
    def test_list_comp(self):
        @Entrypoint
        def listComp(x):
            return [a + 1 for a in range(x)]

        lst = listComp(10)

        assert isinstance(lst, list)
        assert lst == [a + 1 for a in range(10)]

    def test_list_from_listcomp(self):
        @Entrypoint
        def listComp(x):
            return ListOf(int)([a + 1 for a in range(x)])

        lst = listComp(10)

        assert isinstance(lst, ListOf(int))
        assert lst == [a + 1 for a in range(10)]

    @flaky(max_runs=3, min_passes=1)
    def test_list_from_listcomp_perf(self):
        def sum(iterable):
            res = 0
            for s in iterable:
                res += s
            return res

        @Entrypoint
        def listCompSumConverted(x):
            return sum(ListOf(int)([a + 1 for a in range(x)]))

        @Entrypoint
        def listCompSumGenerator(x):
            return sum(ListOf(int)(a + 1 for a in range(x)))

        @Entrypoint
        def tupleCompSumConverted(x):
            return sum(TupleOf(int)([a + 1 for a in range(x)]))

        @Entrypoint
        def tupleCompSumGenerator(x):
            return sum(TupleOf(int)(a + 1 for a in range(x)))

        @Entrypoint
        def listCompSumMasquerade(x):
            return sum([a + 1 for a in range(x)])

        def listCompSumUntyped(x):
            return sum([a + 1 for a in range(x)])

        listCompSumConverted(1000)
        listCompSumGenerator(1000)
        tupleCompSumConverted(1000)
        tupleCompSumGenerator(1000)
        listCompSumMasquerade(1000)

        compiledTimes = [
            timeIt(lambda: listCompSumConverted(1000000)),
            timeIt(lambda: listCompSumGenerator(1000000)),
            timeIt(lambda: tupleCompSumConverted(1000000)),
            timeIt(lambda: tupleCompSumGenerator(1000000)),
            timeIt(lambda: listCompSumMasquerade(1000000)),
        ]
        untypedTime = timeIt(lambda: listCompSumUntyped(1000000))

        print(compiledTimes)

        avgCompiledTime = sum(compiledTimes) / len(compiledTimes)

        # they should be about the same
        for timeElapsed in compiledTimes:
            assert .75 <= timeElapsed / avgCompiledTime <= 1.25

        # but python is much slower. I get about 30 x.
        assert untypedTime / avgCompiledTime > 10

    @flaky(max_runs=3, min_passes=1)
    def test_untyped_tuple_from_listcomp_perf(self):
        def sum(iterable):
            res = 0
            for s in iterable:
                res += s
            return res

        @Entrypoint
        def tupleCompSumConverted(x):
            return sum(tuple([a + 1 for a in range(x)]))

        @Entrypoint
        def tupleCompSumConvertedGenerator(x):
            return sum(tuple(a + 1 for a in range(x)))

        @Entrypoint
        def listCompSumConvertedGenerator(x):
            return sum(list(a + 1 for a in range(x)))

        def listCompSumUntyped(x):
            return sum(tuple(a + 1 for a in range(x)))

        tupleCompSumConverted(1000)
        tupleCompSumConvertedGenerator(1000)
        listCompSumConvertedGenerator(1000)

        tupleCompiled = timeIt(lambda: tupleCompSumConverted(10000000))
        tupleCompiledGenerator = timeIt(lambda: tupleCompSumConvertedGenerator(10000000))
        listCompiledGenerator = timeIt(lambda: listCompSumConvertedGenerator(10000000))
        untypedTime = timeIt(lambda: listCompSumUntyped(10000000))

        print("tupleCompiled = ", tupleCompiled)
        print("tupleCompiledGenerator = ", tupleCompiledGenerator)
        print("listCompiledGenerator = ", listCompiledGenerator)
        print("untypedTime = ", untypedTime)

        # they should be about the same
        assert .75 <= tupleCompiled / tupleCompiledGenerator <= 1.25
        assert .75 <= listCompiledGenerator / tupleCompiledGenerator <= 1.25

        # but python is much slower. I get about 30 x.
        assert untypedTime / listCompiledGenerator > 10

    def executeInLoop(self, f, duration=.25, threshold=1.0):
        gc.collect()
        memUsage = currentMemUsageMb()

        t0 = time.time()

        count = 0
        while time.time() - t0 < duration:
            f()
            count += 1

        gc.collect()
        print("count=", count, "allocated=", currentMemUsageMb() - memUsage)
        self.assertLess(currentMemUsageMb() - memUsage, threshold)

    @pytest.mark.skipif('sys.platform=="darwin"')
    def test_listcomp_doesnt_leak(self):
        @Entrypoint
        def listComp(x):
            return [a + 1 for a in range(x)]

        @Entrypoint
        def sumListComp(x):
            l = listComp(x)
            res = 0
            for val in l:
                res += val
            return res

        # burn it in
        sumListComp(1000000)
        self.executeInLoop(lambda: sumListComp(1000000), duration=.1, threshold=20.0)
        self.executeInLoop(lambda: sumListComp(1000000), duration=.25, threshold=1.0)

    def test_call_generator(self):
        @Entrypoint
        def generateInts(ct):
            yield 1
            yield 2

        assert list(generateInts(100)) == [1, 2]

    def test_call_generator_with_branch(self):
        @Entrypoint
        def generateInts(ct):
            yield 1

            if ct > 0:
                yield 2
            else:
                yield 3

            yield 4

        assert list(generateInts(1)) == [1, 2, 4]
        assert list(generateInts(-1)) == [1, 3, 4]

    def test_call_generator_with_loop(self):
        @Entrypoint
        def generateInts(ct):
            x = 0
            while x < ct:
                yield x
                x = x + 1
            else:
                yield -1
            yield -2

        assert list(generateInts(10)) == list(range(10)) + [-1, -2]
        assert list(generateInts(0)) == list(range(0)) + [-1, -2]

    def test_call_generator_with_closure_var(self):
        xInClosure = 100

        @Entrypoint
        def generateInts(ct):
            yield 1
            yield xInClosure
            yield ct
            yield 2

        assert list(generateInts(10)) == [1, 100, 10, 2]

    def test_call_generator_with_closure_var_cant_assign(self):
        xInClosure = 100

        @Entrypoint
        def generateInts(ct):
            yield 1
            yield xInClosure
            xInClosure = xInClosure + 1  # noqa
            yield ct
            yield 2

        with self.assertRaises(UnboundLocalError):
            assert list(generateInts(10)) == [1, 100, 10, 2]
