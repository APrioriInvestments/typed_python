#   Coyright 2017-2019 Nativepython Authors
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

from typed_python import ListOf
from nativepython import SpecializedEntrypoint
from nativepython.runtime import Runtime
import time
import unittest


def add(aList, toAdd):
    res = type(aList)()

    for i in range(len(aList)):
        res.append(aList[i] + toAdd)

    return res


IntList = ListOf(int)
FloatList = ListOf(float)


class TestCompileSpecializedEntrypoints(unittest.TestCase):
    def test_specialized_entrypoint(self):
        compiledAdd = SpecializedEntrypoint(add)

        self.assertEqual(type(compiledAdd(IntList([1, 2, 3]), 1)), IntList)
        self.assertEqual(type(compiledAdd(FloatList([1, 2, 3]), 1)), FloatList)

        self.assertEqual(compiledAdd(IntList([1, 2, 3]), 1), add(IntList([1, 2, 3]), 1))
        self.assertEqual(compiledAdd(FloatList([1, 2, 3]), 1), add(FloatList([1, 2, 3]), 1))

    def test_specialized_entrypoint_doesnt_recompile(self):
        compiledAdd = SpecializedEntrypoint(add)

        compileCount = Runtime.singleton().timesCompiled

        someInts = IntList(range(1000))
        someFloats = FloatList(range(1000))

        for _ in range(10):
            compiledAdd(someInts, 1)

        for _ in range(10):
            compiledAdd(someFloats, 1)

        self.assertEqual(Runtime.singleton().timesCompiled - compileCount, 2)

    def test_specialized_entrypoint_perf_difference(self):
        compiledAdd = SpecializedEntrypoint(add)

        for T in [IntList, FloatList]:
            aList = T(range(1000000))

            t0 = time.time()
            add(aList, 1)
            t1 = time.time()
            compiledAdd(aList, 1)
            t2 = time.time()

            speedup = (t1 - t0)/(t2 - t1)
            self.assertGreater(speedup, 10)

            print("speedup for ", T, " is ", speedup)  # I get about 70x
