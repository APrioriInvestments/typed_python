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
from typed_python._types import touchCompiledSpecializations
from nativepython import SpecializedEntrypoint
from nativepython.runtime import Runtime
import traceback
import threading
import time
import unittest


def add(aList, toAdd):
    res = type(aList)()

    for i in range(len(aList)):
        res.append(aList[i] + toAdd)

    return res


class AClass:
    @staticmethod
    def aMethod(x):
        return x + 1


IntList = ListOf(int)
FloatList = ListOf(float)


class TestCompileSpecializedEntrypoints(unittest.TestCase):
    def test_specialized_entrypoint(self):
        compiledAdd = SpecializedEntrypoint(add)

        self.assertEqual(type(compiledAdd(IntList([1, 2, 3]), 1)), IntList)
        self.assertEqual(type(compiledAdd(FloatList([1, 2, 3]), 1)), FloatList)

        self.assertEqual(compiledAdd(IntList([1, 2, 3]), 1), add(IntList([1, 2, 3]), 1))
        self.assertEqual(compiledAdd(FloatList([1, 2, 3]), 1), add(FloatList([1, 2, 3]), 1))

    def test_specialized_entrypoint_on_staticmethod(self):
        compiled = SpecializedEntrypoint(AClass.aMethod)
        self.assertEqual(compiled(10), 11)

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

    def test_many_threads_compiling_same_specialization(self):
        @SpecializedEntrypoint
        def sumFun(a, b):
            """Create a new instance of type 'A' and filter where the flags are True in 'flags'"""
            res = 0
            while a < b:
                res = res + a
                a = a + 1

            return res

        failed = []
        done = [False]

        def threadloop():
            try:
                while not done[0]:
                    sumFun(0.0, 10**4)
            except Exception:
                traceback.print_exc()
                failed.append(True)

        threads = [threading.Thread(target=threadloop) for x in range(10)]
        for t in threads:
            t.daemon = True
            t.start()

        t0 = time.time()
        while time.time() - t0 < 10.0:
            touchCompiledSpecializations(sumFun.__typed_python_function__.overloads[0].functionTypeObject, 0)

        done[0] = True

        for t in threads:
            t.join()

        assert not failed
