#   Copyright 2017-2019 typed_python Authors
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

from typed_python import ListOf, Class, Member
from typed_python._types import touchCompiledSpecializations
from typed_python import Entrypoint
from typed_python.compiler.runtime import Runtime
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
        compiledAdd = Entrypoint(add)

        self.assertEqual(type(compiledAdd(IntList([1, 2, 3]), 1)), IntList)
        self.assertEqual(type(compiledAdd(FloatList([1, 2, 3]), 1)), FloatList)

        self.assertEqual(compiledAdd(IntList([1, 2, 3]), 1), add(IntList([1, 2, 3]), 1))
        self.assertEqual(compiledAdd(FloatList([1, 2, 3]), 1), add(FloatList([1, 2, 3]), 1))

    def test_specialized_entrypoint_on_staticmethod(self):
        compiled = Entrypoint(AClass.aMethod)
        self.assertEqual(compiled(10), 11)

    def test_specialized_entrypoint_doesnt_recompile(self):
        compiledAdd = Entrypoint(add)

        compileCount = Runtime.singleton().timesCompiled

        someInts = IntList(range(1000))
        someFloats = FloatList(range(1000))

        for _ in range(10):
            compiledAdd(someInts, 1)

        for _ in range(10):
            compiledAdd(someFloats, 1)

        self.assertEqual(Runtime.singleton().timesCompiled - compileCount, 2)

    def test_specialized_entrypoint_perf_difference(self):
        compiledAdd = Entrypoint(add)

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
        @Entrypoint
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

    def test_specialization_noniterable_after_iterable(self):

        class AClass():
            pass

        test_cases = [
            ListOf(int)(),
            AClass(),
        ]

        @Entrypoint
        def specialized_f(x):
            return True

        for x in test_cases:
            r = specialized_f(x)
            self.assertTrue(r)

    def test_can_use_entrypoints_from_functions(self):
        @Entrypoint
        def f(x):
            if x <= 0:
                return 0
            return g(x-1) + 1

        @Entrypoint
        def g(x):
            if x <= 0:
                return 0
            return g(x-1) + 1

        self.assertEqual(f(100), 100)

    def test_can_use_entrypoints_on_class_methods(self):
        class ARandomPythonClass:
            def __init__(self, x=10):
                self.x = x

            @Entrypoint
            def f(self, x: int):
                return self.x + x

            @Entrypoint
            @staticmethod
            def g(x: int, y: int):
                return x + y

            @staticmethod
            @Entrypoint
            def g2(x: int, y: int):
                return x + y

        self.assertEqual(ARandomPythonClass(23).f(10), 33)

    def test_can_use_entrypoints_on_typed_class_methods(self):
        class AClass(Class):
            x = Member(int)

            @Entrypoint
            def f(self, x: int):
                res = 0
                for i in range(x):
                    res += self.x
                return res

            @staticmethod
            @Entrypoint
            def g(x: int, y: float):
                return x + y

            @Entrypoint
            @staticmethod
            def g2(x: int, y: float):
                return x + y

        self.assertEqual(AClass(x=23).f(10), 230)
        t0 = time.time()
        AClass(x=23).f(100000000)
        self.assertLess(time.time() - t0, 2.0)

        self.assertEqual(AClass(x=23).g(10, 20.5), 30.5)
        self.assertEqual(AClass(x=23).g2(10, 20.5), 30.5)
