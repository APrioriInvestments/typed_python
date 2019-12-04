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

from typed_python import ListOf, Class, Member, Final, TupleOf, DisableCompiledCode
from typed_python._types import touchCompiledSpecializations
from typed_python import Entrypoint, NotCompiled, Function
from typed_python.compiler.runtime import Runtime
from flaky import flaky
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
    def test_entrypoint_functions_work(self):
        @Entrypoint
        def f(x: TupleOf(int)):
            return x

        self.assertEqual(f.resultTypeFor((1, 2, 3)), TupleOf(int))

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

    def test_can_pass_types_as_values(self):
        @Entrypoint
        def append(T, x):
            res = ListOf(T)()
            res.append(x)
            return res

        self.assertEqual(append(int, 1.5), [1])

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
        class AClass(Class, Final):
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

    @flaky(max_runs=3, min_passes=1)
    def test_can_pass_functions_as_objects(self):
        def addsOne(x):
            return x + 1

        def doubles(x):
            return x * 2

        @Entrypoint
        def sumOver(count, f):
            res = 0.0
            for i in range(count):
                res += f(i)
            return res

        self.assertEqual(sumOver(10, addsOne), 55)
        self.assertEqual(sumOver(10, doubles), 90)

        # make sure it's fast
        count = 100000000

        t0 = time.time()
        sumOver(count, addsOne)
        elapsed = time.time() - t0
        self.assertLess(elapsed, .4)

        @Entrypoint
        def sumDoubles(count):
            res = 0.0
            for i in range(count):
                res += i * i
            return res

        sumDoubles(1)

        t0 = time.time()
        sumDoubles(count)
        elapsedNoFunc = time.time() - t0

        ratio = elapsed / elapsedNoFunc

        self.assertTrue(.7 <= ratio <= 1.3, ratio)

    def test_type_of_passed_function_object(self):
        @Entrypoint
        def typeOf(f):
            return type(f)

        def f(x):
            return x

        self.assertEqual(typeOf(f), Function(f))

    @flaky(max_runs=3, min_passes=1)
    def test_sequential_append_performance(self):
        @Entrypoint
        def cumSum1(x):
            out = type(x)()
            for i in x:
                out.append(i)
            return out

        @Entrypoint
        def cumSum2(x):
            out = type(x)()
            for i in x:
                out.append(i)
            return out

        i = ListOf(int)(range(1000000))

        cumSum1(i)
        cumSum2(i)

        t0 = time.time()
        cumSum1(i)
        t1 = time.time()
        cumSum2(i)
        t2 = time.time()

        ratio = (t2 - t1) / (t1 - t0)

        self.assertTrue(.8 <= ratio <= 1.2, ratio)

    def test_nocompile_works(self):
        thisWouldNotBeVisible = set()

        @NotCompiled
        def f(x: float) -> float:
            thisWouldNotBeVisible.add(x)
            return x

        @Entrypoint
        def sumOver(count, x):
            # we should know this is a float
            seed = f(x)

            for i in range(count):
                seed += i * 1.0

            return seed

        sumOver(100, 3.0)

        self.assertEqual(thisWouldNotBeVisible, set([3.0]))

        t0 = time.time()
        sumOver(100000000, 3.0)
        time.time() - t0

        self.assertLess(time.time() - t0, .4)

    def test_disable_compiled_code(self):
        @Entrypoint
        def sum(x: int):
            res = 0.0
            for i in range(x):
                res += float(i)
            return res

        # prime the function
        sum(10)

        t0 = time.time()
        sum(1000000)
        t1 = time.time()

        print(f"Took {t1 - t0} to compute 1mm sum.")

        self.assertFalse(DisableCompiledCode.isDisabled())

        with DisableCompiledCode():
            self.assertTrue(DisableCompiledCode.isDisabled())

            t2 = time.time()
            sum(1000000)
            t3 = time.time()

            print(f"Took {t3 - t2} to compute 1mm sum without the compiler.")

            self.assertGreater(t3 - t2, (t1 - t0) * 10)

        self.assertFalse(DisableCompiledCode.isDisabled())

        t4 = time.time()
        sum(1000000)
        t5 = time.time()

        print(f"Took {t5 - t4} to compute 1mm sum with the compiler turned back on.")

        # this should be fast again
        self.assertLess(t5 - t4, (t1 - t0) * 2)
