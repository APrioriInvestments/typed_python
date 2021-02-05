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

from typed_python import (
    ListOf, Class, Member, Final, TupleOf, DisableCompiledCode,
    isCompiled, SerializationContext
)
from typed_python._types import touchCompiledSpecializations
from typed_python import Entrypoint, NotCompiled
from typed_python.compiler.runtime import Runtime, RuntimeEventVisitor
from flaky import flaky
import pytest
import traceback
import threading
import time
import unittest
import numpy


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

        self.assertEqual(f.resultTypeFor(object).typeRepresentation, TupleOf(int))

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
        def add(x, y):
            return x + y

        compiledAdd = Entrypoint(add)

        compileCount = Runtime.singleton().timesCompiled

        for i in range(10):
            compiledAdd(i, 1)

        for i in range(10):
            compiledAdd(i, 1.5)

        self.assertEqual(Runtime.singleton().timesCompiled - compileCount, 2)

    @flaky(max_runs=3, min_passes=1)
    def test_specialized_entrypoint_dispatch_perf(self):
        def add(x, y):
            return x + y

        compiledAdd = Entrypoint(add)

        t0 = time.time()

        for i in range(1000000):
            compiledAdd(i, 1)

        # I get about .5 seconds on my laptop
        self.assertTrue(time.time() - t0 < 5.0, time.time() - t0)

    @flaky(max_runs=3, min_passes=1)
    def test_specialized_entrypoint_perf_difference(self):
        compiledAdd = Entrypoint(add)

        for T in [IntList, FloatList]:
            aList = T(range(1000000))
            compiledAdd(T([1]), 1)

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
            touchCompiledSpecializations(sumFun.overloads[0].functionTypeObject, 0)

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

        print(AClass(x=23).g)

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

    @flaky(max_runs=3, min_passes=1)
    @pytest.mark.skipif('sys.platform=="darwin"')
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

    @flaky(max_runs=3, min_passes=1)
    @pytest.mark.skipif('sys.platform=="darwin"')
    def test_nocompile_works(self):
        thisWouldNotBeVisible = set()

        @NotCompiled
        def f(x: float) -> float:
            assert not isCompiled()
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

    def test_call_entrypoint_with_default_argument(self):
        @Entrypoint
        def f(x, y=1, z=2):
            return x + y + z

        self.assertEqual(f(1), 4)
        self.assertEqual(f(1, 4), 7)

    def test_event_visitors(self):
        out = {}

        class Visitor(RuntimeEventVisitor):
            def onNewFunction(
                self,
                identifier,
                functionConverter,
                nativeFunction,
                funcName,
                funcCode,
                funcGlobals,
                closureVars,
                inputTypes,
                outputType,
                yieldType,
                variableTypes,
                conversionType
            ):
                out[funcName] = (inputTypes, outputType, variableTypes)

        @Entrypoint
        def f(x):
            y = x + 1
            return str(y)

        with Visitor():
            f.resultTypeFor(int)

        self.assertTrue('f' in out, out)
        self.assertEqual(out['f'][2]['y'], int)

    def test_star_args_on_entrypoint(self):
        @Entrypoint
        def argCount(*args):
            return len(args)

        self.assertEqual(argCount(1, 2, 3), 3)
        self.assertEqual(argCount(1, 2, 3, 4), 4)

    def test_star_args_on_entrypoint_typed(self):
        @Entrypoint
        def argCount():
            return 0

        @argCount.overload
        def argCount(x, *args: int):
            return x + argCount(*args)

        self.assertEqual(argCount(), 0)
        self.assertEqual(argCount(1), 1)
        self.assertEqual(argCount(1, 2), 3)

        with self.assertRaisesRegex(TypeError, "annot find a valid overload"):
            argCount(1, 2, "hi")

        self.assertEqual(argCount.resultTypeFor().typeRepresentation, int)
        self.assertEqual(argCount.resultTypeFor(int).typeRepresentation, int)
        self.assertEqual(argCount.resultTypeFor(int, int).typeRepresentation, int)

        self.assertEqual(argCount.resultTypeFor(int, int, str), None)

    def test_entrypoint_no_coercion(self):
        @Entrypoint
        def f(x):
            return type(x)

        self.assertEqual(f(1.5), float)
        self.assertEqual(f(1), int)

    def test_entrypoint_and_static_recursion_untyped_classes(self):
        class X:
            @staticmethod
            @Entrypoint
            def sum(x):
                if x > 0:
                    return X.sum(x - 1) + x
                return 0

        self.assertEqual(X.sum(10), sum(range(11)))

    def test_entrypoint_and_static_recursion_typed_classes(self):
        class X(Class):
            @staticmethod
            @Entrypoint
            def sum(x):
                if x > 0:
                    return X.sum(x - 1) + x
                return 0

        self.assertEqual(X.sum(10), sum(range(11)))

    def test_is_compiled(self):
        @Entrypoint
        def callIsCompiled():
            return isCompiled()

        self.assertTrue(callIsCompiled())

        with DisableCompiledCode():
            self.assertFalse(callIsCompiled())

    def test_entrypoint_always_compiles_when_called_with_numpy_array(self):
        @Entrypoint
        def f(x: ListOf(int)):
            assert isCompiled()

            res = 0
            for i in x:
                res += i
            return res

        a = numpy.arange(100).astype('int')

        self.assertEqual(f(a), a.sum())

    def test_can_compile_deserialized_functions(self):
        @Entrypoint
        def callIt(f, x):
            return f(x)

        def aFun(x):
            return x + 1

        sc = SerializationContext()

        aFun2 = sc.deserialize(sc.serialize(aFun))

        self.assertEqual(callIt(aFun2, 10), aFun2(10))

    def test_can_serialize_entrypoints(self):
        @Entrypoint
        def f(x):
            return x + 1

        sc = SerializationContext()

        f2 = sc.deserialize(sc.serialize(f))

        assert f2.isEntrypoint

        self.assertEqual(f(10), f2(10))

    def test_can_serialize_nocompile(self):
        @NotCompiled
        def f(x):
            return x + 1

        sc = SerializationContext()

        f2 = sc.deserialize(sc.serialize(f))

        assert f2.isNocompile

        self.assertEqual(f(10), f2(10))

    def test_can_serialize_functions_with_multiple_overloads(self):
        @Entrypoint
        def f(x):
            return x + 1

        @f.overload
        def f(x, y):
            return "yes"

        sc = SerializationContext()

        f2 = sc.deserialize(sc.serialize(f))

        self.assertEqual(f(10), f2(10))
        self.assertEqual(f(10, 11), f2(10, 11))

    def test_can_serialize_closures(self):
        x = 10

        def adder(y):
            return x + y

        self.assertEqual(adder(20), 30)

        sc = SerializationContext()

        adder2 = sc.deserialize(sc.serialize(adder))

        self.assertEqual(adder(10), adder2(10))

    def test_not_compiled_references_work(self):
        @NotCompiled
        def simple():
            return 10

        def makeNotCompiled(x):
            @NotCompiled
            def f():
                return x
            return f

        l1 = []
        l2 = []

        f1 = makeNotCompiled(l1)
        f2 = makeNotCompiled(l2)
        f3 = makeNotCompiled(l1)

        assert f1() is l1
        assert f2() is l2
        assert f3() is l1

        @Entrypoint
        def callIt(aFun):
            return aFun()

        assert callIt(simple) == 10
        assert callIt(f1) is l1
        assert callIt(f2) is l2
        assert callIt(f3) is l1
