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
import pytest
from flaky import flaky

from typed_python import (
    Class, Final, Function, NamedTuple, bytecount, ListOf, TypedCell, Forward,
    PyCell, Tuple, TupleOf, NotCompiled, TypeFunction, Member, typeKnownToCompiler,
    isCompiled
)

from typed_python.compiler.runtime import RuntimeEventVisitor, Entrypoint
from typed_python._types import refcount, identityHash
from typed_python.test_util import currentMemUsageMb


@NotCompiled
def moduleLevelNotCompiled(x) -> int:
    return x


class DidCompileVisitor(RuntimeEventVisitor):
    def __init__(self):
        super().__init__()

        self.didCompile = False

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
        self.didCompile = True


class TestCompilingClosures(unittest.TestCase):
    def test_closure_of_untyped_function(self):
        x = 10

        @Function
        def f():
            return x

        self.assertEqual(f.ClosureType.ElementTypes[0].ElementTypes[0], PyCell)
        self.assertTrue('x' in f.overloads[0].closureVarLookups)

    def test_lambda_with_same_code_doesnt_retrigger_compile(self):
        def makeAdder():
            def add(x, y):
                return x + y
            return add

        @Entrypoint
        def callFun(f, x, y):
            return f(x, y)

        vis = DidCompileVisitor()

        with vis:
            self.assertEqual(callFun(makeAdder(), 1, 2), 3)

        self.assertTrue(vis.didCompile)

        vis = DidCompileVisitor()

        with vis:
            self.assertEqual(callFun(makeAdder(), 1, 2), 3)

        # the second time, the code for the adder should have been the same, so
        # we shouldn't have triggered compilation.
        self.assertFalse(vis.didCompile)

    def test_closure_grabbing_types(self):
        T = int

        @Function
        def f(x):
            return T(x)

        pf = f.extractPyFun(0)
        self.assertEqual(pf.__closure__[0].cell_contents, T)
        self.assertEqual(f(1.5), 1)
        self.assertEqual(bytecount(f.ClosureType), 8)

    def test_closure_grabbing_closures(self):
        x = 10

        @Function
        def f():
            return x

        @Function
        def g():
            return f()

        self.assertEqual(g(), x)

    def test_calling_closure_with_bound_args(self):
        x = 10

        @Function
        def f(y):
            return y + x

        @Entrypoint
        def callIt(arg):
            return f(arg)

        self.assertEqual(callIt(20), 30)

    def test_passing_closures_as_arguments(self):
        x = 10

        @Function
        def f(y):
            return y + x

        @Entrypoint
        def callIt(f, arg):
            return f(arg)

        self.assertEqual(callIt(f, 20), 30)

    @flaky(max_runs=3, min_passes=1)
    def test_calling_closures_perf(self):
        ct = 1000000

        aList1 = ListOf(int)([])

        def makeAppender(l):
            @Function
            def append(y):
                l.append(y)
            return append

        @Entrypoint
        def callManyTimes(c1, ct):
            for i in range(ct):
                c1(i)

        callManyTimes(makeAppender(aList1), ct)

        aList1.clear()

        t0 = time.time()
        callManyTimes(makeAppender(aList1), ct)
        t1 = time.time()

        aList1.clear()

        elapsedCompiled = t1 - t0

        aList1 = []

        def makeAppender(l):
            def append(y):
                l.append(y)
            return append

        t0 = time.time()
        callManyTimes(makeAppender(aList1), ct)
        t1 = time.time()

        elapsedNontyped = t1 - t0

        print(elapsedCompiled, elapsedNontyped)

        print(elapsedNontyped / elapsedCompiled, " times faster")
        # for me, the compiled form is about 280 times faster than the uncompiled form
        # AlexT: I got from ~83x to ~240x on 2020/04/23, so I increased the count
        # which caused the variance to drop drastically (the elapsedCompiled is now
        # in the order of 10ms on my AWS cloud worker.
        self.assertLess(elapsedCompiled * 120, elapsedNontyped)

    @flaky(max_runs=3, min_passes=1)
    def test_assigning_closures_as_values(self):
        ct = 100000

        aList1 = ListOf(int)([])
        aList2 = ListOf(int)([])

        def makeAppender(l):
            @Function
            def append(y):
                l.append(y)
            return append

        @Entrypoint
        def alternatingCall(c1, c2, ct):
            for i in range(ct):
                c1(i)
                temp = c1
                c1 = c2
                c2 = temp

        c1 = makeAppender(aList1)
        c2 = makeAppender(aList2)

        assert identityHash(type(c1)) == identityHash(type(c2))

        self.assertEqual(type(c1), type(c2))

        alternatingCall(c1, c2, ct)

        self.assertEqual(len(aList1), ct // 2)
        self.assertEqual(len(aList2), ct // 2)

        aList1.clear()
        aList2.clear()

        t0 = time.time()
        alternatingCall(c1, c2, ct)
        t1 = time.time()

        aList1.clear()
        aList2.clear()

        elapsedCompiled = t1 - t0

        aList1 = []
        aList2 = []

        def makeAppender(l):
            def append(y):
                l.append(y)
            return append

        def alternatingCall(c1, c2, ct):
            for i in range(ct):
                c1(i)
                temp = c1
                c1 = c2
                c2 = temp

        c1 = makeAppender(aList1)
        c2 = makeAppender(aList2)

        t0 = time.time()
        alternatingCall(c1, c2, ct)
        t1 = time.time()

        elapsedNontyped = t1 - t0

        # for me, elapsedCompiled is 3x faster than elapsedNontyped, but
        # elapsedNoncompiled is about 5x slower than elapsedNontyped, because
        # typed python is not very efficient yet as an interpreter.
        # there is a _lot_ of overhead in repeatedly swapping because we
        # end up increffing the contained list many many times.
        self.assertTrue(elapsedCompiled * 2 < elapsedNontyped)

    @pytest.mark.skip(reason="compiled code can't call closures holding untyped cells yet.")
    def test_closure_in_listof(self):
        def makeAdder(x):
            @Function
            def f(y):
                return x + y
            return f

        self.assertEqual(type(makeAdder(10)), type(makeAdder(20)))

        T = ListOf(type(makeAdder(10)))

        aList = T()
        aList.append(makeAdder(1))
        aList.append(makeAdder(2))

        def callEachItemManyTimes(l, times):
            res = 0
            for count in range(times):
                for item in l:
                    res = item(res)
            return res

        resUncompiled = callEachItemManyTimes(aList, 10)

        resCompiled = Entrypoint(callEachItemManyTimes)(aList, 10)

        self.assertEqual(resUncompiled, resCompiled)

    def test_mutually_recursive_closures(self):
        @Function
        def f(x):
            if x == 0:
                return 0
            return g(x-1)

        @Function
        def g(x):
            return f(x-1)

        self.assertEqual(f.ClosureType.ElementTypes[0].ElementNames[0], 'g')
        self.assertEqual(f.ClosureType.ElementTypes[0].ElementTypes[0], PyCell)

        self.assertEqual(f(10), 0)

    def test_typed_cell(self):
        T = TypedCell(int)

        t = T()

        self.assertFalse(t.isSet())

        with self.assertRaises(TypeError):
            t.set("hi")

        with self.assertRaises(Exception):
            t.get()

        t.set(10)

        self.assertTrue(t.isSet())

        self.assertEqual(t.get(), 10)

        t.clear()

        self.assertFalse(t.isSet())

        with self.assertRaises(Exception):
            t.get()

    def test_typed_cell_in_tuple(self):
        TC = TypedCell(int)

        aTup = NamedTuple(x=TC)(x=TC())
        aTup.x.set(1)

    def test_typed_cell_with_forwards(self):
        Tup = Forward("Tup")
        Tup = Tup.define(NamedTuple(cell=TypedCell(Tup), x=int))
        TC = TypedCell(Tup)

        self.assertEqual(Tup.ElementTypes[0], TC)

        t1 = Tup(cell=TC(), x=1)
        t2 = Tup(cell=TC(), x=2)
        t1.cell.set(t2)
        t2.cell.set(t1)

        self.assertEqual(t1.cell.get().x, 2)
        self.assertEqual(t2.cell.get().x, 1)
        self.assertEqual(t1.cell.get().cell.get().x, 1)
        self.assertEqual(t2.cell.get().cell.get().x, 2)

    def test_function_overload_with_closures(self):
        @Function
        def f():
            return x

        @f.overload
        def f(anArg):
            return y

        @Function
        def f2(anArg, anArg2):
            return z

        @f2.overload
        def f2(anArg, anArg2, anArg3):
            return w

        x = 10
        y = 20
        z = 30
        w = 40

        self.assertEqual(f(), 10)
        self.assertEqual(f(1), 20)

        self.assertEqual(f2(1, 2), 30)
        self.assertEqual(f2(1, 2, 3), 40)

        combo = f.overload(f2)

        self.assertEqual(combo(), 10)
        self.assertEqual(combo(1), 20)
        self.assertEqual(combo(1, 2), 30)
        self.assertEqual(combo(1, 2, 3), 40)

    def test_pass_untyped_function_to_entrypoint_ignores_signature(self):
        @Entrypoint
        def callWithFloat(f):
            return f(1.5)

        def f(x: int):
            return x

        self.assertEqual(callWithFloat(f), 1.5)
        self.assertEqual(callWithFloat(Function(f)), 1)

    def test_pass_untyped_function_to_entrypoint(self):
        @Entrypoint
        def returnIt(f):
            return f

        def f(x):
            return x + 1

        returnIt(f)

    def test_pass_typed_function_to_entrypoint(self):
        @Entrypoint
        def returnIt(f):
            return f

        @Function
        def f(x):
            return x + 1

        self.assertEqual(returnIt(f)(10), 11)

    def test_access_untyped_function(self):
        @Entrypoint
        def callIt(x):
            return f(x)

        @Function
        def f(x):
            return x + 1

        self.assertEqual(callIt(10), 11)

    def test_access_untyped_function_with_string_in_closure(self):
        y = "hi"

        @Entrypoint
        def callIt(x):
            return f(x)

        @Function
        def f(x):
            return x + y

        self.assertEqual(callIt("10"), "10hi")

    def test_pass_function_with_bound_variable(self):
        @Entrypoint
        def returnIt(f):
            return f

        def f():
            return x + 1

        x = 10

        fRes = returnIt(f)

        self.assertEqual(fRes(), f())

        print(fRes.ClosureType)

        self.assertEqual(fRes.ClosureType, Tuple(int))

        x = 11

        fRes2 = returnIt(f)
        self.assertEqual(fRes2(), f())

        # we don't actually hold a typed closure to the parent. That's
        # different from python but unavoidable
        self.assertNotEqual(fRes(), fRes2())

    def test_pass_function_holding_function(self):
        @Entrypoint
        def returnIt(f):
            return f

        def f():
            return g()

        def g():
            return h()

        def h():
            return x

        x = 10

        fRes = returnIt(f)

        self.assertEqual(fRes.getClosure(), (x,))
        self.assertEqual(fRes(), x)

        print(fRes.ClosureType)
        print(fRes.overloads[0].closureVarLookups)
        print(fRes.withEntrypoint(True).isEntrypoint)

        self.assertEqual(fRes.withEntrypoint(True)(), x)

    def test_pass_mutually_recursive_functions(self):
        @Entrypoint
        def returnIt(f):
            return f

        @Entrypoint
        def f(x):
            if x < 0:
                return 100
            return g(x - 1) + 2

        @Entrypoint
        def g(x):
            return f(x - 1) * 2

        fRes = returnIt(f)

        self.assertEqual(fRes(10), f(10))

        # the second time we call it it shouldn't trigger compilation
        vis = DidCompileVisitor()
        with vis:
            returnIt(f)

        self.assertFalse(vis.didCompile)

        t0 = time.time()
        for _ in range(100000):
            returnIt(f)(1)

        # I get about .5 seconds. not as fast as I'd like, but reasonable given
        # the amount of work that's required to do this type inference separately
        # for each function call.
        print("took ", time.time() - t0, " to do 100k closure conversions")

    def test_no_recompile_for_same_function_body(self):
        def makeAdder(x):
            def adder(y):
                return x + y
            return adder

        @Entrypoint
        def callIt(f, x):
            return f(x)

        vis = DidCompileVisitor()
        with vis:
            callIt(makeAdder(10), 20)

        self.assertTrue(vis.didCompile)

        vis = DidCompileVisitor()
        with vis:
            callIt(makeAdder(30), 20)

        self.assertFalse(vis.didCompile)

    def test_pass_closures_from_compiled_code_with_no_capture(self):
        def doIt():
            def f():
                return 10

            def callIt(aFun):
                return aFun()

            return callIt(f)

        self.assertEqual(doIt(), Entrypoint(doIt)())

    def test_call_mutually_recursive_closures_from_compiled_code(self):
        def doIt(count):
            def f(x):
                return g(x-1)

            def g(x):
                if x < 0:
                    return 0
                return f(x - 1) + 1

            return f(count)

        compiledDoIt = Entrypoint(doIt)

        self.assertEqual(doIt(100), compiledDoIt(100))

    def test_closure_in_compiled_code_with_variable(self):
        def doIt():
            x = 10

            def f():
                return x

            return f()

        self.assertEqual(doIt(), Entrypoint(doIt)())

    def test_closure_in_compiled_code_with_variable_that_changes(self):
        def doIt():
            x = 10

            def f():
                return x

            x_0 = f()

            x = 20

            x_1 = f()

            return Tuple(int, int)((x_0, x_1))

        self.assertEqual(doIt(), Entrypoint(doIt)())

    def test_lambdas_in_compiled_code(self):
        def doIt(x):
            return (lambda y: x + y)(x)

        self.assertEqual(Entrypoint(doIt)(10), doIt(10))

    def test_closure_in_compiled_code_bind_pod_arg(self):
        def doIt(x):
            def f(y):
                return x + y

            return f(x)

        self.assertEqual(Entrypoint(doIt)(10), doIt(10))

    def test_closure_in_compiled_code_refcounts(self):
        def doIt(tup):
            def f(y):
                return tup + y

            return f(TupleOf(int)([1, 2, 3]))

        aTup = TupleOf(int)([1, 2, 3])

        res = Entrypoint(doIt)(aTup)
        self.assertEqual(res, doIt(aTup))
        self.assertEqual(refcount(aTup), 1)

    @flaky(max_runs=3, min_passes=1)
    def test_closure_var_lookup_speed(self):
        def sum(count, f):
            res = 0.0
            for i in range(count):
                res += f(i)

            return res

        @Entrypoint
        def sumIt(count):
            y = 1.0

            def f(i):
                return y

            return sum(count, f)

        @Entrypoint
        def sumItDirect(count):
            res = 0.0
            for i in range(count):
                res += 1.0
            return res

        sumIt(10)
        sumItDirect(10)

        t0 = time.time()
        sumIt(100000000)
        t1 = time.time()
        sumItDirect(100000000)
        t2 = time.time()

        closureTime = t1 - t0
        directTime = t2 - t1
        print(f"in closure, took: {closureTime}. in simple loop took {directTime}")
        self.assertTrue(.8 <= closureTime / directTime <= 1.2, closureTime / directTime)

    @flaky(max_runs=3, min_passes=1)
    def test_if_statement_def(self):
        def callIt(x):
            if x % 2:
                def f(y):
                    return y + 1.0
            else:
                def f(y):
                    return y + 2.0

            res = 0.0
            for i in range(x):
                res += f(i)

            return res

        callItE = Entrypoint(callIt)

        for i in range(10):
            self.assertEqual(callItE(i), callIt(i))

        t0 = time.time()
        callItE(1000000)
        t1 = time.time()
        callIt(1000000)
        t2 = time.time()

        speedup = (t2 - t1) / (t1 - t0)
        print("speedup is ", speedup)  # I get about 80
        self.assertGreater(speedup, 60)

    def test_assign_functions_with_closure_works(self):
        def callIt(x):
            y = 10.0

            if x % 2:
                def f(a):
                    return a + y + 1.0
            else:
                def f(a):
                    return a + y + 2.0

            res = 0.0
            for i in range(x):
                x = x + 1
                res += f(i)
            return res

        callItE = Entrypoint(callIt)

        for i in range(10):
            self.assertEqual(callItE(i), callIt(i))

    @flaky(max_runs=3, min_passes=1)
    def test_assign_functions_with_closure_perf(self):
        def callIt(x):
            y = 10.0

            if x % 2:
                def f(a):
                    return a + y + 1.0
            else:
                def f(a):
                    return a + y + 2.0

            res = 0.0
            for i in range(x):
                x = x + 1
                res += f(i)
            return res

        callItE = Entrypoint(callIt)

        for i in range(10):
            self.assertEqual(callItE(i), callIt(i))

        t0 = time.time()
        callItE(1000000)
        t1 = time.time()
        callIt(1000000)
        t2 = time.time()

        speedup = (t2 - t1) / (t1 - t0)
        print("speedup is ", speedup)  # I get about 8
        self.assertGreater(speedup, 6)

    def test_two_closure_vars(self):
        @Entrypoint
        def callIt(x):
            a = 9
            b = 11

            def f():
                return a + b

            if x:
                aFun = f
            else:
                aFun = None  # noqa

            return f()

        self.assertEqual(callIt(1), 20)

    def test_mutually_recursive_assigned_variables(self):
        def callIt(x):
            if x % 2:
                def f(y):
                    return g(y) + 0.0
            else:
                def f(y):
                    return g(y) + 1.0

            if x % 3:
                def g(y):
                    if y > 0:
                        return f(y-2)
                    else:
                        return 0.1
            else:
                def g(y):
                    if y > 0:
                        return f(y-2)
                    else:
                        return 0.2

            return g(x)

        callItE = Entrypoint(callIt)

        callItE(1)

        for i in range(10):
            self.assertEqual(callItE(i), callIt(i))

    def test_closure_bound_in_class(self):
        def makeClass(i):
            def fExternal():
                return i

            class C(Class, Final):
                def f(self):
                    return fExternal()

            return C

        C1 = makeClass(1)
        C2 = makeClass(2)

        assert identityHash(C1.f) != identityHash(C2.f)

        @Entrypoint
        def callF(c):
            return c.f()

        self.assertEqual(callF(C1()), 1)
        self.assertEqual(callF(C2()), 2)

    def test_calling_doesnt_leak(self):
        @Entrypoint
        def g(x):
            return x

        @Entrypoint
        def f(x):
            return g(x)

        f(10)

        m0 = currentMemUsageMb()
        t0 = time.time()

        while time.time() - t0 < 2.0:
            f(10)

        assert currentMemUsageMb() - m0 < 2.0

    def test_building_closures_doesnt_leak(self):
        m0 = currentMemUsageMb()
        t0 = time.time()

        y = 0

        while time.time() - t0 < 2.0:
            @Function
            def f(x):
                return x + y

        assert currentMemUsageMb() - m0 < 2.0

    def test_building_closures_of_closures_doesnt_leak(self):
        m0 = currentMemUsageMb()
        t0 = time.time()

        @NotCompiled
        def f(x):
            return x + 1

        while time.time() - t0 < 2.0:
            @Function
            def g(x):
                return f(x)

        assert currentMemUsageMb() - m0 < 2.0

    def test_building_notcompiled_with_closures_doesnt_leak(self):
        m0 = currentMemUsageMb()
        t0 = time.time()

        i = 0
        while time.time() - t0 < 2.0:
            def makeFun(z):
                @NotCompiled
                def g(x):
                    return x + z

                return g

            makeFun(i)
            i += 1

        assert currentMemUsageMb() - m0 < 2.0

    def test_calling_notcompiled_doesnt_leak(self):
        @Entrypoint
        def callIt(x):
            return moduleLevelNotCompiled(x)

        callIt(10)

        m0 = currentMemUsageMb()
        t0 = time.time()

        while time.time() - t0 < 2.0:
            callIt(10)

        assert currentMemUsageMb() - m0 < 2.0

    def test_calling_entrypoint_with_closures_doesnt_leak(self):
        @Entrypoint
        def callIt(f, x):
            return f(x)

        callIt(moduleLevelNotCompiled, 10)

        m0 = currentMemUsageMb()
        t0 = time.time()

        while time.time() - t0 < 2.0:
            callIt(moduleLevelNotCompiled, 10)

        assert currentMemUsageMb() - m0 < 2.0

    def test_type_inference_doesnt_leak(self):
        def g(x):
            return x + 2.0

        @Entrypoint
        def callIt(x):
            return g(x) + 1

        callIt.resultTypeFor(int)

        m0 = currentMemUsageMb()
        t0 = time.time()

        while time.time() - t0 < 2.0:
            callIt.resultTypeFor(int)

        assert currentMemUsageMb() - m0 < 1.0

    def test_copy_closure_with_cells(self):
        # we need the closure to hold something with a refcount
        # or else we don't generate a closure
        z = "hi"

        @Entrypoint
        def f(x):
            return x + z

        @TypeFunction
        def ClassHolding(T):
            class C(Class, Final):
                t = Member(T)

                def __init__(self, t):
                    self.t = t

            return C

        @Entrypoint
        def makeClassHolding(x):
            return ClassHolding(type(x))(x)

        r = makeClassHolding(f)

        # check that we are actually using a TypedCell
        assert issubclass(r.MemberTypes[0].ClosureType, TypedCell)

    def test_can_use_assigned_lambdas_in_compiled_code(self):
        @Entrypoint
        def callIt(x):
            f = lambda x: x * 2

            return f(1)

        assert callIt(10) == 2

    def test_can_use_reassigned_assigned_lambdas_in_compiled_code(self):
        @Entrypoint
        def callIt(x):
            if x % 2 == 0:
                f = lambda x: x * 2
            else:
                f = lambda x: x * 3

            return f(x) + f(1)

        assert callIt(10) == 22
        assert callIt(11) == 36

    def test_captured_closure_values_have_good_types(self):
        aList = ListOf(int)([1, 2, 3])

        @Entrypoint
        def callIt():
            return typeKnownToCompiler(aList)

        assert callIt() == ListOf(int)

    def test_captured_closure_values_in_class_methods_have_good_types(self):
        aList = ListOf(int)([1, 2, 3])

        class Cls(Class, Final):
            @staticmethod
            @Entrypoint
            def callIt():
                return typeKnownToCompiler(aList)

        assert Cls.callIt() == ListOf(int)

    def test_closure_lambdas_obey_not_compiled(self):
        @NotCompiled
        def g():
            assert not isCompiled()

        @Entrypoint
        def f():
            return g()

        f()

    def test_passed_lambdas_obey_not_compiled(self):
        @NotCompiled
        def g():
            assert not isCompiled()

        @Entrypoint
        def f(g):
            return g()

        f(g)

    def test_interior_lambdas_obey_not_compiled(self):
        @Entrypoint
        def f():
            @NotCompiled
            def g():
                assert not isCompiled()

            return g()

        f()

    def test_compiled_defs_obey_not_compiled(self):
        @Entrypoint
        def makeNocompile():
            @NotCompiled
            def g():
                pass

            return g

        assert makeNocompile().isNocompile

    def test_compiled_defs_obey_entrypoint(self):
        @Entrypoint
        def makeEntrypoint():
            @Entrypoint
            def g():
                pass

            return g

        assert makeEntrypoint().isEntrypoint

    def test_compile_closures_with_type_refs(self):
        class C(Class):
            pass

        class D(C):
            pass

        def final(x):
            return x

        def makeClosure(T1, T2):
            def h(x):
                return final(x + T2(10))

            def g(x: str):
                return h(T1(x))

            @Entrypoint
            def f(c: C, x):
                return g(x)

            return f

        def callIt2(f, x):
            return f(D(), x)

        @Entrypoint
        def callIt(f, x):
            return callIt2(f, x)

        assert callIt(makeClosure(float, int), "1.2") == 11.2
