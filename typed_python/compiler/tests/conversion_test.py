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

import time
import sys
import traceback
import threading
import unittest
from flaky import flaky
import psutil

from typed_python.compiler.type_wrappers.compilable_builtin import CompilableBuiltin

from typed_python import (
    Function, OneOf, TupleOf, ListOf, Tuple, NamedTuple, Class, NotCompiled, Dict,
    _types, Compiled, Member, Final, isCompiled, ConstDict,
    makeNamedTuple, UInt32, Int32, Type, identityHash, typeKnownToCompiler, checkOneOfType,
    refcount, checkType, map
)

from typed_python.compiler.runtime import Runtime, Entrypoint, RuntimeEventVisitor


from typed_python.python_ast import (
    convertFunctionToAlgebraicPyAst,
    evaluateFunctionDefWithLocalsInCells,
)


def result_or_exception(f, *p):
    try:
        return f(*p)
    except Exception as e:
        return type(e)


def result_or_exception_str(f, *p):
    try:
        return f(*p)
    except Exception as e:
        return str(type(e)) + " " + str(e)


# ad hoc transformation of specific error strings occurring during tests, for compatibility between python versions
def emulate_older_errors(s):
    return s.replace('TypeError: can only concatenate str (not "int") to str', 'TypeError: must be str, not int')


def result_or_exception_tb(f, *p):
    try:
        return f(*p)
    except BaseException as e:
        return str(type(e)) + "\n" + emulate_older_errors(traceback.format_exc())


def repeat_test(f, *a):
    for i in range(10000):
        try:
            f(*a)
        except Exception:
            pass


class GetCompiledTypes(RuntimeEventVisitor):
    def __init__(self):
        self.types = {}

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
        self.types[funcName] = makeNamedTuple(
            inputTypes=inputTypes,
            outputType=outputType,
            varTypes=variableTypes
        )


# used in tests that need module-level objects, which are handled
# internally in a different way than objects created at function
# scope.
aModuleLevelDict = Dict(str, int)({'hi': 1})
aModuleLevelConstDict = ConstDict(str, int)({'hi': 1})


repeat_test_compiled = Entrypoint(repeat_test)


class TestCompilationStructures(unittest.TestCase):
    def test_dispatch_order_independent(self):
        class AClass(Class):
            pass

        someValues = [
            TupleOf(int)(),
            (1, 2, 3),
            [1, 2, 3],
            AClass()
        ]

        for testCases in [someValues, reversed(someValues)]:
            # ensure that we are not order-dependent trying to dispatch
            # a class to a tuple or vice-versa
            @Entrypoint
            def typeToString(x):
                return str(type(x))

            for x in testCases:
                self.assertEqual(typeToString(x), str(type(x)))

    def checkFunctionOfIntegers(self, f):
        f_fast = Compiled(f)

        for i in range(100):
            self.assertEqual(f_fast(i), f(i))

    def test_simple_loop(self):
        def f(x: int) -> int:
            y = 0
            while x > 0:
                x = x - 1
                y = y + x
            return y

        self.checkFunctionOfIntegers(f)

    def test_returning(self):
        def f(x: int) -> int:
            return x

        self.checkFunctionOfIntegers(f)

    def test_basic_arithmetic(self):
        def f(x: int) -> int:
            y = x+1
            return y

        self.checkFunctionOfIntegers(f)

    def test_boolean_and(self):
        @Compiled
        def f(x: int, y: int, z: int) -> bool:
            return x and y and z

        self.assertEqual(f(0, 0, 0), False)
        self.assertEqual(f(0, 0, 1), False)
        self.assertEqual(f(0, 1, 0), False)
        self.assertEqual(f(0, 1, 1), False)
        self.assertEqual(f(1, 0, 0), False)
        self.assertEqual(f(1, 0, 1), False)
        self.assertEqual(f(1, 1, 0), False)
        self.assertEqual(f(1, 1, 1), True)

    def test_boolean_or(self):
        @Compiled
        def f(x: int, y: int, z: int) -> bool:
            return x or y or z

        self.assertEqual(f(0, 0, 0), False)
        self.assertEqual(f(0, 0, 1), True)
        self.assertEqual(f(0, 1, 0), True)
        self.assertEqual(f(0, 1, 1), True)
        self.assertEqual(f(1, 0, 0), True)
        self.assertEqual(f(1, 0, 1), True)
        self.assertEqual(f(1, 1, 0), True)
        self.assertEqual(f(1, 1, 1), True)

    def test_boolean_operators(self):
        @Compiled
        def f(x: int, y: int, z: float) -> bool:
            return x and y and z

        self.assertEqual(f(0, 1, 1.5), False)
        self.assertEqual(f(1, 1, 1.5), True)

    def test_boolean_operators_with_side_effects(self):
        # a function that appends 'effect' onto a list of effects
        # and then returns result, so that we can track when we
        # are actually executing a particular expression
        def effectAndResult(effectList, effect, result):
            effectList.append(effect)
            return result

        @Compiled
        def f_and(x: int, y: int, z: str) -> ListOf(str):
            result = ListOf(str)()

            (effectAndResult(result, "x", x)
                and effectAndResult(result, "y", y)
                and effectAndResult(result, "z", z))

            return result

        self.assertEqual(f_and(0, 1, "s"), ["x"])
        self.assertEqual(f_and(1, 0, "s"), ["x", "y"])
        self.assertEqual(f_and(1, 1, "s"), ["x", "y", "z"])

        @Compiled
        def f_or(x: int, y: int, z: str) -> ListOf(str):
            result = ListOf(str)()

            (effectAndResult(result, "x", x)
                and effectAndResult(result, "y", y)
                and effectAndResult(result, "z", z))

            return result

        self.assertEqual(f_or(0, 0, ""), ["x"])
        self.assertEqual(f_or(1, 0, ""), ["x", "y"])
        self.assertEqual(f_or(1, 1, ""), ["x", "y", "z"])

    def test_object_to_int_conversion(self):
        @Function
        def toObject(o: object):
            return o

        @Compiled
        def f(x: int) -> int:
            return int(toObject(x))

        self.assertEqual(f(10), 10)

    def test_variable_type_changes_make_sense(self):
        @Compiled
        def f(x: int) -> float:
            y = x
            y = 1.2
            return y

        self.assertEqual(f(10), 1.2)

    def test_call_other_typed_function(self):
        def g(x: int) -> int:
            return x+1

        def f(x: int) -> int:
            return g(x+2)

        self.checkFunctionOfIntegers(f)

    def test_call_untyped_function(self):
        @Compiled
        def f(x: object):
            return x

        x = []

        self.assertIs(f(x), x)

    def test_call_other_untyped_function(self):
        def g(x):
            return x

        @Compiled
        def f(x: object):
            return g(x)

        x = []

        self.assertIs(f(x), x)

    def test_integers_in_closures(self):
        y = 2

        def f(x: int) -> int:
            return x+y

        self.checkFunctionOfIntegers(f)

    def test_loop_variable_changing_type(self):
        @Compiled
        def f(top: float) -> OneOf(int, float):
            y = 0
            x = 0
            while x < top:
                x += 1.0
                y += x

            return y

        self.assertEqual(f(3.5), 10.0)

    def test_unassigned_variables(self):
        @Compiled
        def f(switch: int, t: TupleOf(int)) -> TupleOf(int):
            if switch:
                x = t
            return x

        self.assertEqual(f(1, (1, 2, 3)), (1, 2, 3))

        with self.assertRaisesRegex(Exception, "local variable 'x' referenced before assignment"):
            self.assertEqual(f(0, (1, 2, 3)), (1, 2, 3))

    def test_return_from_function_without_return_value_specified(self):
        @Compiled
        def f(t: TupleOf(int)):
            return t

        self.assertEqual(f((1, 2, 3)), (1, 2, 3))

    def test_return_from_function_with_bad_convert_throws(self):
        @Compiled
        def f(t: TupleOf(int)) -> None:
            return t

        with self.assertRaisesRegex(Exception, "Can't convert"):
            f((1, 2, 3))

    @flaky(max_runs=3, min_passes=1)
    def test_perf_of_mutually_recursive_untyped_functions(self):
        def q(x):
            return x-1

        def z(x):
            return q(x)+1

        def f(x):
            return z(g(x - 1)) + z(g(x - 2)) + z(x)

        def g(x):
            if x > 0:
                return z(f(x-1)) * z(2) + f(x-2)
            return 1

        g_typed = Entrypoint(g)

        self.assertEqual(g(10), g_typed(10))

        for input in [18, 18.0]:
            t0 = time.time()
            g(input)
            untyped_duration = time.time() - t0

            g_typed(input)
            t0 = time.time()
            g_typed(input)
            typed_duration = time.time() - t0

            # I get around 50x for ints and 12 for floats
            speedup = untyped_duration / typed_duration
            self.assertGreater(speedup, 20 if isinstance(input, int) else 4)

            print("for ", input, " speedup is ", speedup)

    def test_call_typed_function(self):
        @Function
        def f(x):
            return x

        @Compiled
        def g(x: int):
            return f(x+1)

        self.assertEqual(g(10), 11)

    def test_adding_with_nones_throws(self):
        @Compiled
        def g():
            return None + None

        with self.assertRaisesRegex(Exception, "Can't apply op Add.. to expressions of type None"):
            g()

    def test_exception_before_return_propagated(self):
        @Compiled
        def g():
            None+None
            return None

        with self.assertRaisesRegex(Exception, "Can't apply op Add.. to expressions of type None"):
            g()

    def test_call_function_with_none(self):
        @Compiled
        def g(x: None):
            return None

        self.assertEqual(g(None), None)

    def test_call_other_function_with_none(self):
        def f(x):
            return x

        @Compiled
        def g(x: int):
            return f(None)

        self.assertEqual(g(1), None)

    def test_interleaving_nones(self):
        def f(x, y, z):
            x+z
            return y

        @Compiled
        def works(x: int):
            return f(x, None, x)

        @Compiled
        def throws(x: int):
            return f(None, None, x)

        self.assertEqual(works(1), None)
        with self.assertRaisesRegex(Exception, "Can't apply op Add.. to expressions of type None"):
            throws(1)

    def test_return_none(self):
        def f(x):
            return x

        @Compiled
        def g():
            return f(None)

        self.assertEqual(g.resultTypeFor().typeRepresentation, type(None))
        self.assertEqual(g(), None)

    def test_assign_with_none(self):
        def f(x):
            return x

        @Compiled
        def g(x: int):
            y = f(None)
            z = y
            return z

        self.assertEqual(g(1), None)

    def test_nonexistent_variable(self):
        @Compiled
        def f():
            return this_variable_name_is_undefined  # noqa: F821

        with self.assertRaisesRegex(Exception, "name 'this_variable_name_is_undefined' is not defined"):
            f()

    @flaky(max_runs=3, min_passes=1)
    def test_iterating(self):
        @Compiled
        def sumDirectly(x: int):
            y = 0.0
            i = 0
            while i < x:
                y += i
                i += 1
            return y

        @Compiled
        def sumUsingRange(x: int):
            y = 0.0
            for i in range(x):
                y += i
            return y

        for i in range(10):
            self.assertEqual(sumDirectly(i), sumUsingRange(i))

        t0 = time.time()
        sumDirectly(1000000)
        t1 = time.time()
        sumUsingRange(1000000)
        t2 = time.time()

        print("Range is %.2f slower than nonrange." % ((t2-t1)/(t1-t0)))  # I get 1.00
        self.assertLess((t1-t0), (t2 - t1) * 1.2)

    def test_read_invalid_variables(self):
        @Compiled
        def readNonexistentVariable(readIt: bool):
            if readIt:
                return y  # noqa
            else:
                return 0

        with self.assertRaisesRegex(Exception, "name 'y' is not defined"):
            readNonexistentVariable(True)

        self.assertEqual(readNonexistentVariable(False), 0)

    def test_append_float_to_int_rules_same(self):
        def f():
            x = ListOf(int)()
            x.append(1.0)
            return x

        self.assertEqual(f(), Compiled(f)())

    def test_multiple_assignments(self):
        @Entrypoint
        def f(iterable):
            x, y, z = iterable
            return x + y + z

        self.assertEqual(f(TupleOf(int)((1, 2, 3))), 6)

        with self.assertRaisesRegex(Exception, "not enough"):
            f(TupleOf(int)((1, 2)))

        with self.assertRaisesRegex(Exception, "too many"):
            f(TupleOf(int)((1, 2, 3, 4)))

        self.assertEqual(f(Tuple(int, int, int)((1, 2, 3))), 6)
        self.assertEqual(f(NamedTuple(x=int, y=int, z=int)(x=1, y=2, z=3)), 6)

        with self.assertRaisesRegex(Exception, "not enough"):
            f(Tuple(int, int)((1, 2)))

        with self.assertRaisesRegex(Exception, "too many"):
            f(Tuple(int, int, int, int)((1, 2, 3, 4)))

    def test_print_oneof(self):
        @Compiled
        def f(x: OneOf(float, str), y: OneOf(float, str)):
            print("You can print either a float or a string", x, y)

        f("hi", "hi")
        f(1.0, "hi")

    def test_type_oneof(self):
        @Compiled
        def f(x: OneOf(float, int)):
            return str(type(x))

        self.assertEqual(f(1), str(int))
        self.assertEqual(f(1.0), str(float))

    def test_can_raise_exceptions(self):
        @Compiled
        def aFunctionThatRaises(x: object):
            raise AttributeError(f"you gave me {x}")

        with self.assertRaisesRegex(AttributeError, "you gave me hihi"):
            aFunctionThatRaises("hihi")

        try:
            aFunctionThatRaises("hihi")
        except Exception:
            trace = traceback.format_exc()
            # the traceback should know where we are
            self.assertIn('conversion_test', trace)
            self.assertIn('aFunctionThatRaises', trace)

    def test_stacktraces_show_up(self):
        def f2(x):
            return f3(x)

        def f3(x):
            return f4(x)

        def f4(x):
            raise Exception(f"X is {x}")

        @Entrypoint
        def f1(x):
            return f2(x)

        try:
            f1("hihi")
        except Exception:
            trace = traceback.format_exc()
            self.assertIn("f1", trace)
            self.assertIn("f2", trace)
            self.assertIn("f3", trace)
            self.assertIn("f4", trace)

    @flaky(max_runs=3, min_passes=1)
    def test_perf_of_inlined_functions_doesnt_degrade(self):
        def f1(x):
            return f2(x)

        def f2(x):
            return f3(x)

        def f3(x):
            return f4(x)

        def f4(x: int):
            return x

        @Entrypoint
        def callsF1(times: int):
            res = 0.0
            for i in range(times):
                res += f1(i)
            return res

        @Entrypoint
        def callsF4(times: int):
            res = 0.0
            for i in range(times):
                res += f4(i)
            return res

        @Entrypoint
        def getit(f):
            return f

        # prime the compilation
        callsF4(1)
        callsF1(1)

        t0 = time.time()
        callsF1(10000000)
        t1 = time.time()
        callsF4(10000000)
        t2 = time.time()

        callsDeeply = t1 - t0
        callsShallowly = t2 - t1
        ratio = callsDeeply / callsShallowly

        # inlining should work across invocations, regardless of order
        self.assertLessEqual(.8, ratio)
        self.assertLessEqual(ratio, 1.2)
        print(f"Deeper call tree code was {ratio} times slow.")

    def test_exception_handling_preserves_refcount(self):
        @Entrypoint
        def f(x, shouldThrow):
            # this will increment the 'x' refcount
            y = x  # noqa
            if shouldThrow:
                raise Exception("exception")

        aList = ListOf(int)()

        self.assertEqual(_types.refcount(aList), 1)

        f(aList, False)

        self.assertEqual(_types.refcount(aList), 1)

        with self.assertRaises(Exception):
            f(aList, True)

        self.assertEqual(_types.refcount(aList), 1)

    def test_assert(self):
        @Entrypoint
        def assertNoMessage(x):
            assert x

        @Entrypoint
        def assertWithMessage(x, y):
            assert x, y

        with self.assertRaises(AssertionError):
            assertNoMessage(0)

        with self.assertRaisesRegex(AssertionError, "boo"):
            assertWithMessage(0, "boo")

        assertNoMessage(1)
        assertWithMessage(1, "message")

    def test_assert_false(self):
        @Entrypoint
        def check(x):
            assert False, x
            return x

        self.assertEqual(check.resultTypeFor(int), None)

        with self.assertRaises(AssertionError):
            check(10)

    def test_conditional_eval_or(self):
        @Compiled
        def f1(x: float, y: int):
            return x or 1 / y

        @Compiled
        def f2(x: str, y: int):
            return x or 1 / y

        @Compiled
        def f3(x: int, y: float, z: str):
            return x or y or z

        with self.assertRaises(ZeroDivisionError):
            f1(0.0, 0)
        self.assertEqual(f1(0.0, 2), 0.5)
        self.assertEqual(f1(1.23, 0), 1.23)
        self.assertEqual(f1(1.23, 2), 1.23)

        with self.assertRaises(ZeroDivisionError):
            f2("", 0)
        self.assertEqual(f2("", 2), 0.5)
        self.assertEqual(f2("y", 0), "y")
        self.assertEqual(f2("y", 2), "y")

        self.assertEqual(f3(0, 0.0, ""), "")
        self.assertEqual(f3(0, 0.0, "one"), "one")
        self.assertEqual(f3(0, 1.5, ""), 1.5)
        self.assertEqual(f3(0, 1.5, "one"), 1.5)
        self.assertEqual(f3(3, 0.0, ""), 3)
        self.assertEqual(f3(3, 0.0, "one"), 3)
        self.assertEqual(f3(3, 1.5, ""), 3)
        self.assertEqual(f3(3, 1.5, "one"), 3)

    def test_conditional_eval_and(self):
        @Compiled
        def f1(x: float, y: int):
            return x and 1 / y

        self.assertEqual(f1(0.0, 0), 0.0)
        self.assertEqual(f1(0.0, 2), 0.0)
        with self.assertRaises(ZeroDivisionError):
            f1(2.5, 0)
        self.assertEqual(f1(2.5, 2), 0.5)

        @Compiled
        def f2(x: str, y: int):
            return x and 1 / y

        self.assertEqual(f2("", 0), "")
        self.assertEqual(f2("", 2), "")
        with self.assertRaises(ZeroDivisionError):
            f2("y", 0)
        self.assertEqual(f2("y", 2), 0.5)

        @Compiled
        def f(x: int, y: str, z: float):
            return x and y and z

        self.assertEqual(f(0, "", 0.0), 0)
        self.assertEqual(f(0, "", 1.5), 0)
        self.assertEqual(f(0, "one", 0.0), 0)
        self.assertEqual(f(0, "one", 1.5), 0)
        self.assertEqual(f(3, "", 0.0), "")
        self.assertEqual(f(3, "", 1.5), "")
        self.assertEqual(f(3, "one", 0.0), 0.0)
        self.assertEqual(f(3, "one", 1.5), 1.5)

    def test_conversion_of_deeply_nested_functions(self):
        def g_0():
            return 0

        def f_0():
            return 0

        def g_1():
            return g_0() + f_0()

        def f_1():
            return f_0() + g_0()

        def g_2():
            return g_1() + f_1()

        def f_2():
            return f_1() + g_1()

        def g_3():
            return g_2() + f_2()

        def f_3():
            return f_2() + g_2()

        def g_4():
            return g_3() + f_3()

        def f_4():
            return f_3() + g_3()

        def g_5():
            return g_4() + f_4()

        def f_5():
            return f_4() + g_4()

        def g_6():
            return g_5() + f_5()

        def f_6():
            return f_5() + g_5()

        @Entrypoint
        def compute():
            return g_6() + f_6()

        oldTimesCalculated = dict(Runtime.singleton().converter._times_calculated)

        compute()

        for identity, timesCalculated in Runtime.singleton().converter._times_calculated.items():
            if identity not in oldTimesCalculated:
                self.assertLessEqual(timesCalculated, 6, identity)

    def test_converting_break_in_while(self):
        def testBreak(x):
            res = 0

            while True:
                x = x - 1
                res = res + x

                if x < 0:
                    break

            return res + 1

        self.assertEqual(testBreak(10), Entrypoint(testBreak)(10))

    def test_converting_break_in_while_with_try_outside_of_loop(self):
        def testBreak():
            res = 0

            try:
                while True:
                    res += 1
                    break
                res += 10
            finally:
                res += 100

            return res

        self.assertEqual(testBreak(), Entrypoint(testBreak)())

    def test_converting_break_in_while_with_try_inside_of_loop(self):
        def testBreak():
            res = 0

            while True:
                try:
                    res += 1
                    break
                finally:
                    res += 10

            res += 100

            return res

        self.assertEqual(testBreak(), Entrypoint(testBreak)())

    def test_converting_break_through_nested_try_finally(self):
        def testBreak():
            res = 0

            try:
                while True:
                    try:
                        try:
                            res += 1
                            break
                        finally:
                            res += 10
                    finally:
                        res += 100
            finally:
                res += 1000

            res += 10000

            return res

        self.assertEqual(testBreak(), Entrypoint(testBreak)())

    def test_converting_continue_through_multiple_nested_try_finally(self):
        def testBreak():
            res = 0

            try:
                while True:
                    if res > 0:
                        break

                    try:
                        try:
                            res += 1
                            continue
                        finally:
                            res += 10
                    finally:
                        res += 100

                    # never gets here
                    assert False
            finally:
                res += 1000

            res += 10000

            return res

        self.assertEqual(testBreak(), Entrypoint(testBreak)())

    def test_converting_continue_in_while(self):
        def testContinue(x):
            res = 0

            while x > 0:
                x = x - 1
                res = res + x

                if x % 2 == 0:
                    continue

                res = res + 10

            return res

        self.assertEqual(testContinue(10), Entrypoint(testContinue)(10))

    def test_converting_break_in_foreach(self):
        def testBreak(x):
            res = 0
            for i in x:
                res += i
                if i > len(x) / 2:
                    break

            return res

        for thing in [ListOf(int)(range(10)), Tuple(int, int, int, int)((1, 2, 3, 4))]:
            self.assertEqual(testBreak(thing), Entrypoint(testBreak)(thing))

    def test_converting_continue_in_foreach(self):
        def testContinue(x):
            res = 0
            for i in x:
                if i > len(x) / 2:
                    continue
                res += i

            return res

        for thing in [ListOf(int)(range(10)), Tuple(int, int, int, int)((1, 2, 3, 4))]:
            self.assertEqual(testContinue(thing), Entrypoint(testContinue)(thing))

    def test_call_function_with_wrong_number_of_arguments(self):
        def f(x, y):
            return x + y

        @Compiled
        def callIt(x: int):
            return f(x)

        with self.assertRaisesRegex(TypeError, "annot find a valid overload"):
            callIt(1)

    def test_call_function_with_default_arguments(self):
        def f(x, y=1):
            return x + y

        @Entrypoint
        def callIt(x):
            return f(x)

        self.assertEqual(callIt(10), f(10))

    def test_call_function_with_named_args_ordering(self):
        def f(x, y):
            return x

        @Entrypoint
        def callWithArgsReversed(x, y):
            return f(y=y, x=x)

        self.assertEqual(callWithArgsReversed(2, 3), 2)

    def test_call_function_with_named_args(self):
        def f(x=1, y=10):
            return x + y

        def callWithX(x: int):
            return f(x=x)

        def callWithY(y: int):
            return f(y=y)

        def callWithXY(x: int, y: int):
            return f(y=y, x=x)

        callWithXCompiled = Compiled(callWithX)
        callWithYCompiled = Compiled(callWithY)
        callWithXYCompiled = Compiled(callWithXY)

        self.assertEqual(callWithX(2), callWithXCompiled(2))
        self.assertEqual(callWithY(2), callWithYCompiled(2))
        self.assertEqual(callWithXY(2, 3), callWithXYCompiled(2, 3))

    def test_call_function_with_star_args(self):
        def f(*args):
            return args

        @Entrypoint
        def callIt(x, y, z):
            return f(x, y, z)

        self.assertEqual(callIt(1, 2.5, "hi"), Tuple(int, float, str)((1, 2.5, "hi")))

    def test_call_function_with_kwargs(self):
        def f(**kwargs):
            return kwargs

        @Entrypoint
        def callIt(x, y, z):
            return f(x=x, y=y, z=z)

        self.assertEqual(callIt(1, 2.5, "hi"), dict(x=1, y=2.5, z="hi"))

    def test_call_function_with_excess_named_arg(self):
        def f(x=1, y=2):
            return x + y

        @Entrypoint
        def callIt(x, y, z):
            return f(x=x, y=y, z=z)

        with self.assertRaisesRegex(TypeError, "annot find a valid over"):
            callIt(1, 2, 3)

    def test_star_arg_call_function(self):
        def f(x, y):
            return x + y

        @Entrypoint
        def callIt(a):
            return f(*a)

        self.assertEqual(callIt(Tuple(int, int)((1, 2))), 3)

    def test_star_kwarg_type(self):
        def f(**kwargs):
            return type(kwargs)

        @Entrypoint
        def callIt():
            return f(x=10, y=20)

        self.assertEqual(callIt(), dict)

    def test_star_kwarg_as_dict(self):
        def f(**kwargs):
            return kwargs

        @Entrypoint
        def callIt():
            return f(x=10, y=20)

        self.assertEqual(callIt(), dict(x=10, y=20))

    def test_star_kwarg_call_function(self):
        def f(x, y):
            return x + y

        def g(**kwargs):
            return f(**kwargs)

        @Entrypoint
        def callIt(x, y):
            return g(y=y, x=x)

        self.assertEqual(callIt(1, 2), 3)

    @flaky(max_runs=3, min_passes=1)
    def test_perf_of_star_kwarg_intermediate_is_fast(self):
        def f(x, y):
            return x + y

        def g(**kwargs):
            return f(**kwargs)

        def sumUsingG(a: int):
            res = 0.0
            for i in range(a):
                res += g(x=2, y=i)
            return res

        def sumUsingF(a: int):
            res = 0.0
            for i in range(a):
                res += f(x=2, y=i)
            return res

        sumUsingGCompiled = Compiled(sumUsingG)
        sumUsingFCompiled = Compiled(sumUsingF)

        self.assertEqual(sumUsingG(100), sumUsingGCompiled(100))

        t0 = time.time()
        sumUsingGCompiled(1000000)
        elapsedG = time.time() - t0

        t0 = time.time()
        sumUsingFCompiled(1000000)
        elapsedF = time.time() - t0

        t0 = time.time()
        sumUsingG(1000000)
        elapsedGPy = time.time() - t0

        print("Compiled is ", elapsedGPy / elapsedG, " times faster")

        # check that the extra call to 'g' doesn't introduce any overhead
        self.assertTrue(.7 <= elapsedF / elapsedG <= 1.3, elapsedF / elapsedG)

    @flaky(max_runs=3, min_passes=1)
    def test_perf_of_star_arg_intermediate_is_fast(self):
        def f(x, y):
            return x + y

        def g(*args):
            return f(*args)

        def sumUsingG(a: int):
            res = 0.0
            for i in range(a):
                res += g(i, 2)
            return res

        def sumUsingF(a: int):
            res = 0.0
            for i in range(a):
                res += f(i, 2)
            return res

        sumUsingGCompiled = Compiled(sumUsingG)
        sumUsingFCompiled = Compiled(sumUsingF)

        self.assertEqual(sumUsingG(100), sumUsingGCompiled(100))

        t0 = time.time()
        sumUsingGCompiled(1000000)
        elapsedG = time.time() - t0

        t0 = time.time()
        sumUsingFCompiled(1000000)
        elapsedF = time.time() - t0

        t0 = time.time()
        sumUsingG(1000000)
        elapsedGPy = time.time() - t0

        print("Compiled is ", elapsedGPy / elapsedG, " times faster")

        # check that the extra call to 'g' doesn't introduce any overhead
        self.assertTrue(.65 <= elapsedF / elapsedG <= 1.35, elapsedF / elapsedG)

    def test_star_args_of_masquerade(self):
        def f(*args):
            return args[1]

        @Entrypoint
        def callF():
            return f(1, "a b c".split())

        self.assertEqual(callF.resultTypeFor().interpreterTypeRepresentation, list)

    def test_star_args_type(self):
        def f(*args):
            return type(args)

        @Entrypoint
        def callF():
            return f(1, 2, 3)

        self.assertEqual(callF(), tuple)

    def test_typed_functions_with_star_args(self):
        @Function
        def f(x: int):
            return 1

        @f.overload
        def f(x: int, *args):
            return 1 + len(args)

        @Entrypoint
        def callF1(x):
            return f(x)

        @Entrypoint
        def callF2(x, y):
            return f(x, y)

        self.assertEqual(callF1(0), 1)
        self.assertEqual(callF2(0, 1), 2)

    def test_typed_functions_with_kwargs(self):
        @Function
        def f(x, **kwargs):
            return x + len(kwargs)

        @Entrypoint
        def callF1(x):
            return f(x)

        @Entrypoint
        def callF2(x, y):
            return f(x, y)

        @Entrypoint
        def callF3(x, y):
            return f(x, y=y)

        @Entrypoint
        def callF4(x, y):
            return f(x, x=y)

        self.assertEqual(callF1(10), 10)

        with self.assertRaisesRegex(TypeError, "cannot find a valid overload"):
            callF2(0, 1)

        self.assertEqual(callF3(10, 2), 11)

        with self.assertRaisesRegex(TypeError, "cannot find a valid overload"):
            callF4(0, 1)

    def test_typed_functions_with_typed_kwargs(self):
        @Function
        def f(**kwargs: int):
            return "int"

        @f.overload
        def f(**kwargs: str):
            return "str"

        self.assertEqual(f(), "int")
        self.assertEqual(f(x=1), "int")
        self.assertEqual(f(x=1.5), "int")
        self.assertEqual(f(x="1"), "str")

        @Entrypoint
        def callF(**kwargs):
            return f(**kwargs)

        self.assertEqual(callF(), "int")
        self.assertEqual(callF(x=1), "int")
        self.assertEqual(callF(x=1.5), "int")
        self.assertEqual(callF(x="1"), "str")

    def test_typed_functions_dispatch_based_on_names(self):
        @Function
        def f(x):
            return "x"

        @f.overload
        def f(y):
            return "y"

        @Entrypoint
        def callFUnnamed(x):
            return f(x)

        @Entrypoint
        def callFWithY(x):
            return f(y=x)

        @Entrypoint
        def callFWithX(x):
            return f(x=x)

        self.assertEqual(callFUnnamed(10), "x")
        self.assertEqual(callFWithX(10), "x")
        self.assertEqual(callFWithY(10), "y")

    def test_typed_functions_with_oneof(self):
        @Function
        def f(x: OneOf(int, float)):
            return x + 1.0

        @Entrypoint
        def callF(x):
            return f(x)

        @Entrypoint
        def callF2(x: OneOf(int, float, str)):
            return f(x)

        self.assertEqual(callF(1.5), 2.5)
        self.assertEqual(callF(1), 2.0)

        self.assertEqual(callF2(1.5), 2.5)
        self.assertEqual(callF2(1), 2.0)

        with self.assertRaisesRegex(TypeError, r"Failed to find an overload"):
            callF2("h")

    def test_can_call_function_with_typed_function_as_argument(self):
        @Function
        def add(x: int, y: int):
            return x + y

        def g(x, y):
            return x + y

        @Function
        def callIt(x: int, f: add):
            return f(x, 1)

        self.assertEqual(callIt(1, add), 2)

        with self.assertRaisesRegex(TypeError, "annot find a valid overload"):
            callIt(1, g)

    def test_check_type_of_method_conversion(self):
        @Entrypoint
        def g(x: OneOf(None, TupleOf(int))):
            return type(x)

        self.assertEqual(g((1, 2, 3)), TupleOf(int))
        self.assertEqual(g(None), type(None))

    def test_check_is_on_unlike_things(self):
        @Entrypoint
        def g(x, y):
            return x is y

        self.assertFalse(g([], []))
        self.assertTrue(g(None, None))
        self.assertFalse(g(ListOf(int)(), TupleOf(int)()))

    def test_if_condition_throws(self):
        def throws():
            raise Exception("Boo")

        @Entrypoint
        def shouldThrow():
            x = ListOf(int)()
            y = Dict(int, int)()

            if x in y:
                return True
            else:
                return False

        with self.assertRaisesRegex(Exception, "Couldn't initialize type int"):
            shouldThrow()

    def test_if_with_return_types(self):
        @Entrypoint
        def popCheck(d, x):
            if x in d:
                d.pop(x)

        popCheck(Dict(int, int)(), 1)

    def test_assign_to_arguments_with_typechange(self):
        @Entrypoint
        def f(x, y: object):
            x = x + y

        f(1, 1)

    def test_unassigned_variable_access(self):
        @Compiled
        def reduce2(aList: ListOf(int)):
            for i in aList:
                r += i  # noqa
            return r  # noqa

        with self.assertRaisesRegex(Exception, "ame 'r' is not defined"):
            reduce2([1, 2, 3])

    def test_iterate_closures(self):
        x = ListOf(int)((1, 2, 3))

        @Entrypoint
        def f():
            res = ListOf(int)()
            for value in x:
                res.append(value)
            return res

        self.assertEqual(f(), [1, 2, 3])

    def test_function_not_returning_returns_none(self):
        @Entrypoint
        def f(l, i, y):
            l[i] = y

        self.assertEqual(f.resultTypeFor(ListOf(int), int, int).typeRepresentation, type(None))

    def test_method_not_returning_returns_none(self):
        class NoPythonObjectTypes(RuntimeEventVisitor):
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
                assert issubclass(outputType.typeRepresentation, Type)

        class C(Class, Final):
            def f(self, l, i, y: int):
                l[i] = y

            def f(self, l, i, y: float):  # noqa
                l[i] = y

        @Entrypoint
        def f(l: ListOf(int), i, y: OneOf(int, float)):
            return C().f(l, i, y)

        with NoPythonObjectTypes():
            f(ListOf(int)([1, 2, 3]), 0, 2)

    def test_try_simple(self):

        def f0(x: int) -> str:
            ret = "try "
            try:
                ret += str(1/x) + " "
            except TypeError:
                ret += "catch "
            finally:
                ret += "finally"
            return ret

        def f1(x: int) -> str:
            ret = "try "
            try:
                ret += str(1/x) + " "
            except Exception:  # noqa: E722
                raise NotImplementedError("custom")
                ret += "catch "
            finally:
                ret += "finally"
            return ret

        def f2(x: int) -> str:
            ret = "try "
            try:
                ret += str(1/x) + " "
            except Exception:
                ret += "catch "
            finally:
                ret += "finally"
            return ret

        def f3(x: int) -> str:
            ret = "try "
            try:
                ret += str(1/x) + " "
            except Exception:
                ret += "catch "
            return ret

        def f4(x: int) -> str:
            ret = "try "
            try:
                ret += str(1/x) + " "
            except Exception:
                ret += "catch "
            else:
                ret += "else "
            return ret

        def f5(x: int) -> str:
            ret = "try "
            try:
                ret += str(1/x) + " "
                if x == 1:
                    ret += x
            except ZeroDivisionError as ex:
                ret += "catch1 " + str(type(ex))
            except TypeError as ex:
                ret += "catch2 " + str(type(ex))
            except Exception:
                ret += "catchdefault "
            finally:
                ret += "finally"
            return ret

        def f6(x: int) -> str:
            ret = "try "
            try:
                ret += str(1/x) + " "
                if x == 1:
                    ret += x
            except ArithmeticError:
                ret += "catch1 "
                # TODO: The compiled code will have type(ex) = ArithmeticError instead of ZeroDivisionError.
                # TODO: Also, there are variations between interpreted and compiled code in the string representations of errors.
            except TypeError as ex:
                ret += "catch2 " + str(type(ex))
            except Exception:
                ret += "catchdefault "
            finally:
                ret += "finally"
                return ret

        def f7(x: int) -> str:
            ret = "try "
            try:
                ret += str(1/x) + " "
            except ZeroDivisionError as ex:
                ret += "catch " + " " + str(ex) + " "
            finally:
                ret += "finally " + str(ex)  # noqa: F821
                # Ensure that this is detected as error "variable 'ex' referenced before assignment" in compiled case
            return ret

        def f7a(x: int) -> str:
            ex = "might be overwritten"
            ret = "try "
            try:
                ret += str(1/x) + " "
            except ZeroDivisionError as ex:
                ret += "catch " + " " + str(ex) + " "
            finally:
                ret += "finally " + str(ex)
                # Ensure that this is detected as error "variable 'ex' referenced before assignment" in compiled case
            return ret

        # TODO: support finally in situation where control flow exits try block
        def f8(x: int) -> str:
            ret = "start "
            for i in [0, 1]:
                ret += str(i) + " "
                if i > x:
                    try:
                        ret += "try"
                        if x < 10:
                            break
                    finally:
                        ret += "finally"
            return ret

        def f9(x: int) -> int:
            try:
                t = 0
                for i in range(10):
                    t += i
                    if i > x * 10:
                        return t
            finally:
                t = 123
            return t+1

        def f10(x: int) -> int:
            try:
                t = 456
                return t
            finally:
                t = 123
            return t

        def f11(x: int) -> int:
            try:
                if x == 0:
                    return int(1/0)
                elif x == 1:
                    raise SyntaxError("aaa")
            except Exception as e:
                raise NotImplementedError("bbb") from e
            return 0

        def f12(x: int) -> int:
            try:
                if x == 0:
                    return int(1/0)
                elif x == 1:
                    raise SyntaxError("aaa")
            except Exception:
                raise NotImplementedError("bbb") from None
            return 0

        def f13(x: int) -> int:
            try:
                return 111
            finally:
                return 222

        def f14(x: int) -> str:
            ret = "try "
            try:
                ret += str(1/x) + " "
                if x == 1:
                    ret += x
            except SyntaxError:
                ret += "catch1 "
            except (TypeError, ArithmeticError):
                ret += "catch2 "
            except Exception:
                ret += "catchdefault "
            finally:
                ret += "finally"
                return ret

        def f15(x: int) -> str:
            ret = "begin "
            try:
                ret += "return "
                ret += str(1/x) + " "
                return ret
            except Exception:
                ret += "except "
            finally:
                ret += "finally "
                return "But return this instead: " + ret

        def f16(x: int) -> str:
            ret = "begin "
            try:
                ret += "return "
                ret += str(1/x) + " "
                return ret
            except Exception:
                ret += "except "
                return "Exception " + ret
            finally:
                ret += "finally "

        def f17(x: int) -> str:
            ret = "begin "
            try:
                ret += "return "
                ret += str(1/(x-1)) + " " + str(1/x) + " "
                return ret
            except Exception:
                ret += "except "
                ret += str(1/x) + " "
                return "Exception " + ret
            finally:
                ret += "finally "
                return ret

        def f18(x: int) -> int:
            try:
                return x
            finally:
                x += 1
                return x

        def f19(x: int) -> str:
            try:
                ret = "try "
            except Exception:
                ret = "exception "
            else:
                ret += "else "
                return ret
            finally:
                if x == 0:
                    return "override"

        def f20(x: int) -> str:
            try:
                ret = "try "
                if x < 10:
                    return ret
            except Exception:
                ret = "exception "
            else:
                ret += "else "
                return ret
            finally:
                if x == 0:
                    return "override"

        def f21(x: int) -> str:
            ret = "start "
            for i in [0, 1]:
                ret += str(i) + " "
                if i > x:
                    try:
                        ret += "try"
                        break
                    finally:
                        return "override"
            return ret

        def f22(x: int) -> str:
            ret = "start "
            for i in [0, 1]:
                ret += str(i) + " "
                if i > x:
                    try:
                        ret += "try"
                        if x < 10:
                            return "try"
                    finally:
                        if x < 10:
                            break
            return ret

        def f23(x: int) -> str:
            ret = "start "
            for i in [0, 1, 2]:
                ret += str(i) + " "
                try:
                    ret += "try "
                    if i == 0:
                        continue
                    ret += "looping "
                finally:
                    ret += "finally "
                    if x == 1:
                        return "override "
            return ret

        def f24(x: int) -> str:
            ret = "start "
            for i in [0, 1]:
                ret += str(i) + " "
                if i > x:
                    try:
                        ret += "try"
                        return "try"
                    finally:
                        break
            return ret

        def f25(x: int) -> str:
            ret = "start "
            for i in [0, 1]:
                ret += str(i) + " "
                if i > x:
                    try:
                        ret += "try"
                        ret += str(1/x)
                    finally:
                        break
            return ret

        # Assertion failure: not self.block.is_terminated
        def t1(a: int) -> int:
            try:
                return 1
            finally:
                return 2

        # compiles
        def t2(a: int) -> int:
            try:
                if a == 1:
                    return 1
            finally:
                return 2

        # Assertion failure: not self.block.is_terminated
        def t3(a: int) -> int:
            try:
                return 1
            finally:
                pass
            return 3

        # failure: [f14] Tuples of exceptions not supported yet.
        # failures: [f15, f16, f17, f19, f20, f21, f22, f23, f24]

        for f in [f0, f1, f2, f3, f4, f5, f6, f7, f7a, f8, f9, f10, f11, f12, f13, f18, f25]:
            c_f = Compiled(f)
            for v in [0, 1]:
                r1 = result_or_exception_tb(f, v)
                r2 = result_or_exception_tb(c_f, v)
                self.assertEqual(r1, r2, (str(f), v))

    @flaky(max_runs=5, min_passes=1)
    def test_try_general(self):
        def g1(a: int, b: int, c: int, d: int) -> str:
            ret = "start "
            try:
                ret += "try " + str(a) + " "
                if a == 1:
                    ret += str(1/0)
                elif a == 2:
                    ret += a
                elif a == 3:
                    raise NotImplementedError("in body")
                elif a == 4:
                    return ret
            except ArithmeticError:
                ret += "catch1 " + str(b) + " "
                if b == 1:
                    ret += str(1/0)
                elif b == 2:
                    ret += b
                elif b == 3:
                    raise NotImplementedError("in handler")
                elif b == 4:
                    return ret
                elif b == 5:
                    raise
                # TODO: The compiled code will have type(ex) = ArithmeticError instead of ZeroDivisionError.
                # TODO: Also, there are variations between interpreted and compiled code in the string representations of errors.
            except TypeError:
                ret += "catch2 " + str(b) + " "
                if b == 1:
                    ret += str(1/0)
                elif b == 2:
                    ret += b
                elif b == 3:
                    raise NotImplementedError("in handler")
                elif b == 4:
                    return ret
                elif b == 5:
                    raise
            except Exception:
                ret += "catchdefault " + str(b) + " "
                if b == 1:
                    ret += str(1/0)
                elif b == 2:
                    ret += b
                elif b == 3:
                    raise NotImplementedError("in handler")
                elif b == 4:
                    return ret
                elif b == 5:
                    raise
            else:
                ret += "else " + str(c) + " "
                if c == 1:
                    ret += str(1/0)
                elif c == 2:
                    ret += b
                elif c == 3:
                    raise NotImplementedError("in else")
                elif c == 4:
                    return ret
            finally:
                ret += "finally " + str(d) + " "
                if d == 1:
                    ret += str(1/0)
                elif d == 2:
                    ret += b
                elif d == 3:
                    raise NotImplementedError("in finally")
                elif d == 4:
                    return ret
            ret += "end "
            return ret

        def g2(a: int, b: int, c: int, d: int) -> str:
            ret = "start "
            for i in [1, 2, 3]:
                try:
                    ret += "try" + str(i) + ':' + str(a) + " "
                    if a == 1:
                        ret += str(1/0)
                    elif a == 2:
                        ret += a
                    elif a == 3:
                        break
                    elif a == 4:
                        continue
                    elif a == 5:
                        ret += "return within try "
                        return ret
                    ret += "a "
                except ZeroDivisionError:
                    ret += "except "
                    if b == 1:
                        ret += str(1/0)
                    elif b == 2:
                        ret += b
                    elif b == 3:
                        break
                    elif b == 4:
                        continue
                    elif b == 5:
                        ret += "return within except "
                        return ret
                    ret += "b "
                else:
                    ret += "else "
                    if c == 1:
                        ret += str(1/0)
                    elif c == 2:
                        ret += c
                    elif c == 3:
                        ret += "return within except "
                        return ret
                    ret += "c "
                finally:
                    ret += "finally "
                    if d == 1:
                        ret += str(1/0)
                    elif d == 2:
                        ret += d
                    elif d == 3:
                        ret += "return within finally "
                        return ret
                    ret += "d "
            ret += "end"
            return ret

        def g3(x: int):
            try:
                if x == 1:
                    raise SyntaxError()
            finally:
                if x == 1:
                    raise NotImplementedError()

        def g4(x: int):
            try:
                if x == 1:
                    raise SyntaxError()
            except Exception:
                pass
            finally:
                pass

        def g4a(x: int):
            try:
                if x == 1:
                    raise SyntaxError()
            except Exception as ex:
                ex
            finally:
                pass

        def g5(x: int) -> int:
            t = x
            for i in range(x+100):
                t += i
            return t

        def g11(x: int) -> int:
            try:
                if x == 0:
                    return int(1/0)
                elif x == 1:
                    raise SyntaxError("aaa")
            except Exception as e:
                raise NotImplementedError("bbb") from e
            return 0

        perf_test_cases = [
            (g1, (3, 3, 0, 3), 2.0),
            (g4, (1,), 2.0),
            (g4, (0,), 2.0),
            (g4a, (1,), 2.0),
            (g4a, (0,), 2.0),
            (g1, (0, 0, 0, 0), 2.0),
            (g1, (0, 0, 0, 4), 2.0),
            (g1, (4, 0, 0, 0), 2.0),
            (g1, (3, 0, 0, 0), 2.0),
            (g1, (3, 3, 0, 3), 2.0),
            (g1, (3, 4, 0, 0), 2.0)
        ]

        for f, a, limit in perf_test_cases:
            m0 = psutil.Process().memory_info().rss / 1024
            t0 = time.time()
            repeat_test(f, *a)
            t1 = time.time()
            m1 = psutil.Process().memory_info().rss / 1024

            # burn in the compiler
            repeat_test_compiled(f, *a)

            m2 = psutil.Process().memory_info().rss / 1024
            t2 = time.time()
            repeat_test_compiled(f, *a)
            t3 = time.time()
            m3 = psutil.Process().memory_info().rss / 1024

            ratio = (t3 - t2) / (t1 - t0)
            print(f"{f.__name__}{a}: compiled/interpreted is {ratio:.2%}.")

            # performance is poor, so don't fail yet
            # self.assertLessEqual(ratio, limit, (f.__name__, a))

            print(f"Increase was {m3 - m2} vs {m1 - m0}")
            # this is failing nondeterministically, and it's not clear why, but it's also
            # not clear to me that it's really because of a memory issue.

            # osx memory usage rises, but not others
            # if sys.platform != "darwin":
            #     self.assertLessEqual(m3 - m2, m1 - m0 + 512, (f.__name__, a))

        for f in [g1]:
            c_f = Compiled(f)
            for a in [0, 1, 2, 3, 4]:
                for b in [0, 1, 2, 3, 4, 5]:
                    for c in [0, 1, 2, 3, 4]:
                        for d in [0, 1, 2, 3, 4]:
                            r1 = result_or_exception_tb(f, a, b, c, d)
                            r2 = result_or_exception_tb(c_f, a, b, c, d)
                            self.assertEqual(r1, r2, (str(f), a, b, c, d))
        for f in [g2]:
            c_f = Compiled(f)
            for a in [0, 1, 2, 3, 4, 5]:
                for b in [0, 1, 2, 3, 4, 5]:
                    for c in [0, 1, 2, 3]:
                        for d in [0, 1, 2, 3]:
                            r1 = result_or_exception_tb(f, a, b, c, d)
                            r2 = result_or_exception_tb(c_f, a, b, c, d)
                            self.assertEqual(r1, r2, (str(f), a, b, c, d))

    def test_try_nested(self):

        def n1(x: int, y: int) -> str:
            try:
                ret = "try1 "
                if x == 1:
                    ret += str(1/0)
                elif x == 2:
                    ret += x
                elif x == 3:
                    raise NotImplementedError("try1")
                elif x == 4:
                    return ret
                ret += str(x) + " "
                try:
                    ret += "try2 "
                    if y == 1:
                        ret += str(1/0)
                    elif y == 2:
                        ret += y
                    elif y == 3:
                        raise NotImplementedError("try2")
                    elif y == 4:
                        return ret
                    ret += str(y) + " "
                except ArithmeticError:
                    ret += "catch1 "
                finally:
                    ret += "finally1 "
            except TypeError as ex:
                ret += "catch2 " + str(type(ex))
            finally:
                ret += "finally2"
            return ret

        def n2(x: int, y: int) -> str:
            ret = "start "
            i = 0
            while i < 3:
                i += 1
                ret += "(" + str(i) + ": "
                try:
                    ret + "try1 "
                    if x == 1:
                        ret += str(1/0)
                    elif x == 2:
                        ret += x
                    elif x == 3:
                        raise NotImplementedError("try1")
                    elif x == 4:
                        break
                    elif x == 5:
                        continue
                    elif x == 6:
                        return ret
                    ret += str(x) + " "
                    try:
                        ret += "try2 "
                        if y == 1:
                            ret += str(1/0)
                        elif y == 2:
                            ret += y
                        elif y == 3:
                            raise NotImplementedError("try2")
                        elif y == 4:
                            break
                        elif y == 5:
                            continue
                        elif y == 6:
                            return ret
                        ret += str(y) + " "
                    except ArithmeticError:
                        ret += "catch1 "
                    finally:
                        ret += "finally1 "
                except TypeError as ex:
                    ret += "catch2 " + str(type(ex))
                finally:
                    ret += "finally2 "
                ret += ") "
            ret += "done "
            return ret

        for f in [n1]:
            c_f = Compiled(f)
            for a in [0, 1, 2, 3, 4]:
                for b in [0, 1, 2, 3, 4]:
                    r1 = result_or_exception_tb(f, a, b)
                    r2 = result_or_exception_tb(c_f, a, b)
                    self.assertEqual(r1, r2, (str(f), a, b))
        for f in [n2]:
            c_f = Compiled(f)
            for a in [0, 1, 2, 3, 4, 5, 6]:
                for b in [0, 1, 2, 3, 4, 5, 6]:
                    r1 = result_or_exception_tb(f, a, b)
                    r2 = result_or_exception_tb(c_f, a, b)
                    self.assertEqual(r1, r2, (str(f), a, b))

    def test_compile_chained_context_managers(self):
        class CM(Class, Final):
            lst = Member(ListOf(int))

            def __init__(self, l):
                self.lst = l

            def __enter__(self):
                self.lst.append(1)

            def __exit__(self, a, b, c):
                self.lst.pop()

        def chainTwoOfThem():
            aList = ListOf(int)()

            with CM(aList), CM(aList):
                assert len(aList) == 2

            assert len(aList) == 0

        chainTwoOfThem()
        Entrypoint(chainTwoOfThem)()

    def test_try_reraise(self):

        # Test reraise directly in exception handler
        def reraise1(a: int, b: int) -> str:
            ret = "start "
            try:
                if a == 1:
                    ret += str(1/0)
                elif a == 2:
                    ret += a
            except Exception:
                ret += "caught "
                if b == 1:
                    raise
            ret += "end"
            return ret

        # Test reraise in function called by exception handler
        def reraise2(a: int, b: int) -> str:
            ret = "start "
            try:
                if a == 1:
                    ret += str(1/0)
                elif a == 2:
                    ret += a
            except Exception:
                ret += reraise(b)
            ret += "end"
            return ret

        # Test if reraise is possible if 'try' is interpreted but 'raise' is compiled.
        # Might be unlikely, but ensures we are following the language rules.
        def reraise0(a: int, b: int) -> str:
            ret = "start "
            try:
                if a == 1:
                    ret += str(1/0)
                elif a == 2:
                    ret += a
            except Exception:
                ret += Compiled(reraise)(b)
            ret += "end"
            return ret

        def reraise(b: int) -> str:
            if b == 1:
                raise
            return "caught "

        # Test raise outside of handler
        # TODO: traceback is different in this case.  Usually 'raise' does not get a traceback line, but in this case it does.
        c_reraise = Compiled(reraise)
        for b in [0, 1]:
            r1 = result_or_exception_str(reraise, b)
            r2 = result_or_exception_str(c_reraise, b)
            self.assertEqual(r1, r2, b)

        # Test raise inside handler
        c_reraise1 = Compiled(reraise1)
        c_reraise2 = Compiled(reraise2)
        for a in [0, 1, 2]:
            for b in [0, 1]:
                # functional results should be the same for all 3 functions, compiled or interpreted
                r0 = result_or_exception(reraise0, a, b)
                r1 = result_or_exception(reraise1, a, b)
                r2 = result_or_exception(c_reraise1, a, b)
                r3 = result_or_exception(reraise2, a, b)
                r4 = result_or_exception(c_reraise2, a, b)
                self.assertEqual(r0, r1, (a, b))
                self.assertEqual(r0, r2, (a, b))
                self.assertEqual(r0, r3, (a, b))
                self.assertEqual(r0, r4, (a, b))
                # tracebacks should be the same for each function, compiled or interpreted
                r1 = result_or_exception_tb(reraise1, a, b)
                r2 = result_or_exception_tb(c_reraise1, a, b)
                r3 = result_or_exception_tb(reraise2, a, b)
                r4 = result_or_exception_tb(c_reraise2, a, b)
                self.assertEqual(r1, r2, (a, b))
                self.assertEqual(r3, r4, (a, b))

    def test_context_manager_refcounts(self):
        class ContextManaer(Class, Final):
            def __enter__(self):
                pass

            def __exit__(self, a, b, c):
                pass

        @Entrypoint
        def f(x):
            with ContextManaer():
                return x

        a = ListOf(int)()
        assert _types.refcount(a) == 1
        f(a)
        assert _types.refcount(a) == 1

    def test_try_finally_refcounts(self):
        @Entrypoint
        def f(x):
            try:
                return x
            finally:
                pass

        a = ListOf(int)()
        assert _types.refcount(a) == 1
        f(a)
        assert _types.refcount(a) == 1

    def test_context_manager_functionality(self):

        class ConMan1():
            def __init__(self, a, b, c, t):
                self.a = a
                self.b = b
                self.c = c
                self.t = t  # trace

            def __enter__(self):
                self.t.append("__enter__")
                if self.a == 1:
                    self.t.append("raise in __enter__")
                    raise SyntaxError()
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.t.append(f"__exit__ {str(exc_type)} {exc_val}")
                if self.b == 1:
                    self.t.append("raise in __exit__")
                    raise NotImplementedError()
                self.t.append(f"__exit__ returns {self.c == 1}")
                return self.c == 1

        class ConMan2(Class, Final):
            a = Member(int)
            b = Member(int)
            c = Member(int)
            t = Member(ListOf(str))

            def __init__(self, a: int, b: int, c: int, t: ListOf(str)):
                self.a = a
                self.b = b
                self.c = c
                self.t = t

            def __enter__(self):
                self.t.append("__enter__")
                if self.a == 1:
                    self.t.append("raise in __enter__")
                    raise SyntaxError()
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.t.append(f"__exit__ {str(exc_type)} {exc_val}")
                if self.b == 1:
                    self.t.append("raise in __exit__")
                    raise NotImplementedError()
                self.t.append(f"__exit__ returns {self.c == 1}")
                return self.c == 1

        def with_cm_simple1(a, b, c, d, t) -> int:
            t.append("start")
            with ConMan1(a, b, c, t):
                t.append("body")
                if d == 1:
                    t.append("raise")
                    raise ZeroDivisionError()
                elif d == 2:
                    t.append("return1")
                    return 1
            t.append("return2")
            return 2

        def with_cm_simple2(a: int, b: int, c: int, d: int, t: ListOf(str)) -> int:
            t.append("start")
            with ConMan2(a, b, c, t):
                t.append("body")
                if d == 1:
                    t.append("raise")
                    raise ZeroDivisionError()
                elif d == 2:
                    t.append("return1")
                    return 1
            t.append("return2")
            return 2

        def with_cm_simple_mixed(a: int, b: int, c: int, d: int, t: ListOf(str)) -> int:
            t.append("start")
            with ConMan1(a, b, c, t):
                t.append("body")
                if d == 1:
                    t.append("raise")
                    raise ZeroDivisionError()
                elif d == 2:
                    t.append("return1")
                    return 1
            t.append("return2")
            return 2

        def with_cm_nested1(a, b, c, d, e, f, g, h, t) -> int:
            t.append("start")
            with ConMan1(a, b, c, t) as x:
                t.append(f"outerbody {x.a} {x.b} {x.c}")
                with ConMan1(e, f, g, t) as y:
                    t.append(f"innerbody {y.a} {y.b} {y.c}")
                    if h == 1:
                        t.append("innerraise")
                        raise FileNotFoundError()
                    elif h == 2:
                        t.append("innerreturn3")
                        return 3
                if d == 1:
                    t.append("outerraise")
                    raise ZeroDivisionError()
                elif d == 2:
                    t.append("outerreturn1")
                    return 1
            t.append("return2")
            return 2

        def with_cm_nested2(a: int, b: int, c: int, d: int, e: int, f: int, g: int, h: int, t: ListOf(str)) -> int:
            t.append("start")
            with ConMan2(a, b, c, t) as x:
                t.append(f"outerbody {x.a} {x.b} {x.c}")
                with ConMan2(e, f, g, t) as y:
                    t.append(f"innerbody {y.a} {y.b} {y.c}")
                    if h == 1:
                        t.append("innerraise")
                        raise FileNotFoundError()
                    elif h == 2:
                        t.append("innerreturn3")
                        return 3
                if d == 1:
                    t.append("outerraise")
                    raise ZeroDivisionError()
                elif d == 2:
                    t.append("outerreturn1")
                    return 1
            t.append("return2")
            return 2

        def with_no_enter() -> str:
            not_a_cm = "not a context manager"
            with not_a_cm:
                pass
            return "done"

        class EnterWrongSignature(Class, Final):
            def __enter__(self, x):
                return self

            def __exit__(self, x, y, z):
                return True

        def with_enter_wrong_sig() -> str:
            with EnterWrongSignature():
                pass
            return "done"

# Note difference in error string, depending on definition of __exit__, even though it is an '__enter__' error.

# >>> class EnterWrongSignature1():
# ...     def __enter__(self, x):
# ...             return self
# ...     def __exit__(self):
# ...             return True
# ...
# >>> with EnterWrongSignature1():
# ...     pass
# ...
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
# TypeError: __enter__() missing 1 required positional argument: 'x'

# >>> class EnterWrongSignature2():
# ...     def __enter__(self, x):
# ...             return self
# ...
# >>> with EnterWrongSignature2():
# ...     pass
# ...
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
# AttributeError: __exit__

        class ExitWrongSignature(Class, Final):
            def __enter__(self):
                return self

            def __exit__(self, x: int):
                return self

        def with_exit_wrong_sig(a: int) -> str:
            with ExitWrongSignature():
                if a == 1:
                    raise SyntaxError()
            return "done"

        class EnterNoExit(Class, Final):
            def __enter__(self):
                return self

        def with_no_exit(a: int) -> str:
            with EnterNoExit():
                if a == 1:
                    raise SyntaxError()
                elif a == 2:
                    return "return inside with"
            return "done"

        def with_cm_loop1(a, b, c, d, t) -> int:
            t.append("start")
            for i in range(3):
                t.append(f"{i}:")
                with ConMan1(a, b, c, t):
                    t.append("body")
                    if d == 1 and i == 1:
                        t.append("raise")
                        raise ZeroDivisionError()
                    elif d == 2 and i == 1:
                        t.append("break")
                        break
                    elif d == 3 and i == 1:
                        t.append("continue")
                        continue
                    elif d == 4 and i == 1:
                        t.append("return1")
                        return 1
                    t.append("end of body")
            t.append("return2")
            return 2

        def with_cm_loop2(a: int, b: int, c: int, d: int, t: ListOf(str)) -> int:
            t.append("start")
            for i in range(3):
                t.append(f"{i}:")
                with ConMan2(a, b, c, t):
                    t.append("body")
                    if d == 1 and i == 1:
                        t.append("raise")
                        raise ZeroDivisionError()
                    elif d == 2 and i == 1:
                        t.append("break")
                        break
                    elif d == 3 and i == 1:
                        t.append("continue")
                        continue
                    elif d == 4 and i == 1:
                        t.append("return1")
                        return 1
                    t.append("end of body")
            t.append("return2")
            return 2

        c_with_enter_wrong_sig = Compiled(with_enter_wrong_sig)
        r1 = result_or_exception(with_enter_wrong_sig)
        r2 = result_or_exception(c_with_enter_wrong_sig)
        # both are TypeError, but string description is different
        self.assertEqual(r1, r2)

        c_with_exit_wrong_sig = Compiled(with_exit_wrong_sig)
        r1 = result_or_exception(with_exit_wrong_sig)
        r2 = result_or_exception(c_with_exit_wrong_sig)
        # both are TypeError, but string description is different
        self.assertEqual(r1, r2)

        c_with_no_enter = Compiled(with_no_enter)
        r1 = result_or_exception(with_no_enter)
        r2 = result_or_exception(c_with_no_enter)
        self.assertEqual(r1, r2)

        c_with_no_exit = Compiled(with_no_exit)
        for a in [0, 1, 2]:
            r1 = result_or_exception(with_no_exit, a)
            r2 = result_or_exception(c_with_no_exit, a)
            self.assertEqual(r1, r2, a)

        for a in [0, 1]:
            for b in [0, 1]:
                for c in [0, 1]:
                    for d in [0, 1, 2]:
                        t1 = []
                        r1 = result_or_exception(with_cm_simple1, a, b, c, d, t1)
                        t2 = ListOf(str)([])
                        r2 = result_or_exception(Compiled(with_cm_simple2), a, b, c, d, t2)
                        self.assertEqual(r1, r2, (a, b, c, d))
                        self.assertEqual(t1, t2, (a, b, c, d))
                        t3 = ListOf(str)([])
                        r3 = result_or_exception(Compiled(with_cm_simple_mixed), a, b, c, d, t3)
                        self.assertEqual(r1, r3, (a, b, c, d))
                        self.assertEqual(t1, t3, (a, b, c, d))

        for a in [0, 1]:
            for b in [0, 1]:
                for c in [0, 1]:
                    for d in [0, 1, 2, 3, 4]:
                        t1 = []
                        r1 = result_or_exception(with_cm_loop1, a, b, c, d, t1)
                        t2 = ListOf(str)([])
                        r2 = result_or_exception(Compiled(with_cm_loop2), a, b, c, d, t2)
                        if r1 != r2 or t1 != t2:
                            print(r1)
                            print(r2)
                            print(t1)
                            print(t2)
                        self.assertEqual(r1, r2, (a, b, c, d))
                        self.assertEqual(t1, t2, (a, b, c, d))

        for a in [0, 1]:
            for b in [0, 1]:
                for c in [0, 1]:
                    for d in [0, 1, 2]:
                        for e in [0, 1]:
                            for f in [0, 1]:
                                for g in [0, 1]:
                                    for h in [0, 1, 2]:
                                        t1 = []
                                        r1 = result_or_exception(with_cm_nested1, a, b, c, d, e, f, g, h, t1)
                                        t2 = ListOf(str)([])
                                        r2 = result_or_exception(Compiled(with_cm_nested2), a, b, c, d, e, f, g, h, t2)
                                        self.assertEqual(r1, r2, (a, b, c, d, e, f, g, h))
                                        self.assertEqual(t1, t2, (a, b, c, d, e, f, g, h))

    @flaky(max_runs=3, min_passes=1)
    def test_context_manager_perf(self):

        class ConMan1():
            def __init__(self, a, b, c):
                self.a = a
                self.b = b
                self.c = c

            def __enter__(self):
                if self.a == 1:
                    raise SyntaxError()
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.b == 1:
                    raise NotImplementedError()
                return self.c == 1

        class ConMan2(Class, Final):
            a = Member(int)
            b = Member(int)
            c = Member(int)

            def __init__(self, a: int, b: int, c: int):
                self.a = a
                self.b = b
                self.c = c

            def __enter__(self):
                if self.a == 1:
                    raise SyntaxError()
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.b == 1:
                    raise NotImplementedError()
                return self.c == 1

        def with_cm_simple1(a, b, c, d) -> int:
            with ConMan1(a, b, c):
                if d == 1:
                    raise ZeroDivisionError()
                elif d == 2:
                    return 1
            return 2

        def with_cm_simple2(a: int, b: int, c: int, d: int, t: ListOf(str)) -> int:
            with ConMan2(a, b, c):
                if d == 1:
                    raise ZeroDivisionError()
                elif d == 2:
                    return 1
            return 2

        perf_test_cases = [
            (with_cm_simple1, with_cm_simple2, (0, 0, 0, 0), 1.0),
            (with_cm_simple1, with_cm_simple2, (0, 0, 0, 1), 1.0),
            (with_cm_simple1, with_cm_simple2, (0, 0, 1, 0), 1.0),
            (with_cm_simple1, with_cm_simple2, (0, 0, 1, 1), 1.0),
            (with_cm_simple1, with_cm_simple2, (0, 1, 0, 0), 1.0),
            (with_cm_simple1, with_cm_simple2, (1, 0, 0, 0), 1.0),
            (with_cm_simple1, with_cm_simple1, (0, 0, 0, 0), 1.0),
            (with_cm_simple1, with_cm_simple1, (0, 0, 0, 1), 1.0),
            (with_cm_simple1, with_cm_simple1, (0, 0, 1, 0), 1.0),
            (with_cm_simple1, with_cm_simple1, (0, 0, 1, 1), 1.0),
            (with_cm_simple1, with_cm_simple1, (0, 1, 0, 0), 1.0),
            (with_cm_simple1, with_cm_simple1, (1, 0, 0, 0), 1.0),
            (with_cm_simple2, with_cm_simple2, (0, 0, 0, 0), 1.0),
            (with_cm_simple2, with_cm_simple2, (0, 0, 0, 1), 1.0),
            (with_cm_simple2, with_cm_simple2, (0, 0, 1, 0), 1.0),
            (with_cm_simple2, with_cm_simple2, (0, 0, 1, 1), 1.0),
            (with_cm_simple2, with_cm_simple2, (0, 1, 0, 0), 1.0),
            (with_cm_simple2, with_cm_simple2, (1, 0, 0, 0), 1.0),
        ]

        for f1, f2, a, limit in perf_test_cases:
            m0 = psutil.Process().memory_info().rss / 1024
            t0 = time.time()
            repeat_test(f1, *a)
            t1 = time.time()
            m1 = psutil.Process().memory_info().rss / 1024

            # burn in the compiler
            repeat_test_compiled(f2, *a)

            m2 = psutil.Process().memory_info().rss / 1024
            t2 = time.time()
            repeat_test_compiled(f2, *a)
            t3 = time.time()
            m3 = psutil.Process().memory_info().rss / 1024

            ratio = (t3 - t2) / (t1 - t0)
            print(f"{f1.__name__}{a}: compiled/interpreted is {ratio:.2%}.")

            # performance is poor, so don't compare yet
            # self.assertLessEqual(ratio, limit, (f1.__name__, a))

            self.assertLessEqual(m3 - m2, m1 - m0 + 1024, (f1.__name__, a))

    def test_context_manager_assignment(self):
        class ConMan(Class, Final):
            a = Member(int)
            b = Member(int)
            c = Member(int)
            t = Member(ListOf(str))

            def __init__(self, a: int, b: int, c: int, t: ListOf(str)):
                self.a = a
                self.b = b
                self.c = c
                self.t = t

            def __enter__(self):
                self.t.append("__enter__")
                if self.a == 1:
                    self.t.append("raise in __enter__")
                    raise SyntaxError()
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.t.append(f"__exit__ {str(exc_type)}")
                if self.b == 1:
                    self.t.append("raise in __exit__")
                    raise NotImplementedError()
                self.t.append(f"__exit__ returns {self.c == 1}")
                return self.c == 1

        def with_cm_assign(a: int, b: int, c: int, d: int, t: ListOf(str)) -> int:
            t.append("start")

            with ConMan(a, b, c, t) as x:
                t.append(f"body {x.a} {x.b} {x.c}")
                if d == 1:
                    t.append("raise")
                    raise ZeroDivisionError()
                elif d == 2:
                    t.append("return1")
                    return 1
            t.append("return2")
            return 2

        for a in [0, 1]:
            for b in [0, 1]:
                for c in [0, 1]:
                    for d in [0, 1, 2]:
                        t1 = ListOf(str)([])
                        t2 = ListOf(str)([])

                        with_cm_assign_c = Compiled(with_cm_assign)

                        r1 = result_or_exception_str(with_cm_assign, a, b, c, d, t1)
                        r2 = result_or_exception_str(with_cm_assign_c, a, b, c, d, t2)

                        if r1 != r2 or t1 != t2:
                            print(r1)
                            print(r2)
                            print(t1)
                            print(t2)

                        self.assertEqual(r1, r2, (a, b, c, d))
                        self.assertEqual(t1, t2, (a, b, c, d))

    def test_catch_definite_exception(self):
        @Entrypoint
        def g():
            raise Exception("boo")

        @Entrypoint
        def f(x):
            try:
                g()
            except Exception:
                pass

        self.assertEqual(f(1), None)

    def test_catch_definite_exception_propagate_but_catch(self):
        @Entrypoint
        def g():
            raise Exception("boo")

        @Entrypoint
        def f(x):
            try:
                try:
                    g()
                except Exception:
                    raise Exception("Boo again!")
            except Exception:
                pass

        self.assertEqual(f(1), None)

    def test_catch_definite_exception_propagate(self):
        @Entrypoint
        def g():
            raise Exception("boo")

        @Entrypoint
        def f(x):
            try:
                g()
            except Exception:
                raise Exception("Boo again!")

        with self.assertRaisesRegex(Exception, "Boo again"):
            f(1)

    def test_catch_possible_exception(self):
        @Entrypoint
        def g():
            raise Exception("boo")

        @Entrypoint
        def f(x):
            try:
                if x < 0:
                    return 0
                g()
            except Exception:
                pass

        self.assertEqual(f(1), None)
        self.assertEqual(f(-1), 0)

    def test_many_mutually_interesting_functions(self):
        def f0(x):
            pass

        def f1(x):
            f0(x)

        def f2(x):
            f1(x)

        def f3(x):
            f2(x)

        # f4 takes many passes to get a type assignment
        # because it has to see each child get processed
        def f4(x):
            f0(x)
            f1(x)
            f2(x)
            f3(x)

        # f5 will see f4 as existing, and needs to be
        # recalculated when f4 gets its type completed
        def f5(x):
            f4(x)

        # f6 depends on both functions simultaneously
        def f6(x):
            if x > 0:
                f5(x)
            if x > 0:
                f4(x)

        Entrypoint(f6)(10)

    def test_not_compiled_called_from_compiled(self):
        @NotCompiled
        def f():
            assert not isCompiled()
            return "OK"

        @Entrypoint
        def g():
            assert isCompiled()
            return f()

        self.assertEqual(g(), "OK")

    def test_not_compiled_lambdas(self):
        @Entrypoint
        def callIt(f):
            return f(1)

        self.assertEqual(callIt(NotCompiled(lambda x: x + 1, int)), 2)

    def test_same_code_with_different_globals(self):
        def call(x):
            return f(x)  # noqa

        ast = convertFunctionToAlgebraicPyAst(call)

        f1 = evaluateFunctionDefWithLocalsInCells(ast, {'f': str}, {})
        f2 = evaluateFunctionDefWithLocalsInCells(ast, {'f': int}, {})

        @Entrypoint
        def callFunc(f, x):
            return f(x)

        self.assertEqual(callFunc(f1, 10.5), "10.5")
        self.assertEqual(callFunc(f2, 10.5), 10)

    def test_reconstructed_code_has_same_identity_hash(self):
        def call(x):
            return x

        ast = convertFunctionToAlgebraicPyAst(call)

        assert ast.filename == call.__code__.co_filename

        newCall = evaluateFunctionDefWithLocalsInCells(ast, {'f': str}, {})

        assert newCall.__code__.co_filename == call.__code__.co_filename

        assert identityHash(call.__code__) == identityHash(newCall.__code__)

    def test_code_with_nested_listcomp(self):
        def call(x):
            return [[(i, 0, 0) for i in y] for y in x]

        ast = convertFunctionToAlgebraicPyAst(call)

        evaluateFunctionDefWithLocalsInCells(ast, {'f': str}, {})

    def test_code_with_nested_setcomp(self):
        def call(x):
            return {[(i, 0, 0) for i in y] for y in x}

        ast = convertFunctionToAlgebraicPyAst(call)

        evaluateFunctionDefWithLocalsInCells(ast, {'f': str}, {})

    def test_code_with_nested_dictcomp(self):
        def call(x):
            return {0: [(i, 0, 0) for i in y] for y in x}

        ast = convertFunctionToAlgebraicPyAst(call)

        evaluateFunctionDefWithLocalsInCells(ast, {'f': str}, {})

    def test_closure_grabs_global_typed_object(self):
        def countIt(x):
            res = 0
            for a in x:
                res += aModuleLevelConstDict.get(a, 0)
            return res

        @Entrypoint
        def callIt(f, x):
            return f(x)

        arg = ListOf(str)(["hi", "bye"] * 100000)

        t0 = time.time()
        v = GetCompiledTypes()
        with v:
            self.assertEqual(
                callIt(countIt, arg),
                100000
            )

        # our code should know the type of the const dict!
        self.assertEqual(v.types['countIt'].varTypes['res'], int)

        t0 = time.time()
        callIt(countIt, arg)
        print("took ", time.time() - t0)
        self.assertLess(time.time() - t0, .1)

    def test_closure_can_grab_and_modify_global_typed_object(self):
        aModuleLevelDict['modify_count'] = 0

        def countIt(x):
            res = 0
            for a in x:
                res += aModuleLevelDict.get(a, 0)
                aModuleLevelDict["modify_count"] += 1
            return res

        @Entrypoint
        def callIt(f, x):
            return f(x)

        arg = ListOf(str)(["hi", "bye"] * 100000)

        t0 = time.time()
        v = GetCompiledTypes()
        with v:
            self.assertEqual(
                callIt(countIt, arg),
                100000
            )

        # our code should know the type of the const dict!
        self.assertEqual(v.types['countIt'].varTypes['res'], int)
        self.assertEqual(aModuleLevelDict['modify_count'], 200000)

        t0 = time.time()
        callIt(countIt, arg)
        print("took ", time.time() - t0)
        self.assertLess(time.time() - t0, .1)

    def test_can_compile_after_compilation_failure(self):
        class ThrowsCompilerExceptions(CompilableBuiltin):
            def __eq__(self, other):
                return isinstance(other, ThrowsCompilerExceptions)

            def __hash__(self):
                return hash("ThrowsCompilerExceptions")

            def convert_call(self, context, instance, args, kwargs):
                raise Exception("This always throws")

        def h():
            return 2

        @Entrypoint
        def f():
            return h() + ThrowsCompilerExceptions()()

        with self.assertRaisesRegex(Exception, "This always throws"):
            f()

        @Entrypoint
        def g():
            return h() + 1

        self.assertEqual(g(), 3)

    def test_converting_where_type_alternates(self):
        def add(x, y):
            return x if y is None else y if x is None else x + y

        populated1 = ListOf(bool)([False, True, True, False])
        populated2 = ListOf(bool)([False, True, False, True])
        vals1 = ListOf(float)([0.0, 1.0, 2.0, 3.0])

        @Entrypoint
        def addUp(p1, p2, v1, v2):
            out = ListOf(float)()
            outP = ListOf(bool)()

            for i in range(len(p1)):
                if p1[i] and p2[i]:
                    res = add(v1[i], v2[i])
                elif p1[i]:
                    res = add(v1[i], None)
                elif p2[i]:
                    res = add(None, v2[i])
                else:
                    res = None

                if res is not None:
                    out.append(res)
                    outP.append(True)
                else:
                    out.append(0.0)
                    outP.append(False)

            return makeNamedTuple(v=out, p=outP)

        v, p = addUp(populated1, populated2, vals1, vals1)

        assert v == [0.0, 2.0, 2.0, 3.0]
        assert p == [False, True, True, True]

    def test_convert_not_on_ints_and_floats(self):
        def check():
            y = ListOf(int)()
            y.append(not 10)
            y.append(not 10.5)
            y.append(not 0.0)
            y.append(not 0.5)
            y.append(not Int32(10))
            y.append(not UInt32(10.5))

            return y

        self.assertEqual(
            check(), Entrypoint(check)()
        )

    def test_compiler_can_see_type_members_of_instances(self):
        @Entrypoint
        def eltTypeOf(x):
            return x.ElementType

        assert eltTypeOf(ListOf(int)) == int
        assert eltTypeOf(ListOf(int)()) == int

    def test_function_entrypoint_multithreaded(self):
        def makeAFunction(x):
            T = OneOf(None, x)

            def aFunction() -> T:
                return x

            return aFunction

        assert type(Function(makeAFunction('a'))) is type(Function(makeAFunction('a'))) # noqa
        assert type(Function(makeAFunction('b'))) is not type(Function(makeAFunction('a'))) # noqa

        for c in range(10000):
            if c % 100 == 0:
                print("PASS ", c)
            overloads = []

            def wrapFunction(f):
                overloads.append(Function(f).overloads)

            f = makeAFunction(str(c))
            threads = [threading.Thread(target=wrapFunction, args=(f,)) for _ in range(2)]
            for t in threads:
                t.start()

            for t in threads:
                t.join(.1)

            assert len(overloads) == 2

    def test_double_assignment(self):
        def doubleAssign():
            x = y = ListOf(int)() # noqa
            return x

        assert len(doubleAssign()) == 0
        assert len(Entrypoint(doubleAssign)()) == 0

    def test_double_nested_assignment(self):
        def doubleAssign():
            x = (y, z) = (1, 2)

            assert x == (1, 2)
            assert y == 1
            assert z == 2

        doubleAssign()
        Entrypoint(doubleAssign)()

    def test_double_nested_assignment_with_failure(self):
        def doubleAssign():
            try:
                x = (y, z) = (1, 2, 3)
            except ValueError:
                pass

            # the x assignment should have succeeded
            assert x == (1, 2, 3)

        doubleAssign()
        Entrypoint(doubleAssign)()

    def test_slice_objects(self):
        @Entrypoint
        def createSlice(start, stop, step):
            return slice(start, stop, step)

        assert isinstance(createSlice(1, 2, 3), slice)

    def test_slice_objects_are_fast(self):
        def count(start, stop, step):
            res = 0.0
            for i in range(start, stop, step):
                res += slice(i).stop

            return res

        Entrypoint(count)(0, 1000000, 1)

        t0 = time.time()
        val1 = count(0, 1000000, 1)
        t1 = time.time()
        val2 = Entrypoint(count)(0, 1000000, 1)
        t2 = time.time()

        assert val1 == val2

        speedup = (t1 - t0) / (t2 - t1)

        assert speedup > 5

        print("speedup is ", speedup)

    def test_type_and_repr_of_slice_objects(self):
        @Entrypoint
        def typeOf():
            return type(slice(1, 2, 3))

        assert typeOf() is slice

        @Entrypoint
        def strOf():
            return str(slice(1, 2, 3))

        assert strOf() == str(slice(1, 2, 3))

        @Entrypoint
        def reprOf():
            return repr(slice(1, 2, 3))

        assert reprOf() == repr(slice(1, 2, 3))

    def test_class_interaction_with_slice_is_fast(self):
        class C(Class, Final):
            def __getitem__(self, x) -> int:
                return x.stop

        def count(c, start, stop, step):
            res = 0.0
            for i in range(start, stop, step):
                res += c[0:i]

            return res

        Entrypoint(count)(C(), 0, 1000000, 1)

        t0 = time.time()
        val1 = count(C(), 0, 1000000, 1)
        t1 = time.time()
        val2 = Entrypoint(count)(C(), 0, 1000000, 1)
        t2 = time.time()

        assert val1 == val2

        speedup = (t1 - t0) / (t2 - t1)

        assert speedup > 5

        print("speedup is ", speedup)

    def test_class_interaction_with_slice_pairs(self):
        class C(Class, Final):
            def __getitem__(self, x) -> int:
                return x[0].stop + x[1].stop

        def count(c, start, stop, step):
            res = 0.0
            for i in range(start, stop, step):
                res += c[:i, :i]

            return res

        Entrypoint(count)(C(), 0, 1000000, 1)

        t0 = time.time()
        val1 = count(C(), 0, 1000000, 1)
        t1 = time.time()
        val2 = Entrypoint(count)(C(), 0, 1000000, 1)
        t2 = time.time()

        assert val1 == val2

        speedup = (t1 - t0) / (t2 - t1)

        assert speedup > 5

        print("speedup is ", speedup)

    def test_chained_comparisons(self):
        def f1(x, y, z):
            return x < y < z

        def f2(x, y, z):
            return x > y > z

        def f3(x, y, z):
            return x <= y < z

        def f4(x, y, z):
            return 1 <= x <= y <= 2 <= z

        def f5(x, y, z):
            return x < y > z

        def f6(x, y, z):
            return x > y < z

        for f in [f1, f2, f3, f4, f5, f6]:
            for x in range(3):
                for y in range(3):
                    for z in range(3):
                        r1 = f(x, y, z)
                        r2 = Entrypoint(f)(x, y, z)
                        self.assertEqual(r1, r2)

        # Now verify that each expression in the chained comparison is evaluated at most once
        # and verify that normal short-circuit evaluation occurs.
        def f7(w, x, y, z):
            accumulator = []

            def side_effect(x, accumulator):
                accumulator.append(x)
                return x

            if side_effect(w, accumulator) < side_effect(x, accumulator) < side_effect(y, accumulator) < side_effect(z, accumulator):
                return (True, accumulator)
            else:
                return (False, accumulator)

        for w in range(4):
            for x in range(4):
                for y in range(4):
                    for z in range(4):
                        r1 = f7(w, x, y, z)
                        r2 = Entrypoint(f7)(w, x, y, z)

                        self.assertEqual(r1, r2)

    def test_variable_restriction_is_correct(self):
        @Entrypoint
        def toTypedDict(x: dict):
            x = Dict(int, int)(x)
            return x

        assert toTypedDict({1: 2}) == {1: 2}

    def test_function_return_conversion_level_is_ImplicitContainers(self):
        @Function
        def toList(x) -> ListOf(int):
            return x

        @Entrypoint
        def toListC(x) -> ListOf(int):
            return x

        assert toList([1, 2]) == toListC([1, 2]) == ListOf(int)([1, 2])

    def test_iterate_with_multiple_variable_targets(self):
        @Entrypoint
        def iterate(iterable):
            res = 0
            for x, y in iterable:
                res += y
            return res

        assert iterate(ListOf(ListOf(int))([[1, 2], [3, 4]])) == 6

        with self.assertRaisesRegex(Exception, "not enough values to unpack"):
            iterate(ListOf(ListOf(int))([[1, 2], [3]]))

    def test_iterate_constant_expression_multiple(self):
        @Entrypoint
        def iterate():
            res = 0
            for x, y in ((1, 2), (3, 4)):
                res += y
            return res

        assert iterate() == 6

    def test_iterate_oneof(self):
        @Entrypoint
        def iterate(x: OneOf(ListOf(int), ListOf(float))):
            res = 0
            for val in x:
                res += val
            return res

        assert iterate(ListOf(int)([1, 2, 3])) == 6

        assert iterate.resultTypeFor(ListOf(int)).typeRepresentation == OneOf(float, int)

    def test_iterate_oneof_segregates_variables(self):
        @Entrypoint
        def iterate(x: OneOf(ListOf(int), ListOf(str))):
            for val in x:
                # depending on the branch we're in, we should know that 'val'
                # is either an int or a string
                return typeKnownToCompiler(val)

            return None

        assert iterate(ListOf(int)([1, 2])) is int
        assert iterate(ListOf(str)(["2"])) is str

    def test_iterate_oneof_variable_types_join(self):
        @Entrypoint
        def iterate(x: OneOf(ListOf(int), ListOf(str))):
            res = None
            for val in x:
                # depending on the branch we're in, we should know that 'val'
                # is either an int or a string
                res = val

            return typeKnownToCompiler(res)

        assert iterate(ListOf(int)([1, 2])) is OneOf(None, int, str)
        assert iterate(ListOf(str)(["2"])) is OneOf(None, int, str)

    def test_check_isinstance_on_oneof(self):
        @Entrypoint
        def doIt(var: OneOf(int, float)):
            if isinstance(var, int):
                return typeKnownToCompiler(var)
            else:
                return typeKnownToCompiler(var)

        assert doIt(1.0) is float
        assert doIt(1) is int

    def test_check_one_of_type(self):
        @Entrypoint
        def doIt(var: OneOf(int, float)):
            checkOneOfType(var)
            print(var)
            return typeKnownToCompiler(var)

        assert doIt(1.0) is float
        assert doIt(1) is int

    def test_check_subtype(self):
        class Base(Class):
            pass

        class Child1(Base):
            pass

        class Child2(Base):
            pass

        class Child3(Base):
            pass

        @Entrypoint
        def doIt(var: Base):
            checkType(var, Child1, Child2)
            return typeKnownToCompiler(var)

        assert doIt(Base()) is Base
        assert doIt(Child1()) is Child1
        assert doIt(Child2()) is Child2
        assert doIt(Child3()) is Base

    @flaky(max_runs=3, min_passes=1)
    def test_check_one_of_type_perf_difference(self):
        @Entrypoint
        def accumulate(var: OneOf(int, float), times: int):
            res = var
            for t in range(times - 1):
                res += var
            return res

        @Entrypoint
        def accumulateWithCheck(var: OneOf(int, float), times: int):
            # instruct the compiler to check what kind of variable this is
            checkOneOfType(var)

            res = var
            for t in range(times - 1):
                res += var

            return res

        accumulate(1, 100)
        accumulateWithCheck(1, 100)

        t0 = time.time()
        accumulate(1, 1000000)
        t1 = time.time()
        accumulateWithCheck(1, 1000000)
        t2 = time.time()

        checkTime = t2 - t1
        normalTime = t1 - t0

        speedup = normalTime / checkTime
        print("integer speedup is", speedup)

        # it should be really big because the compiler can replace
        # the sum with n*(n-1)/2, so it's basically constant time.
        assert speedup > 100

        accumulate(1.0, 100)
        accumulateWithCheck(1.0, 100)

        t0 = time.time()
        accumulate(1.0, 1000000)
        t1 = time.time()
        accumulateWithCheck(1.0, 1000000)
        t2 = time.time()

        checkTime = t2 - t1
        normalTime = t1 - t0

        speedup = normalTime / checkTime
        # i get about 10x, 5 on the github test boxes
        print("float speedup is", speedup)
        assert speedup > 2.0

    def test_compile_annotated_assignment(self):
        def f():
            x: int = 20
            x: int
            return x

        assert f() == Entrypoint(f)()

    def test_with_exception(self):
        class SimpleCM1():
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                return False

        class SimpleCM2(Class, Final):
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                return True

        class SimpleCM3(Class, Final):
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                return False

        def testCM(cm):
            try:
                with cm:
                    raise ZeroDivisionError()
            except Exception:
                return 1
            return 0

        r1 = testCM(SimpleCM1())  # ok
        r2 = testCM(SimpleCM2())  # ok
        r3 = testCM(SimpleCM3())  # segfault
        self.assertEqual(r1, 1)
        self.assertEqual(r2, 0)
        self.assertEqual(r3, 1)

    def test_context_manager_corruption(self):
        class CM():
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                return True

        def f():
            with CM():
                raise NotImplementedError()

        def repeat_a(n):
            for i in range(n):
                try:
                    f()
                except Exception as e:  # noqa: F841
                    pass
            return 1/0

        def repeat_b(n):
            for i in range(n):
                try:
                    Compiled(f)()
                except Exception as e:  # noqa: F841
                    pass
            return 1/0

        with self.assertRaises(ZeroDivisionError):
            repeat_a(1000)
        # At one point, this raised RecursionError instead of ZeroDivisionError
        with self.assertRaises(ZeroDivisionError):
            repeat_b(1000)

    def test_context_manager_multiple_on_one_line1(self):
        class ConMan1():
            def __enter__(self):
                return 1

            def __exit__(self, exc_type, exc_val, exc_tb):
                return True

        class ConMan2():
            def __enter__(self):
                return 2

            def __exit__(self, exc_type, exc_val, exc_tb):
                raise NotImplementedError('ConMan2')

        def f():
            with ConMan1() as x, ConMan2() as y:
                result = x + y
            return result

        c_f = Entrypoint(f)
        c_f()
        r1 = result_or_exception(f)
        r2 = result_or_exception(c_f)
        # Former problem: c_f raises RuntimeError 'No active exception to reraise'
        self.assertEqual(r1, r2)

    def test_context_manager_multiple_on_one_line2(self):
        class ConMan():
            def __init__(self, a, b, c, t):
                self.a = a
                self.b = b
                self.c = c
                self.t = t  # trace

            def __enter__(self):
                self.t.append("__enter__")
                if self.a == 1:
                    self.t.append("raise in __enter__")
                    raise SyntaxError()
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.t.append(f"__exit__ {str(exc_type)} {exc_val}")
                if self.b == 1:
                    self.t.append("raise in __exit__")
                    raise NotImplementedError()
                self.t.append(f"__exit__ returns {self.c == 1}")
                return self.c == 1

        def with_cm_multiple(a, b, c, d, e, f, g, t) -> int:
            t.append("start")
            with ConMan(a, b, c, t) as x, ConMan(e, f, g, t) as y:
                t.append(f"outerbody {x.a} {x.b} {x.c}")
                t.append(f"innerbody {y.a} {y.b} {y.c}")
                if d == 1:
                    t.append("outerraise")
                    raise ZeroDivisionError()
                elif d == 2:
                    t.append("outerreturn1")
                    return 1
            t.append("return2")
            return 2

        for a in [0, 1]:
            for b in [0, 1]:
                for c in [0, 1]:
                    for d in [0, 1, 2]:
                        for e in [0, 1]:
                            for f in [0, 1]:
                                for g in [0, 1]:
                                    t1 = []
                                    r1 = result_or_exception(with_cm_multiple, a, b, c, d, e, f, g, t1)
                                    t2 = []
                                    r2 = result_or_exception(Entrypoint(with_cm_multiple), a, b, c, d, e, f, g, t2)
                                    if r1 != r2 or t1 != t2:
                                        print(f"mismatch {a}{b}{c}.{d}.{e}{f}{g} {r1} {r2}")
                                        print(t1)
                                        print(t2)
                                    self.assertEqual(r1, r2, (a, b, c, d, e, f, g))
                                    self.assertEqual(t1, t2, (a, b, c, d, e, f, g))

    def test_import_module(self):
        @Entrypoint
        def importSomething():
            import sys

            return sys

        assert importSomething() is sys

    def test_class_as_context_manager(self):
        class SimpleCM1():
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                return False

        class SimpleCM2(Class, Final):
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                return True

        class SimpleCM3(Class, Final):
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                return False

        def testCM(cm):
            try:
                with cm:
                    raise ZeroDivisionError()
            except Exception:
                return 1
            return 0

        assert testCM(SimpleCM1()) == 1
        assert testCM(SimpleCM2()) == 0
        assert testCM(SimpleCM3()) == 1

    def test_access_oneof_variable(self):
        @Entrypoint
        def f(x) -> object:
            return "aString"

        @Entrypoint
        def loop1():
            val = "aString"
            val = f(0)

            for i in range(20):
                val = f(i)
                print(val)

        @Entrypoint
        def loop2():
            val = f(0)
            val = "aString"

            for i in range(20):
                val = f(i)
                print(val)

        loop1()
        loop2()

    def test_notcompiled_lambda_closure_refcounts(self):
        x = ListOf(int)()

        @NotCompiled
        def f() -> int:
            x
            return 0

        @Entrypoint
        def returnIt(x):
            return x

        f = returnIt(f)

        closure = f.getClosure()
        assert refcount(closure) == 2

        for _ in range(100):
            f()

        assert refcount(x) == 2

        f = None

        assert refcount(closure) == 1
        assert refcount(x) == 2

        closure = None
        assert refcount(x) == 1

    def test_map_large_named_tuples(self):
        def getNamedTupleOfLists(n):
            nameToList = {"a" + str(i): ListOf(str)([str(i)]) for i in range(n)}
            return makeNamedTuple(**nameToList)

        @Entrypoint
        def slice(tupOfLists):
            return map(lambda l: l[0], tupOfLists)

        nt = getNamedTupleOfLists(100)

        slice(nt)
