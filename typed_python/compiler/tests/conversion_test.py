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

import time
import traceback
import unittest
from flaky import flaky

from typed_python import Function, OneOf, TupleOf, ListOf, Tuple, NamedTuple, Class, _types
from typed_python.compiler.runtime import Runtime, Entrypoint


def Compiled(f):
    f = Function(f)
    return Runtime.singleton().compile(f)


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
        r = Runtime.singleton()

        f_fast = r.compile(f)

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
        def f(x: int, y: int, z: str) -> bool:
            return x and y and z

        self.assertEqual(f(0, 1, "s"), False)
        self.assertEqual(f(1, 1, "s"), True)

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
        @Function
        def f(x):
            return x

        Runtime.singleton().compile(f)

        x = []

        self.assertIs(f(x), x)

    def test_call_other_untyped_function(self):
        def g(x):
            return x

        @Function
        def f(x):
            return g(x)

        Runtime.singleton().compile(f)

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

    def test_mutually_recursive_untyped_functions(self):
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

        g_typed = Function(g)

        Runtime.singleton().compile(g_typed, {'x': int})
        Runtime.singleton().compile(g_typed, {'x': float})

        self.assertEqual(g(10), g_typed(10))

        for input in [18, 18.0]:
            t0 = time.time()
            g(input)
            untyped_duration = time.time() - t0

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

        @Function
        def g(x: int):
            return f(x+1)

        Runtime.singleton().compile(g)
        Runtime.singleton().compile(f)

        self.assertEqual(g(10), 11)

    def test_adding_with_nones_throws(self):
        @Function
        def g():
            return None + None

        Runtime.singleton().compile(g)

        with self.assertRaisesRegex(Exception, "Can't apply op Add.. to expressions of type NoneType"):
            g()

    def test_exception_before_return_propagated(self):
        @Function
        def g():
            None+None
            return None

        Runtime.singleton().compile(g)

        with self.assertRaisesRegex(Exception, "Can't apply op Add.. to expressions of type NoneType"):
            g()

    def test_call_function_with_none(self):
        @Function
        def g(x: None):
            return None

        Runtime.singleton().compile(g)

        self.assertEqual(g(None), None)

    def test_call_other_function_with_none(self):
        def f(x):
            return x

        @Function
        def g(x: int):
            return f(None)

        Runtime.singleton().compile(g)

        self.assertEqual(g(1), None)

    def test_interleaving_nones(self):
        def f(x, y, z):
            x+z
            return y

        @Function
        def works(x: int):
            return f(x, None, x)

        @Function
        def throws(x: int):
            return f(None, None, x)

        Runtime.singleton().compile(works)
        Runtime.singleton().compile(throws)

        self.assertEqual(works(1), None)
        with self.assertRaisesRegex(Exception, "Can't apply op Add.. to expressions of type NoneType"):
            throws(1)

    def test_assign_with_none(self):
        def f(x):
            return x

        @Function
        def g(x: int):
            y = f(None)
            z = y
            return z

        Runtime.singleton().compile(g)

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
        self.assertLess((t1-t0), (t2 - t1) * 1.1)

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
        def aFunctionThatRaises(x):
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
        @Entrypoint
        def f1(x):
            return f2(x)

        def f2(x):
            return f3(x)

        def f3(x):
            return f4(x)

        def f4(x):
            raise Exception(f"X is {x}")

        try:
            f1("hihi")
        except Exception:
            trace = traceback.format_exc()
            self.assertIn("f1", trace)
            self.assertIn("f2", trace)
            self.assertIn("f3", trace)
            self.assertIn("f4", trace)

    @flaky(max_runs=3, min_passes=1)
    def test_inlining_is_fast(self):
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

        # prime the compilation
        callsF4(1)
        callsF1(1)

        t0 = time.time()
        callsF1(100000000)
        t1 = time.time()
        callsF4(100000000)
        t2 = time.time()

        callsDeeply = t1 - t0
        callsShallowly = t2 - t1
        ratio = callsDeeply / callsShallowly

        # inlining should work across invocations, regardless of order
        self.assertLessEqual(.9, ratio)
        self.assertLessEqual(ratio, 1.1)
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

        self.assertEqual(check.resultTypeFor(10), None)

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
                self.assertLessEqual(timesCalculated, 4, identity)

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
                res += i
                if i > len(x) / 2:
                    continue

            return res

        for thing in [ListOf(int)(range(10)), Tuple(int, int, int, int)((1, 2, 3, 4))]:
            self.assertEqual(testContinue(thing), Entrypoint(testContinue)(thing))
