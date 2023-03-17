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

import operator
import sys
from math import isfinite, trunc, floor, ceil, nan, inf
from typed_python import (
    OneOf, ListOf, Tuple, Compiled, makeNamedTuple,
    Int8, Int16, Int32, Float32,
    UInt8, UInt16, UInt32, UInt64,
)
from typed_python.type_promotion import (
    computeArithmeticBinaryResultType, isSignedInt, isUnsignedInt, bitness
)
from typed_python import Entrypoint
import unittest


In = OneOf(int, float)
Out = OneOf(int, float, bool)


def add(x: In, y: In) -> Out:
    return x+y


def sub(x: In, y: In) -> Out:
    return x-y


def mul(x: In, y: In) -> Out:
    return x*y


def div(x: In, y: In) -> Out:
    return x/y


def mod(x: In, y: In) -> Out:
    return x%y


def lshift(x: In, y: In) -> Out:
    return x << y


def rshift(x: In, y: In) -> Out:
    return x >> y


def pow(x: In, y: In) -> Out:
    return x ** y


def bitxor(x: In, y: In) -> Out:
    return x ^ y


def bitand(x: In, y: In) -> Out:
    return x&y


def bitor(x: In, y: In) -> Out:
    return x|y


def less(x: In, y: In) -> Out:
    return x < y


def greater(x: In, y: In) -> Out:
    return x > y


def lessEq(x: In, y: In) -> Out:
    return x <= y


def greaterEq(x: In, y: In) -> Out:
    return x >= y


def eq(x: In, y: In) -> Out:
    return x == y


def neq(x: In, y: In) -> Out:
    return x != y


ALL_OPERATIONS = [
    add, sub, mul, div, mod, lshift, rshift,
    pow, bitxor, bitand, bitor, less,
    greater, lessEq, greaterEq, eq, neq
]


class TestArithmeticCompilation(unittest.TestCase):
    def test_compile_simple(self):
        @Compiled
        def f(x: int) -> int:
            return x+x+x

        self.assertEqual(f(20), 60)
        self.assertEqual(f(10), 30)

    def test_in_to_out(self):
        @Compiled
        def identity(x: In) -> Out:
            return x

        self.assertIsInstance(identity(0.5), float)
        self.assertIsInstance(identity(0.0), float)
        self.assertIsInstance(identity(1), int)

    def SKIPtest_binary_operators(self):
        failures = 0
        successes = 0
        for f in ALL_OPERATIONS:
            if f in [pow]:
                lvals = range(-5, 5)
                rvals = range(5)

                lvals = list(lvals) + [x * 1 for x in lvals]
                rvals = list(rvals) + [x * 1 for x in rvals]

            else:
                lvals = list(range(-20, 20))
                rvals = lvals

                lvals = list(lvals) + [x / 3 for x in lvals]
                rvals = list(rvals) + [x / 3 for x in rvals]

            f_fast = Compiled(f)

            for val1 in lvals:
                for val2 in rvals:
                    try:
                        pyVal = f(val1, val2)
                    except Exception:
                        pyVal = "Exception"

                    try:
                        llvmVal = f_fast(val1, val2)
                    except Exception:
                        llvmVal = "Exception"

                    if type(pyVal) is not type(llvmVal) or pyVal != llvmVal:
                        print("FAILURE", f, val1, val2, pyVal, llvmVal)
                        failures += 1
                    else:
                        successes += 1

        self.assertEqual(failures, 0, successes)

    def checkFunctionOfIntegers(self, f):
        f_fast = Compiled(f)

        for i in range(100):
            self.assertEqual(f_fast(i), f(i))

    def test_assignment_of_pod(self):
        def f(x: int) -> int:
            y = x
            return y

        self.checkFunctionOfIntegers(f)

    def test_simple_loop(self):
        def f(x: int) -> int:
            y = 0
            while x > 0:
                x = x - 1
                y = y + x
            return y

        self.checkFunctionOfIntegers(f)

    def test_call_other_typed_function(self):
        def g(x: int) -> int:
            return x+1

        def f(x: int) -> int:
            return g(x+2)

        self.checkFunctionOfIntegers(f)

    def test_basic_type_conversion(self):
        def f(x: int) -> int:
            y = 1.5
            return int(y)

        self.checkFunctionOfIntegers(f)

    def test_integers_in_closures(self):
        y = 2

        def f(x: int) -> int:
            return x+y

        self.checkFunctionOfIntegers(f)

    def test_negation(self):
        @Compiled
        def negate_int(x: int):
            return -x

        @Compiled
        def negate_float(x: float):
            return -x

        self.assertEqual(negate_int(10), -10)
        self.assertEqual(negate_float(20.5), -20.5)

    def test_can_stringify_unsigned(self):
        @Entrypoint
        def toString(x):
            return str(x)

        self.assertEqual(toString(UInt64(10)), "10u64")
        self.assertEqual(toString(UInt32(10)), "10u32")
        self.assertEqual(toString(UInt16(10)), "10u16")
        self.assertEqual(toString(UInt8(10)), "10u8")

    def test_can_compile_register_builtins(self):
        registerTypes = [bool, Int8, Int16, Int32, int, UInt8, UInt16, UInt32, UInt64, Float32, float]

        def suitable_range(t):
            if t in [bool]:
                return [0, 1]
            elif isUnsignedInt(t):
                return [0, 5, 10, 15, 50, 100, 150]
            elif isSignedInt(t):
                return [0, 5, 10, 15, 50, 100, -5, -10, -15, -50, -100]
            elif t in [Float32, float]:
                return [x / 2.0 for x in range(-100, 100)]

        for T in registerTypes:
            def f_round0(x: T):
                return round(x)

            def f_round1(x: T):
                return round(x, 1)

            def f_round2(x: T):
                return round(x, 2)

            def f_round_1(x: T):
                return round(x, -1)

            def f_round_2(x: T):
                return round(x, -2)

            def f_trunc(x: T):
                return trunc(x)

            def f_floor(x: T):
                return floor(x)

            def f_ceil(x: T):
                return ceil(x)

            def f_int(x: T):
                return int(x)

            def f_float(x: T):
                return float(x)

            def f_complex(x: T):
                return complex(x)

            def f_format(x: T):
                return format(x)

            # not_tested_yet = [f_complex]
            ops = [f_int, f_float, f_format, f_round0, f_round1, f_round2, f_round_1, f_round_2, f_trunc, f_floor, f_ceil]
            for op in ops:
                c_op = Compiled(op)
                for v in suitable_range(T):
                    r1 = op(T(v))
                    r2 = c_op(T(v))
                    self.assertEqual(r1, r2)
                    # note that types are necessarily different sometimes,
                    # e.g. round(float(1), 0) returns int when interpreted,
                    # but float when compiled

    def test_can_call_types_with_no_args(self):
        @Entrypoint
        def makeEmpty(T):
            return T()

        self.assertEqual(makeEmpty(int), 0)
        self.assertEqual(makeEmpty(float), 0.0)
        self.assertEqual(makeEmpty(bool), False)
        self.assertEqual(makeEmpty(str), "")

    def test_not_on_float(self):
        @Entrypoint
        def doit(f):
            if not f:
                return "its false"
            return "its true"

        self.assertEqual(doit(1.0), "its true")
        self.assertEqual(doit(0.0), "its false")

    def test_can_compile_register_operations(self):
        failed = False

        registerTypes = [bool, Int8, Int16, Int32, int, UInt8, UInt16, UInt32, UInt64, Float32, float]

        def result_or_exception(f, *p):
            try:
                return f(*p)
            except Exception:
                return "exception"

        def equal_result(a, b):
            if type(a) in (float, Float32):
                epsilon = float(1e-6)
                if a < 1e-32:  # these results happen to be from calculations that magnify errors
                    epsilon = float(1e-5)
                if a != 0.0:
                    return abs(float(b - a) / a) < epsilon
                else:
                    return abs(float(b - a)) < epsilon
            return a == b

        def signed_overflow(T, v):
            return isUnsignedInt(T) and (v >= 2**(bitness(T) - 1) or v < -2**(bitness(T) - 1))

        def suitable_range(t):
            if t in [bool]:
                return [0, 1]
            elif isUnsignedInt(t):
                return [0, 1, 2, (1 << (bitness(t) // 4)) - 1, (1 << (bitness(t) // 2)) - 1, (1 << bitness(t)) - 1]
            elif isSignedInt(t):
                return [0, 1, 2, (1 << (bitness(t) // 2 - 1)) - 1, (1 << (bitness(t) - 1)) - 1,
                        -1, -2, -(1 << (bitness(t) // 2 - 1)), -(1 << (bitness(t) - 1))]
            elif t in [Float32]:
                return [0.0, 1.0/3.0, 0.5, 1.0, 1.5, 2.0, (2 - 1 / (2**23)) * 2**127,
                        -1.0/3.0, -0.5, -1.0, -1.5, -2.0, -(2 - 1 / (2**23)) * 2**127]
            elif t in [float]:
                return [0.0, 1e-16, 9.876e-16, 1.0/3.0, 0.5, 1.0, 1.5, 10.0/7.0, 2.0, 3.0, 10.0/3.0, 100.0/3.0, sys.float_info.max,
                        -1e-16, -9.876e-16, -1.0/3.0, -0.5, -1.0, -10.0/7.0, -1.5, -2.0, -3.0, -10.0/3.0, -100.0/3.0, -sys.float_info.max]

        for T in registerTypes:
            def not_(x: T):
                return not x

            def invert(x: T):
                return ~x

            def neg(x: T):
                return -x

            def pos(x: T):
                return +x

            def abs_(x: T):
                return abs(x)

            if T is not bool:
                self.assertEqual(T(1) + T(2), T(3))
            if T in [bool]:
                suitable_ops = [not_]
            else:
                suitable_ops = [invert, neg, pos, abs_]

            typed_to_native_op = {
                invert: operator.__inv__,
                not_: operator.__not__,
                neg: operator.__neg__,
                pos: operator.__pos__,
                abs_: operator.__abs__,
            }
            for op in suitable_ops:
                native_op = typed_to_native_op[op]
                compiled_op = Compiled(op)

                for v in suitable_range(T):
                    comparable = True
                    interpreterResult = result_or_exception(native_op, v)
                    if isSignedInt(T):
                        if interpreterResult >= 2**(bitness(T) - 1) or interpreterResult < -2**(bitness(T) - 1):
                            comparable = False
                    elif isUnsignedInt(T):
                        if interpreterResult >= 2**bitness(T) or interpreterResult < 0:
                            comparable = False
                    typedPythonResult = result_or_exception(op, T(v))
                    if typedPythonResult == NotImplemented:
                        typedPythonResult = "exception"
                    compilerResult = result_or_exception(compiled_op, T(v))

                    self.assertEqual(
                        type(typedPythonResult), type(compilerResult),
                        (T, type(typedPythonResult), type(compilerResult), op.__name__)
                    )
                    if comparable:
                        self.assertTrue(equal_result(type(compilerResult)(typedPythonResult), compilerResult),
                                        (T, op.__name__, typedPythonResult, compilerResult))
                        self.assertTrue(equal_result(type(compilerResult)(interpreterResult), compilerResult),
                                        (T, op.__name__, interpreterResult, compilerResult))

        for T1 in registerTypes:
            for T2 in registerTypes:
                def add(x: T1, y: T2):
                    return x + y

                def sub(x: T1, y: T2):
                    return x - y

                def mul(x: T1, y: T2):
                    return x * y

                def div(x: T1, y: T2):
                    return x / y

                def floordiv(x: T1, y: T2):
                    return x // y

                def mod(x: T1, y: T2):
                    return x % y

                def less(x: T1, y: T2):
                    return x < y

                def greater(x: T1, y: T2):
                    return x > y

                def lessEq(x: T1, y: T2):
                    return x <= y

                def greaterEq(x: T1, y: T2):
                    return x >= y

                def bitand(x: T1, y: T2):
                    return x & y

                def bitor(x: T1, y: T2):
                    return x | y

                def bitxor(x: T1, y: T2):
                    return x ^ y

                def neq(x: T1, y: T2):
                    return x != y

                def rshift(x: T1, y: T2):
                    return x >> y

                def lshift(x: T1, y: T2):
                    return x << y

                def pow(x: T1, y: T2):
                    return x ** y

                if T1 in [bool] or T2 in [bool]:
                    suitable_ops = [bitand, bitor, bitxor]
                else:
                    suitable_ops = [
                        add, sub, mul, div, mod, floordiv,
                        less, greater, lessEq, greaterEq, neq,
                        bitand, bitor, bitxor,
                        lshift,
                        rshift, pow
                    ]

                typed_to_native_op = {
                    sub: operator.sub,
                    add: operator.add,
                    mul: operator.mul,
                    div: operator.truediv,
                    floordiv: operator.floordiv,
                    mod: operator.mod,
                    less: operator.lt,
                    greater: operator.gt,
                    lessEq: operator.le,
                    greaterEq: operator.ge,
                    neq: operator.ne,
                    bitand: operator.and_,
                    bitor: operator.or_,
                    bitxor: operator.xor,
                    rshift: operator.rshift,
                    lshift: operator.lshift,
                    pow: operator.pow
                }

                for op in suitable_ops:
                    native_op = typed_to_native_op[op]
                    compiled_op = Compiled(op)

                    for v1 in suitable_range(T1):
                        for v2 in suitable_range(T2):
                            comparable = True
                            T_result = computeArithmeticBinaryResultType(T1, T2)
                            if isSignedInt(T_result):
                                if signed_overflow(T1, v1) or signed_overflow(T2, v2):
                                    comparable = False

                            if (op in (lshift, rshift) and v2 > 1024) \
                                    or (op is pow and (v1 > 1 or v1 < -1) and v2 > 1024):
                                interpreterResult = "expected to disagree"
                            else:
                                interpreterResult = result_or_exception(native_op, v1, v2)

                                T_result = computeArithmeticBinaryResultType(T1, T2)

                                if interpreterResult == "exception":
                                    pass
                                elif type(interpreterResult) is complex:
                                    interpreterResult = "expected to disagree"
                                elif isSignedInt(T_result):
                                    if interpreterResult >= 2**(bitness(T_result) - 1) or interpreterResult < -2**(bitness(T_result) - 1):
                                        comparable = False
                                elif isUnsignedInt(T_result):
                                    if interpreterResult >= 2**bitness(T_result) or interpreterResult < 0:
                                        comparable = False
                                elif T_result in (float, Float32):
                                    if interpreterResult >= 2**128 or interpreterResult <= -2**128:
                                        comparable = False

                            native_comparable = comparable
                            if (T1 is Float32 or T2 is Float32) and op in [mod, floordiv]:
                                # can't expect Float32 mod and floordiv to match native calculations
                                # typed should still match compiled
                                native_comparable = False
                            if ((T1 is Float32 and T2 is float) or (T1 is float and T2 is Float32)) \
                                    and op in [less, greater, lessEq, greaterEq, neq]:
                                native_comparable = False

                            if T1 is int and T2 is int and \
                                    ((op is lshift and v1 != 0 and v2 > 1024) or
                                     (op is pow and (v1 > 1 or v1 < -1) and v2 > 1024)):
                                typedPythonResult = "expected to disagree"
                            else:
                                typedPythonResult = result_or_exception(op, T1(v1), T2(v2))
                                if type(typedPythonResult) in [float, Float32, float] and not isfinite(typedPythonResult):
                                    typedPythonResult = "expected to disagree"
                                if type(typedPythonResult) is complex:
                                    typedPythonResult = "expected to disagree"
                                if op is pow and type(typedPythonResult) is int:
                                    typedPythonResult = float(typedPythonResult)

                            compilerResult = result_or_exception(compiled_op, T1(v1), T2(v2))

                            if type(compilerResult) in [float, Float32, float] and not isfinite(compilerResult):
                                compilerResult = "expected to disagree"

                            # for debugging
                            # if type(typedPythonResult) != type(compilerResult):
                            #     print("type mismatch", type(typedPythonResult).__name__, type(compilerResult).__name__)
                            # if type(typedPythonResult) == type(compilerResult):
                            #     if comparable and not equal_result(typedPythonResult, compilerResult):
                            #         print("result mismatch")
                            # try:
                            #     if native_comparable and not equal_result(type(compilerResult)(interpreterResult), compilerResult):
                            #         print("mismatch")
                            # except Exception:
                            #     print("mismatch exception")

                            if type(typedPythonResult) != type(compilerResult):
                                print(
                                    f"interpreter and compiler types for {op.__name__} on arguments of type "
                                    f"{T1} and {T2} were not the same. the interpreter gave {type(typedPythonResult)} but"
                                    f"compiler gave {type(compilerResult)}"
                                )

                                failed = True

                            if "expected to disagree" not in (typedPythonResult, compilerResult, interpreterResult):
                                if (comparable and not equal_result(type(compilerResult)(typedPythonResult), compilerResult) or
                                        native_comparable and not equal_result(type(compilerResult)(interpreterResult), compilerResult)):
                                    print(
                                        f"results for {op.__name__} on arguments "
                                        f"{v1} of type {T1} and {v2} of type {T2} do not agree: "
                                        f"interpreter gives {interpreterResult}, "
                                        f"typed_python gave {typedPythonResult}, and "
                                        f"the compiler gave {compilerResult}"
                                    )

                                    failed = True

        self.assertFalse(failed)

    def test_int_of_nan(self):
        @Entrypoint
        def f(x):
            return int(x)

        with self.assertRaisesRegex(Exception, "NaN"):
            f(nan)

        with self.assertRaisesRegex(Exception, "infinity"):
            f(inf)

    def test_mod_constant_in_tuple(self):
        @Entrypoint
        def rf(row):
            return row.x % 1.0

        self.assertEqual(rf(makeNamedTuple(x=1.0)), 0.0)

    def test_pow_with_bad_inputs(self):
        @Entrypoint
        def callPow(x, y):
            return x ** y

        with self.assertRaises(ZeroDivisionError):
            callPow(0.0, -1.0)

        with self.assertRaises(ZeroDivisionError):
            callPow(0, -1)

    def test_floordiv_with_bad_inputs(self):
        @Entrypoint
        def callFloordiv(x, y):
            return x // y

        with self.assertRaises(ZeroDivisionError):
            callFloordiv(1.0, 0.0)

        with self.assertRaises(ZeroDivisionError):
            callFloordiv(1, 0)

    def test_shift_with_bad_inputs(self):
        @Entrypoint
        def callShift(x, y):
            return x << y

        with self.assertRaises(ValueError):
            callShift(1, -2)

        with self.assertRaises(OverflowError):
            callShift(1, 2048)

    def test_formatting_with_format_strings_works(self):
        @Entrypoint
        def format(x):
            return f"{x:.6g}"

        assert format(1.0) == f"{1.0:.6g}"

    def test_formatting_strings_in_loop(self):
        @Entrypoint
        def format(a, b, c, d):
            res = ListOf(str)()
            for i in range(10000):
                res.append(f"{i} {a} {b} {c} {d}")
            return res

        format(1, 2, 3, 4)

    def test_string_joining_in_loop(self):
        @Entrypoint
        def format(a, b, c, d):
            res = ListOf(str)()
            for i in range(10000):
                l = ListOf(str)()
                l.append(str(a))
                l.append(str(b))
                l.append(str(c))
                res.append("".join(l))
            return res

        format(1, 2, 3, 4)

    def test_unary_ops_on_constants(self):
        @Entrypoint
        def sliceAtPositiveZero(x):
            return x[+0]

        self.assertIs(sliceAtPositiveZero.resultTypeFor(Tuple(int, float)).interpreterTypeRepresentation, int)

        @Entrypoint
        def sliceAtNegativeZero(x):
            return x[-0]

        self.assertIs(sliceAtNegativeZero.resultTypeFor(Tuple(int, float)).interpreterTypeRepresentation, int)

        @Entrypoint
        def sliceAtInvertZero(x):
            return x[~0]

        self.assertIs(sliceAtInvertZero.resultTypeFor(Tuple(int, float)).interpreterTypeRepresentation, float)

        @Entrypoint
        def sliceAtNotZero(x):
            return x[int(not 0)]

        self.assertIs(sliceAtNotZero.resultTypeFor(Tuple(int, float)).interpreterTypeRepresentation, float)

        @Entrypoint
        def sliceAtPositiveZeroFloat(x):
            return x[int(+0.0)]

        self.assertIs(sliceAtPositiveZeroFloat.resultTypeFor(Tuple(int, float)).interpreterTypeRepresentation, int)

        @Entrypoint
        def sliceAtNegativeZeroFloat(x):
            return x[int(-0.0)]

        self.assertIs(sliceAtNegativeZeroFloat.resultTypeFor(Tuple(int, float)).interpreterTypeRepresentation, int)

        @Entrypoint
        def sliceAtNotZeroFloat(x):
            return x[int(not 0.0)]

        self.assertIs(sliceAtNotZeroFloat.resultTypeFor(Tuple(int, float)).interpreterTypeRepresentation, float)

        @Entrypoint
        def sliceAtNotFalse(x):
            return x[int(not False)]

        self.assertIs(sliceAtNotZeroFloat.resultTypeFor(Tuple(int, float)).interpreterTypeRepresentation, float)
