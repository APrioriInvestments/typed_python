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

import operator
import sys
from math import isfinite, trunc, floor, ceil
from typed_python import (
    Function, OneOf,
    Bool,
    Int8, Int16, Int32, Int64,
    UInt8, UInt16, UInt32, UInt64,
    Float32, Float64
)
from typed_python.type_promotion import computeArithmeticBinaryResultType
from nativepython import SpecializedEntrypoint
from nativepython.runtime import Runtime
import unittest


def Compiled(f):
    f = Function(f)
    return Runtime.singleton().compile(f)


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
    def test_runtime_singleton(self):
        self.assertTrue(Runtime.singleton() is Runtime.singleton())

    def test_compile_simple(self):
        @Function
        def f(x: int) -> int:
            return x+x+x

        r = Runtime.singleton()
        r.compile(f.overloads[0])

        self.assertEqual(f(20), 60)
        self.assertEqual(f(10), 30)

    def test_in_to_out(self):
        def identity(x: In) -> Out:
            return x

        compiled_identity = Runtime.singleton().compile(identity)

        self.assertIsInstance(compiled_identity(0.5), float)
        self.assertIsInstance(compiled_identity(0.0), float)
        self.assertIsInstance(compiled_identity(1), int)

    def SKIPtest_binary_operators(self):
        r = Runtime.singleton()

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

            f_fast = r.compile(f)

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
        r = Runtime.singleton()

        f_fast = r.compile(f)

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
        @SpecializedEntrypoint
        def toString(x):
            return str(x)

        self.assertEqual(toString(UInt64(10)), "10u64")
        self.assertEqual(toString(UInt32(10)), "10u32")
        self.assertEqual(toString(UInt16(10)), "10u16")
        self.assertEqual(toString(UInt8(10)), "10u8")

    def test_can_compile_register_builtins(self):
        registerTypes = [Bool, Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64, Float32, Float64]

        def suitable_range(t):
            if t in [Bool]:
                return [Bool(0), Bool(1)]
            elif t.IsUnsignedInt:
                return [0, 5, 10, 15, 100, 150]
            elif t.IsSignedInt:
                return [0, 5, 10, 15, 100, -5, -10, -15, -100]
            elif t in [Float32, Float64]:
                return [x/2.0 for x in range(-100, 100)]

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

            ops = [f_round0, f_round1, f_round2, f_round_1, f_round_2, f_trunc, f_floor, f_ceil]
            for op in ops:
                c_op = Compiled(op)
                for v in suitable_range(T):
                    r1 = op(v)
                    r2 = c_op(v)
                    self.assertEqual(r1, r2)
                    # note that types are necessarily different sometimes,
                    # e.g. round(Float64(1), 0) returns int when interpreted,
                    # but Float64 when compiled

    def test_can_compile_register_operations(self):
        registerTypes = [Bool, Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64, Float32, Float64]

        def result_or_exception(f, *p):
            try:
                return f(*p)
            except Exception:
                return "exception"

        def equal_result(a, b):
            if (type(a) is float or (hasattr(a, "IsFloat") and a.IsFloat)):
                epsilon = float(1e-6)
                if a < 1e-32:  # these results happen to be from calculations that magnify errors
                    epsilon = float(1e-5)
                if a != 0.0:
                    return abs(float(b - a) / a) < epsilon
                else:
                    return abs(float(b - a)) < epsilon
            return a == b

        def signed_overflow(T, v):
            return T.IsUnsignedInt and (v >= 2**(T.Bits - 1) or v < -2**(T.Bits - 1))

        def suitable_range(t):
            if t in [Bool]:
                return [0, 1]
            elif t.IsUnsignedInt:
                return [0, 1, 2, (1 << (t.Bits // 4)) - 1, (1 << (t.Bits // 2)) - 1, (1 << t.Bits) - 1]
            elif t.IsSignedInt:
                return [0, 1, 2, (1 << (t.Bits // 2 - 1)) - 1, (1 << (t.Bits - 1)) - 1,
                        -1, -2, -(1 << (t.Bits // 2 - 1)), -(1 << (t.Bits - 1))]
            elif t in [Float32]:
                return [0.0, 1.0/3.0, 0.5, 1.0, 1.5, 2.0, (2 - 1 / (2**23)) * 2**127,
                        -1.0/3.0, -0.5, -1.0, -1.5, -2.0, -(2 - 1 / (2**23)) * 2**127]
            elif t in [Float64]:
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

            if T is not Bool:
                self.assertEqual(T(1) + T(2), T(3))
            if T in [Bool]:
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
                    native_result = result_or_exception(native_op, v)
                    if T.IsSignedInt:
                        if native_result >= 2**(T.Bits - 1) or native_result < -2**(T.Bits - 1):
                            comparable = False
                    elif T.IsUnsignedInt:
                        if native_result >= 2**T.Bits or native_result < 0:
                            comparable = False
                    typed_result = result_or_exception(op, T(v))
                    if typed_result == NotImplemented:
                        typed_result = "exception"
                    compiled_result = result_or_exception(compiled_op, T(v))

                    self.assertEqual(
                        type(typed_result), type(compiled_result),
                        (T, type(typed_result), type(compiled_result), op.__name__)
                    )
                    if comparable:
                        self.assertTrue(equal_result(type(compiled_result)(typed_result), compiled_result),
                                        (T, op.__name__, typed_result, compiled_result))
                        self.assertTrue(equal_result(type(compiled_result)(native_result), compiled_result),
                                        (T, op.__name__, native_result, compiled_result))

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

                if T1 in [Bool] or T2 in [Bool]:
                    suitable_ops = [bitand, bitor, bitxor]
                else:
                    suitable_ops = [
                        add, sub, mul, div, mod, floordiv,
                        less, greater, lessEq, greaterEq, neq,
                        bitand, bitor, bitxor, lshift, rshift, pow
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
                            if T_result.IsSignedInt:
                                if signed_overflow(T1, v1) or signed_overflow(T2, v2):
                                    comparable = False
                            if (op is lshift and v1 != 0 and v2 > 1024) \
                                    or (op is pow and (v1 > 1 or v1 < -1) and v2 > 1024):
                                native_result = "exception"
                            else:
                                native_result = result_or_exception(native_op, v1, v2)
                                T_result = computeArithmeticBinaryResultType(T1, T2)
                                if native_result == "exception":
                                    pass
                                elif type(native_result) is complex:
                                    native_result = "exception"
                                elif T_result.IsSignedInt:
                                    if native_result >= 2**(T_result.Bits - 1) or native_result < -2**(T_result.Bits - 1):
                                        comparable = False
                                elif T_result.IsUnsignedInt:
                                    if native_result >= 2**T_result.Bits or native_result < 0:
                                        comparable = False
                                elif T_result.IsFloat:
                                    if native_result >= 2**128 or native_result <= -2**128:
                                        comparable = False

                            native_comparable = comparable
                            if (T1 is Float32 or T2 is Float32) and op in [mod, floordiv]:
                                # can't expect Float32 mod and floordiv to match native calculations
                                # typed should still match compiled
                                native_comparable = False
                            if ((T1 is Float32 and T2 is Float64) or (T1 is Float64 and T2 is Float32)) \
                                    and op in [less, greater, lessEq, greaterEq, neq]:
                                native_comparable = False

                            if T1 is Int64 and T2 is Int64 and \
                                    ((op is lshift and v1 != 0 and v2 > 1024) or
                                     (op is pow and (v1 > 1 or v1 < -1) and v2 > 1024)):
                                typed_result = "exception"
                            else:
                                typed_result = result_or_exception(op, T1(v1), T2(v2))
                                if type(typed_result) in [float, Float32, Float64] and not isfinite(typed_result):
                                    typed_result = "exception"
                                if type(typed_result) is complex:
                                    typed_result = "exception"
                                if op is pow and type(typed_result) is int:
                                    typed_result = float(typed_result)

                            compiled_result = result_or_exception(compiled_op, T1(v1), T2(v2))
                            if type(compiled_result) in [float, Float32, Float64] and not isfinite(compiled_result):
                                compiled_result = "exception"

                            # for debugging
                            # if type(typed_result) != type(compiled_result):
                            #     print("type mismatch", type(typed_result).__name__, type(compiled_result).__name__)
                            # if type(typed_result) == type(compiled_result):
                            #     if comparable and not equal_result(typed_result, compiled_result):
                            #         print("result mismatch")
                            # try:
                            #     if native_comparable and not equal_result(type(compiled_result)(native_result), compiled_result):
                            #         print("mismatch")
                            # except Exception:
                            #     print("mismatch exception")

                            self.assertEqual(
                                type(typed_result), type(compiled_result),
                                (T1, T2, type(typed_result), type(compiled_result), op.__name__)
                            )

                            if comparable:
                                self.assertTrue(equal_result(type(compiled_result)(typed_result), compiled_result),
                                                (T1, T2, op.__name__, typed_result, compiled_result))
                            if native_comparable:
                                self.assertTrue(equal_result(type(compiled_result)(native_result), compiled_result),
                                                (T1, T2, op.__name__, native_result, compiled_result))
