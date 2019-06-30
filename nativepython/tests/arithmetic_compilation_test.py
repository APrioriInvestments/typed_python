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

    def test_can_compile_all_register_types(self):
        registerTypes = [Bool, Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64, Float32, Float64]

        def equal_result(a, b):
            epsilon = float(1e-6)
            if hasattr(a, "IsFloat") and a.IsFloat and a != 0.0:
                return (b - a) / a < epsilon
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
                # testing with 1/3 exposes some problems with mod
                # return [0.0, 1.0/3.0, 1.0, 2.0, (2 - 1 / (2**23)) * 2**127, -1.0/3.0, -1.0, -2.0, -(2 - 1 / (2**23)) * 2**127]
                return [0.0, 1.0, 2.0, (2 - 1 / (2**23)) * 2**127, -1.0, -2.0, -(2 - 1 / (2**23)) * 2**127]
            elif t in [Float64]:
                # testing with 1/3 exposes some problems with mod
                # return [0.0, 1.0/3.0, 1.0, 2.0, sys.float_info.max, -1.0/3.0, -1.0, -2.0, -sys.float_info.max]
                return [0.0, 1.0, 2.0, sys.float_info.max, -1.0, -2.0, -sys.float_info.max]

        for T in registerTypes:
            if T is not Bool:
                self.assertEqual(T(1) + T(2), T(3))

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
                        bitand, bitor, bitxor, lshift, rshift
                    ]
                    # TODO: missing pow

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
                    compiledOp = Compiled(op)

                    for v1 in suitable_range(T1):
                        for v2 in suitable_range(T2):
                            comparable = True
                            try:
                                if op in [lshift, pow] and v1 != 0 and v2 > 1024:
                                    raise Exception("overflow")  # rather than trying to calculate and possibly running out of memory
                                native_result = native_op(v1, v2)
                                T_result = computeArithmeticBinaryResultType(T1, T2)
                                if T_result.IsSignedInt:
                                    if signed_overflow(T1, v1) or signed_overflow(T2, v2):
                                        comparable = False
                                    if native_result >= 2**(T_result.Bits - 1) or native_result < -2**(T_result.Bits - 1):
                                        comparable = False
                                elif T_result.IsUnsignedInt:
                                    if native_result >= 2**T_result.Bits or native_result < 0:
                                        comparable = False
                                elif T_result.IsFloat:
                                    if native_result >= 2**128 or native_result <= -2**128:
                                        comparable = False
                            except Exception:
                                native_result = "exception"

                            try:
                                if T1 is Int64 and T2 is Int64 and op in [lshift, pow] and v1 != 0 and v2 > 1024:
                                    raise Exception("overflow")
                                typed_result = op(T1(v1), T2(v2))
                            except Exception:
                                typed_result = "exception"

                            try:
                                compiled_result = compiledOp(T1(v1), T2(v2))
                            except Exception:
                                compiled_result = "exception"

                            # for debugging
                            # if type(typed_result) != type(compiled_result):
                            #     print("type mismatch", type(typed_result).__name__, type(compiled_result).__name__)
                            # if type(typed_result) == type(compiled_result):
                            #     if comparable and typed_result != compiled_result:
                            #         print("result mismatch")
                            # if comparable and not equal_result(type(compiled_result)(native_result), compiled_result):
                            #     print("mismatch")

                            self.assertEqual(
                                type(typed_result), type(compiled_result),
                                (T1, T2, type(typed_result), type(compiled_result), op.__name__)
                            )
                            if comparable:
                                self.assertEqual(typed_result, compiled_result, (T1, T2, op.__name__))
                                self.assertTrue(equal_result(type(compiled_result)(native_result), compiled_result),
                                                (T1, T2, op.__name__, native_result, compiled_result))
