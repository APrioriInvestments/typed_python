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
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or imp lied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from typed_python import (
    Function, TupleOf, ListOf, String, Dict,
    Bool,
    Int8, Int16, Int32, Int64,
    UInt8, UInt16, UInt32, UInt64,
    Float32, Float64,
)

import unittest
import psutil
from nativepython.runtime import Runtime
from typed_python._types import refcount


def Compiled(f):
    f = Function(f)
    return Runtime.singleton().compile(f)


class HoldsAnA:
    def __init__(self, a):
        self.a = a


class TestPythonObjectOfTypeCompilation(unittest.TestCase):
    def test_can_pass_object_in_and_out(self):
        @Compiled
        def f(x):
            return x

        for thing in [0, 10, f, str]:
            self.assertIs(f(thing), thing)

    def test_can_assign(self):
        @Compiled
        def f(x):
            y = x
            return y

        for thing in [0, 10, f, str, HoldsAnA(10)]:
            for i in range(10000):
                self.assertIs(f(thing), thing)

    def test_member_access(self):
        @Compiled
        def f(x):
            return x.a

        x = '1'
        self.assertIs(f(HoldsAnA(x)), x)
        with self.assertRaises(Exception):
            f(10)

    def test_object_conversions(self):
        cases = [
            (Bool, True),
            (Int8, -128),
            (Int16, -32768),
            (Int32, -2**31),
            (Int64, -2**63),
            (UInt8, 127),
            (UInt16, 65535),
            (UInt32, 2**32-1),
            (UInt64, 2**64-1),
            (Float64, 1.2345),
            (String, "abcd"),
            (TupleOf(Int64), (7, 6, 5, 4, 3, 2, -1)),
            (TupleOf(Int32), (7, 6, 5, 4, 3, 2, -2)),
            (TupleOf(Int16), (7, 6, 5, 4, 3, 2, -3)),
            (TupleOf(Int8), (7, 6, 5, 4, 3, 2, -4)),
            (TupleOf(UInt64), (7, 6, 5, 4, 3, 2, 1)),
            (TupleOf(UInt32), (7, 6, 5, 4, 3, 2, 2)),
            (TupleOf(UInt16), (7, 6, 5, 4, 3, 2, 3)),
            (TupleOf(UInt8), (7, 6, 5, 4, 3, 2, 4)),
            (TupleOf(str), ("a", "b", "c")),
            (ListOf(str), ["a", "b", "d"]),
            (Float32, 1.2345),
            (Dict(str, int), {'y': 7, 'n': 6}),
            (TupleOf(int), tuple(range(10000)))
        ]

        for T, v in cases:
            @Compiled
            def to_and_fro(x: T) -> T:
                return T(object(x))

            @Compiled
            def fro_and_to(x):
                return object(T(x))

            if T in [Float32, Float64]:
                self.assertEqual(to_and_fro(v), T(v))
            else:
                self.assertEqual(to_and_fro(v), v)
            if T in [Float32]:
                self.assertEqual(fro_and_to(v), T(v))
            else:
                self.assertEqual(fro_and_to(v), v)

            x = T(v)
            if T.__typed_python_category__ in ["ListOf", "TupleOf", "Alternative", "ConcreteAlternative",
                                               "Class", "Dict", "ConstDict", "Set"]:
                self.assertEqual(refcount(x), 1)
                self.assertEqual(to_and_fro(x), x)
                self.assertEqual(refcount(x), 1)
                self.assertEqual(fro_and_to(x), x)
                self.assertEqual(refcount(x), 1)

            initMem = psutil.Process().memory_info().rss / 1024 ** 2
            w = v
            for _ in range(10000):
                v = to_and_fro(v)
                w = fro_and_to(w)
            finalMem = psutil.Process().memory_info().rss / 1024 ** 2

            self.assertTrue(finalMem < initMem + 5)
