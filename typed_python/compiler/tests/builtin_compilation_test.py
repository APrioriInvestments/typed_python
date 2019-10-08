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

from typed_python import Function, ListOf, TupleOf, NamedTuple, Dict, ConstDict, OneOf, \
    Int64, Int32, Int16, Int8, UInt64, UInt32, UInt16, UInt8, Bool, Float64, Float32, \
    String, Bytes, Alternative, Set
from typed_python.compiler.runtime import Runtime
import unittest


def Compiled(f):
    f = Function(f)
    return Runtime.singleton().compile(f)


class TestBuiltinCompilation(unittest.TestCase):
    def test_builtins_on_various_types(self):
        NT1 = NamedTuple(a=int, b=float, c=str, d=str)
        NT2 = NamedTuple(s=String, t=TupleOf(int))
        Alt1 = Alternative("Alt1", X={'a': int}, Y={'b': str})
        cases = [
            # (Float64, 1.23456789), # fails with compiled str=1.2345678899999999
            # (Float64, 12.3456789), # fails with compiled str=12.345678899999999
            # (Float64, -1.23456789), # fails with compiled str=-1.2345678899999999
            # (Float64, -12.3456789), # fails with compiled str=-12.345678899999999
            (Bool, True),
            (Float64, 1.0/7.0),  # verify number of digits after decimal in string representation
            (Float64, 8.0/7.0),  # verify number of digits after decimal in string representation
            (Float64, 71.0/7.0),  # verify number of digits after decimal in string representation
            (Float64, 701.0/7.0),  # verify number of digits after decimal in string representation
            (Float64, 1.0/70.0),  # verify exp transition for small numbers
            (Float64, 1.0/700.0),  # verify exp transition for small numbers
            (Float64, 1.0/7000.0),  # verify exp transition for small numbers
            (Float64, 1.0/70000.0),  # verify exp transition for small numbers
            (Float64, 1.0),  # verify trailing zeros in string representation of float
            (Float64, 0.123456789),
            (Float64, 2**32),  # verify trailing zeros in string representation of float
            (Float64, 2**64),  # verify trailing zeros in string representation of float
            (Float64, 1.8e19),  # verify trailing zeros in string representation of float
            (Float64, 1e16),  # verify exp transition for large numbers
            (Float64, 1e16-2),  # verify exp transition for large numbers
            (Float64, 1e16+2),  # verify exp transition for large numbers
            (Float64, -1.0/7.0),  # verify number of digits after decimal in string representation
            (Float64, -8.0/7.0),  # verify number of digits after decimal in string representation
            (Float64, -71.0/7.0),  # verify number of digits after decimal in string representation
            (Float64, -701.0/7.0),  # verify number of digits after decimal in string representation
            (Float64, -1.0/70.0),  # verify exp transition for small numbers
            (Float64, -1.0/700.0),  # verify exp transition for small numbers
            (Float64, -1.0/7000.0),  # verify exp transition for small numbers
            (Float64, -1.0/70000.0),  # verify exp transition for small numbers
            (Float64, -0.123456789),
            (Float64, -1.0),  # verify trailing zeros in string representation of float
            (Float64, -2**32),  # verify trailing zeros in string representation of float
            (Float64, -2**64),  # verify trailing zeros in string representation of float
            (Float64, -1.8e19),  # verify trailing zeros in string representation of float
            (Float64, -1e16),  # verify exp transition in string representation of float
            (Float64, -1e16-2),  # verify exp transition in string representation of float
            (Float64, -1e16+2),  # verify exp transition in string representation of float
            (Alt1, Alt1.X(a=-1)),
            (Alt1, Alt1.Y(b='yes')),
            (Float32, 1.234),
            (int, 3),
            (int, -2**63),
            (Bool, True),
            (Int8, -128),
            (Int16, -32768),
            (Int32, -2**31),
            (UInt8, 127),
            (UInt16, 65535),
            (UInt32, 2**32-1),
            (UInt64, 2**64-1),
            (Float32, 1.234567),
            (Float32, 1.234),
            (String, "abcd"),
            (Bytes, b"\01\00\02\03"),
            (Set(int), [1, 3, 5, 7]),
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
            (Dict(str, int), {'y': 7, 'n': 6}),
            (ConstDict(str, int), {'y': 2, 'n': 4}),
            (TupleOf(int), tuple(range(10000))),
            (OneOf(String, Int64), "ab"),
            (OneOf(String, Int64), 34),
            (NT1, NT1(a=1, b=2.3, c="c", d="d")),
            (NT2, NT2(s="xyz", t=tuple(range(10000))))
        ]

        for T, v in cases:
            @Compiled
            def compiled_str(x: T):
                return str(x)

            @Compiled
            def compiled_format(x: T):
                return format(x)

            @Compiled
            def compiled_dir(x: T):
                return dir(x)

            r1 = str(T(v))
            r2 = compiled_str(v)
            self.assertEqual(r1, r2)

            r1 = format(T(v))
            r2 = compiled_format(v)
            self.assertEqual(r1, r2)

            r1 = dir(T(v))
            r2 = compiled_dir(v)
            self.assertEqual(r1, r2)
