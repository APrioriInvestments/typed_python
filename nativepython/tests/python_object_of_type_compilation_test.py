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
    Class, Member, Function, Tuple, TupleOf, ListOf, String, Bytes, ConstDict, Dict, NamedTuple, Set,
    Alternative, OneOf, NoneType, Bool,
    Int8, Int16, Int32, Int64,
    UInt8, UInt16, UInt32, UInt64,
    Float32, Float64,
)

import unittest
import psutil
from nativepython.runtime import Runtime
from typed_python._types import refcount
from nativepython import SpecializedEntrypoint


def Compiled(f):
    f = Function(f)
    return Runtime.singleton().compile(f)


class HoldsAnA:
    def __init__(self, a):
        self.a = a


class AClassWithBool(Class):
    x = Member(int)

    def __init__(self, i):
        self.x = i

    def __bool__(self):
        return self.x != 0


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
        NT1 = NamedTuple(a=int, b=float, c=str, d=str)
        NT2 = NamedTuple(s=String, t=TupleOf(int))
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
            (TupleOf(int), tuple(range(10000))),
            (OneOf(String, Int64), "ab"),
            (OneOf(String, Int64), 34),
            (NT1, NT1(a=1, b=2.3, c="c", d="d")),
            (NT2, NT2(s="xyz", t=tuple(range(10000))))
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

            self.assertTrue(finalMem < initMem + 2)

    def test_bool_conv2(self):

        IDict = Dict(int, int)
        IConstDict = ConstDict(int, int)
        IList = ListOf(int)
        ITuple = TupleOf(int)
        ISet = Set(int)
        NamedTuple0 = NamedTuple()
        NamedTuple1 = NamedTuple(a=int)
        OneOf2 = OneOf(int, str)
        IntTuple2 = Tuple(int, int)
        # A0 = Alternative("A0")
        A1 = Alternative("A1", a={'a': str})
        A2 = Alternative("A2", a={}, b={})
        A3 = Alternative("A3", a={}, b={}, __bool__=lambda self: self.Name == 'a')
        A4 = Alternative("A4", a={}, b={}, __len__=lambda self: 0 if self.Name == 'a' else 1)

        test_cases = [
            (int, 0),
            (int, 1),
            (Int8, Int8(0)), (Int8, Int8(1)), (UInt8, UInt8(0)), (UInt8, UInt8(-1)),
            (Int16, Int16(0)), (Int16, Int16(1)), (UInt16, UInt16(0)), (UInt16, UInt16(-1)),
            (Int32, Int32(0)), (Int32, Int32(1)), (UInt32, UInt32(0)), (UInt32, UInt32(-1)),
            (Int64, Int64(0)), (Int64, Int64(1)), (UInt64, UInt64(0)), (UInt64, UInt64(-1)),
            (Float64, 0.0), (Float64, 0.1),
            (Float32, 0.0), (Float32, 0.1),
            (NoneType, NoneType()),
            (String, ""), (String, "0"), (String, "1"),
            (Bytes, b""), (Bytes, b"0"), (Bytes, b"\x00"), (Bytes, b"\x01"),
            (IDict, IDict()), (IDict, IDict({0: 0})), (IDict, IDict({1: 1, 2: 4})),
            (IConstDict, IConstDict()), (IConstDict, IConstDict({0: 0})), (IConstDict, IConstDict({1: 1, 2: 4})),
            (NamedTuple0, NamedTuple0()),
            (NamedTuple1, NamedTuple1(a=0)), (NamedTuple1, NamedTuple1(a=1)),
            (OneOf2, OneOf2(0)), (OneOf2, OneOf2("")),
            (OneOf2, OneOf2(1)), (OneOf2, OneOf2("a")),
            (IntTuple2, IntTuple2((0, 0))),
            (IntTuple2, IntTuple2((1, 1))),
            (ISet, ISet([1, 2, 3])),
            (ISet, ISet([])),
            (IList, IList()), (IList, IList([0])), (IList, IList(range(1000))),
            (ITuple, ITuple()), (ITuple, ITuple((0,))), (ITuple, ITuple(range(1000))),
            (AClassWithBool, AClassWithBool(1)),
            (AClassWithBool, AClassWithBool(0)),
            (A1, A1.a(a='')),
            (A1, A1.a(a='a')),
            (A1.a, A1.a(a='')),
            (A1.a, A1.a(a='a')),
            (A2, A2.a()),
            (A2.a, A2.a()),
            (A2, A2.b()),
            (A2.b, A2.b()),
            (A3, A3.a()),
            (A3.a, A3.a()),
            (A3, A3.b()),
            (A3.b, A3.b()),
            (A4, A4.a()),
            (A4.a, A4.a()),
            (A4, A4.b()),
            (A4.b, A4.b()),
        ]

        @SpecializedEntrypoint
        def outer_specialized_bool(x) -> bool:
            return bool(x)

        for T, x in test_cases:

            @Compiled
            def compiled_bool(x: T) -> bool:
                return bool(x)

            @SpecializedEntrypoint
            def inner_specialized_bool(x) -> bool:
                return bool(x)

            r1 = bool(x)
            r2 = compiled_bool(x)
            r3 = inner_specialized_bool(x)
            # TODO: outer_specialized_bool will error unpredictably on AClassWithBool if there are many specializations prior to it
            # it is fine if it is listed among the first 20 or so entries in test_cases
            # but it fails (error or segfault) if it is near the end of the list
            # r4 = outer_specialized_bool(x)
            self.assertEqual(r1, r2)
            self.assertEqual(r1, r3)
            # self.assertEqual(r1, r4)

    def test_obj_to_bool(self):

        def bool_f(x: object):
            return bool(x)

        @SpecializedEntrypoint
        def specialized_bool(x) -> bool:
            return bool(x)

        class ClassTrue:
            def __bool__(self):
                return True

        class ClassFalse:
            def __bool__(self):
                return False

        class ClassBool:
            def __init__(self, b0):
                self.b = bool(b0)

            def __bool__(self):
                return self.b

        class ClassLen:
            def __init__(self, b0):
                self.b = b0

            def __len__(self):
                return self.b

        class ClassNoBoolOrLen:
            pass

        class TPClassTrue(Class):
            def __bool__(self):
                return True

        class TPClassFalse(Class):
            def __bool__(self):
                return False

        class TPClassBool(Class):
            b = Member(bool)

            def __init__(self, b0):
                self.b = b0

            def __bool__(self):
                return self.b

        class TPClassLen(Class):
            b = Member(int)

            def __init__(self, b0):
                self.b = b0

            def __len__(self):
                return self.b

        class TPClassNoBoolOrLen(Class):
            pass

        test_cases = [
            list(),
            [1, 2],
            ClassFalse(),
            ClassTrue(),
            ClassBool(0),
            ClassBool(1),
            ClassLen(0),
            ClassLen(1),
            ClassNoBoolOrLen(),
            TPClassFalse(),
            TPClassTrue(),
            TPClassBool(0),
            TPClassBool(1),
            TPClassLen(0),
            TPClassLen(1),
            TPClassNoBoolOrLen(),
        ]
        compiled_bool = Compiled(bool_f)
        for i, v in enumerate(test_cases):
            r1 = bool(v)
            r2 = bool_f(v)
            r3 = compiled_bool(v)
            r4 = specialized_bool(v)
            self.assertEqual(r1, r2)
            self.assertEqual(r1, r3)
            self.assertEqual(r1, r4)
