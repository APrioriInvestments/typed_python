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
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or imp lied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from typed_python import (
    Class, Member, Function, Tuple, TupleOf, ListOf, String, Bytes, ConstDict, Dict, NamedTuple, Set,
    Alternative, OneOf, NoneType, Bool,
    Int8, Int16, Int32, Int64,
    UInt8, UInt16, UInt32, UInt64,
    Float32, Float64, Final,
    PointerTo
)

import unittest
import psutil
from typed_python.compiler.runtime import Runtime
from typed_python.compiler.python_object_representation import typedPythonTypeToTypeWrapper
from typed_python._types import refcount
from typed_python import Entrypoint


def Compiled(f):
    f = Function(f)
    return Runtime.singleton().compile(f)


class HoldsAnA:
    def __init__(self, a):
        self.a = a


class AClassWithBool(Class, Final):
    x = Member(int)

    def __init__(self, i):
        self.x = i

    def __bool__(self):
        return self.x != 0


class TestPythonObjectOfTypeCompilation(unittest.TestCase):
    def test_typeWrapper_for_object(self):
        self.assertIs(typedPythonTypeToTypeWrapper(object).typeRepresentation.PyType, object)

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

    def test_set_attribute(self):
        @Compiled
        def f_int(x: object, y: int):
            x.a = y

        @Compiled
        def f(x: object, y: object):
            x.a = y

        c = HoldsAnA(0)

        f_int(c, 10)
        self.assertEqual(c.a, 10)

        f(c, "hi")
        self.assertEqual(c.a, "hi")

        with self.assertRaisesRegex(Exception, "'dict' object has no attribute 'a'"):
            f(dict(), "hi")

    def test_getitem(self):
        @Compiled
        def f(x):
            return x[10]

        self.assertEqual(f({10: "hi"}), "hi")

        with self.assertRaisesRegex(Exception, "string index out of range"):
            f("a")

    def test_delitem(self):
        @Compiled
        def f(x, item):
            del x[item]

        d = {1: 2, 3: 4}
        f(d, 1)

        self.assertEqual(d, {3: 4})

        with self.assertRaisesRegex(KeyError, "1"):
            f(d, 1)

    def test_binary_ops(self):
        fcn = [
            lambda x, y: x + y,
            lambda x, y: x - y,
            lambda x, y: x * y,
            lambda x, y: x / y,
            lambda x, y: x // y,
            lambda x, y: x & y,
            lambda x, y: x | y,
            lambda x, y: x ^ y,
            lambda x, y: x ** y,
            lambda x, y: x % y
        ]

        for f in fcn:
            compiled = Compiled(f)

            self.assertEqual(compiled(1, 2), f(1, 2))

            with self.assertRaises(Exception):
                compiled(2.5, "hi")

    def test_unary_ops(self):
        fcn = [
            lambda x: +x,
            lambda x: -x,
            lambda x: ~x,
            lambda x: not x,
        ]

        for f in fcn:
            compiled = Compiled(f)

            self.assertEqual(compiled(1), f(1))
            self.assertEqual(compiled(0), f(0))
            self.assertEqual(compiled(-1), f(-1))

    def test_setitem(self):
        @Compiled
        def f(x, item, y):
            x[item] = y

        d = {}
        f(d, 10, 20)
        self.assertEqual(d, {10: 20})

        with self.assertRaisesRegex(Exception, "unhashable type"):
            f({}, [], [])

    def test_call_with_args_and_kwargs(self):
        @Compiled
        def f(x, a, k):
            return x(a, keyword=k)

        def aFunc(*args, **kwargs):
            return (args, kwargs)

        self.assertEqual(f(aFunc, 'arg', 'the kwarg'), (('arg',), ({'keyword': 'the kwarg'})))

    def test_len(self):
        @Compiled
        def f(x):
            return len(x)

        self.assertEqual(f([1, 2, 3]), 3)

    def test_convert_pyobj_to_oneof_with_string(self):
        @Function
        def toObject(x: object):
            return x

        @Compiled
        def fro_and_to(x):
            return OneOf(String, Int64)(x)

        self.assertEqual(fro_and_to("ab"), "ab")

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
            (NT2, NT2(s="xyz", t=tuple(range(10000)))),
            (PointerTo(int), PointerTo(int)() + 4),
        ]

        for T, v in cases:
            @Function
            def toObject(x: object):
                return x

            @Compiled
            def to_and_fro(x: T) -> T:
                return T(toObject(x))

            @Compiled
            def fro_and_to(x):
                return toObject(T(x))

            if T in [Float32, Float64]:
                self.assertEqual(to_and_fro(v), T(v))
            else:
                self.assertEqual(to_and_fro(v), v)

            if T in [Float32]:
                self.assertEqual(fro_and_to(v), T(v))
            else:
                self.assertEqual(fro_and_to(v), v, (type(v), T))

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

    def test_bool_cast_and_conv(self):

        IDict = Dict(int, int)
        IConstDict = ConstDict(int, int)
        IList = ListOf(int)
        ITuple = TupleOf(int)
        ISet = Set(int)
        NamedTuple0 = NamedTuple()
        NamedTuple1 = NamedTuple(a=int)
        OneOf2 = OneOf(int, str)
        IntTuple2 = Tuple(int, int)

        x0 = ListOf(int)([1, 2])

        # Can define empty Alternative, but can't instantiate it.
        # A0 = Alternative("A0")
        A1 = Alternative("A1", a={}, b={})
        A2 = Alternative("A2", a={}, b={}, __bool__=lambda self: False)
        A3 = Alternative("A3", a={}, b={}, __bool__=lambda self: True)
        A4 = Alternative("A4", a={}, b={}, __len__=lambda self: 0)
        A5 = Alternative("A5", a={}, b={}, __len__=lambda self: 42)
        B1 = Alternative("B1", a={'s': str}, b={'i': int})
        B2 = Alternative("B2", a={'s': str}, b={'i': int}, __bool__=lambda self: False)
        B3 = Alternative("B3", a={'s': str}, b={'i': int}, __bool__=lambda self: True)
        B4 = Alternative("B4", a={'s': str}, b={'i': int}, __len__=lambda self: 0)
        B5 = Alternative("B5", a={'s': str}, b={'i': int}, __len__=lambda self: 42)
        # want to test something like A6 below, but .matches. is not compilable at the moment
        # A6 = Alternative("A6", a={}, b={}, __len__=lambda self: 0 if self.matches.a else 42)

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
            (A1.a, A1.a()),
            (A1.b, A1.b()),
            (A2.a, A2.a()),
            (A2.b, A2.b()),
            (A3.a, A3.a()),
            (A3.b, A3.b()),
            (A4.a, A4.a()),
            (A4.b, A4.b()),
            (A5.a, A5.a()),
            (A5.b, A5.b()),
            (B1.a, B1.a(s='')),
            (B1.a, B1.a(s='a')),
            (B2.a, B2.a(s='')),
            (B2.a, B2.a(s='a')),
            (B3.a, B3.a(s='')),
            (B3.a, B3.a(s='a')),
            (B4.a, B4.a(s='')),
            (B4.a, B4.a(s='a')),
            (B5.a, B5.a(s='')),
            (B5.a, B5.a(s='a')),
            (A1, A1.a()),
            (A1, A1.a()),
            (A1, A1.b()),
            (A2, A2.a()),
            (A2, A2.b()),
            (A3, A3.a()),
            (A3, A3.b()),
            (A4, A4.a()),
            (A4, A4.b()),
            (A5, A5.a()),
            (A5, A5.b()),
            (B1, B1.a(s='')),
            (B1, B1.a(s='a')),
            (B2, B2.a(s='')),
            (B2, B2.a(s='a')),
            (B3, B3.a(s='')),
            (B3, B3.a(s='a')),
            (B4, B4.a(s='')),
            (B4, B4.a(s='a')),
            (B5, B5.a(s='')),
            (PointerTo(Int64), x0.pointerUnsafe(0)),
        ]

        @Entrypoint
        def specialized_cast(x) -> bool:
            return bool(x)

        @Entrypoint
        def specialized_conv(x) -> bool:
            return True if x else False

        for T, x in test_cases:
            @Compiled
            def compiled_cast(x: T) -> bool:
                return bool(x)

            @Compiled
            def compiled_conv(x: T) -> bool:
                return True if x else False

            r1 = bool(x)
            r2 = compiled_cast(x)
            r3 = specialized_cast(x)
            r4 = compiled_conv(x)
            r5 = specialized_conv(x)
            self.assertEqual(r1, r2)
            self.assertEqual(r1, r3)
            self.assertEqual(r1, r4)
            self.assertEqual(r1, r5)

    def test_obj_to_bool(self):

        def bool_f(x: object):
            return bool(x)

        @Entrypoint
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

    def test_some_object_operations(self):
        @Compiled
        def f(x: object, y: object):
            x += y
            y += x
            x += y
            return x

        @Compiled
        def g(x: object, y: object):
            return x >= y

        self.assertEqual(f(1, 2), 8)
        self.assertEqual(g("a", "b"), False)

    def test_create_lists(self):
        @Compiled
        def f():
            return [1, 2, 3, 4]

        res = f()
        self.assertEqual(res, [1, 2, 3, 4])

    def test_create_tuples(self):
        @Compiled
        def f():
            return (1, 2, 3, 4)

        res = f()
        self.assertEqual(res, (1, 2, 3, 4))

    def test_create_set(self):
        @Compiled
        def f():
            return {1, 2, 3, 4}

        res = f()
        self.assertEqual(res, {1, 2, 3, 4})

    def test_create_dict(self):
        @Compiled
        def f():
            return {1: 2, "hi": "bye"}

        res = f()
        self.assertEqual(res, {1: 2, "hi": "bye"})
