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
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or imp lied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from typed_python import (
    Class, Member, Function, Tuple, TupleOf, ListOf,
    ConstDict, Dict, NamedTuple, Set,
    Alternative, OneOf,
    Int8, Int16, Int32,
    UInt8, UInt16, UInt32, UInt64,
    Float32, Final,
    PointerTo, Compiled, NotCompiled
)
import threading
import unittest
import psutil
import time
from typed_python.compiler.python_object_representation import typedPythonTypeToTypeWrapper
from typed_python._types import refcount
from typed_python import Entrypoint


def globalFun():
    pass


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
        self.assertIs(typedPythonTypeToTypeWrapper(object).typeRepresentation, object)

    def test_can_pass_object_in_and_out(self):
        @Compiled
        def f(x: object):
            return x

        for thing in [0, 10, f, str]:
            self.assertIs(f(thing), thing)

    def test_can_assign(self):
        @Compiled
        def f(x: object):
            y = x
            return y

        for thing in [0, 10, f, str, HoldsAnA(10)]:
            for i in range(10000):
                self.assertIs(f(thing), thing)

    def test_member_access(self):
        @Compiled
        def f(x: object):
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
        def f(x: object):
            return x[10]

        self.assertEqual(f({10: "hi"}), "hi")

        with self.assertRaisesRegex(Exception, "string index out of range"):
            f("a")

    def test_delitem(self):
        @Compiled
        def f(x: object, item: object):
            del x[item]

        d = {1: 2, 3: 4}
        f(d, 1)

        self.assertEqual(d, {3: 4})

        with self.assertRaisesRegex(KeyError, "1"):
            f(d, 1)

    def test_binary_ops(self):
        fcn = []

        @fcn.append
        def add(x: object, y: object):
            return x + y

        @fcn.append
        def sub(x: object, y: object):
            return x - y

        @fcn.append
        def mul(x: object, y: object):
            return x * y

        @fcn.append
        def div(x: object, y: object):
            return x / y

        @fcn.append
        def truediv(x: object, y: object):
            return x // y

        @fcn.append
        def bitAnd(x: object, y: object):
            return x & y

        @fcn.append
        def bitOr(x: object, y: object):
            return x | y

        @fcn.append
        def bitXor(x: object, y: object):
            return x ^ y

        @fcn.append
        def pow(x: object, y: object):
            return x ** y

        @fcn.append
        def mod(x: object, y: object):
            return x % y

        for f in fcn:
            compiled = Compiled(f)

            self.assertEqual(compiled(1, 2), f(1, 2))

            with self.assertRaises(Exception):
                compiled(2.5, "hi")

    def test_unary_ops(self):
        fcn = []

        @fcn.append
        def pos(x: object):
            return +x

        @fcn.append
        def neg(x: object):
            return -x

        @fcn.append
        def inv(x: object):
            return ~x

        @fcn.append
        def objNot(x: object):
            return not x

        for f in fcn:
            compiled = Compiled(f)

            self.assertEqual(compiled(1), f(1))
            self.assertEqual(compiled(0), f(0))
            self.assertEqual(compiled(-1), f(-1))

    def test_setitem(self):
        @Compiled
        def f(x: object, item: object, y: object):
            x[item] = y

        d = {}
        f(d, 10, 20)
        self.assertEqual(d, {10: 20})

        with self.assertRaisesRegex(Exception, "unhashable type"):
            f({}, [], [])

    def test_call_with_args_and_kwargs(self):
        @Compiled
        def f(x: object, a: object, k: object):
            return x(a, keyword=k)

        def aFunc(*args, **kwargs):
            return (args, kwargs)

        self.assertEqual(f(aFunc, 'arg', 'the kwarg'), (('arg',), ({'keyword': 'the kwarg'})))

    def test_len(self):
        @Compiled
        def f(x: object):
            return len(x)

        self.assertEqual(f([1, 2, 3]), 3)

    def test_convert_pyobj_to_oneof_with_string(self):
        @Function
        def toObject(x: object):
            return x

        @Compiled
        def fro_and_to(x: object):
            return OneOf(str, int)(x)

        self.assertEqual(fro_and_to("ab"), "ab")

    def test_object_conversions_2(self):
        T = ListOf(int)

        @Function
        def toObject(x: object):
            return x

        @Compiled
        def to_and_fro(x: T) -> T:
            return T(toObject(toObject(x)))

        t = T([1, 2, 3])
        self.assertEqual(refcount(t), 1)
        to_and_fro(t)
        self.assertEqual(refcount(t), 1)

    def test_object_conversions1(self):
        NT1 = NamedTuple(a=int, b=float, c=str, d=str)
        NT2 = NamedTuple(s=str, t=TupleOf(int))
        cases = [
            (bool, True),
            (Int8, -128),
            (Int16, -32768),
            (Int32, -2**31),
            (int, -2**63),
            (UInt8, 127),
            (UInt16, 65535),
            (UInt32, 2**32-1),
            (UInt64, 2**64-1),
            (float, 1.2345),
            (str, "abcd"),
            (TupleOf(int), (7, 6, 5, 4, 3, 2, -1)),
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
            (OneOf(str, int), "ab"),
            (OneOf(str, int), 34),
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
            def fro_and_to(x: object):
                return toObject(T(x))

            self.assertEqual(to_and_fro(T(v)), T(v), (v, type(v), T))

            self.assertEqual(fro_and_to(v), T(v), (v, type(v), T))

            x = T(v)
            if getattr(T, '__typed_python_category__', None) in [
                "ListOf", "TupleOf", "Alternative", "ConcreteAlternative",
                "Class", "Dict", "ConstDict", "Set"
            ]:
                self.assertEqual(refcount(x), 1)
                self.assertEqual(to_and_fro(x), x)
                self.assertEqual(refcount(x), 1, T)
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
            (int, int(0)), (int, int(1)), (UInt64, UInt64(0)), (UInt64, UInt64(-1)),
            (float, 0.0), (float, 0.1),
            (Float32, 0.0), (Float32, 0.1),
            (type(None), None),
            (str, ""), (str, "0"), (str, "1"),
            (bytes, b""), (bytes, b"0"), (bytes, b"\x00"), (bytes, b"\x01"),
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
            (PointerTo(int), x0.pointerUnsafe(0)),
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
            if r1 != r2:
                print(f"bool of {x} of type {type(x)} is {r1} but compiler has {r2}")
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

    def test_iterate_object(self):
        def toList(x: object):
            res = list()
            resAp = res.append
            for i in x:
                resAp(i)
            return res

        toListCompiled = Compiled(toList)

        self.assertEqual(
            toList(range(10)), toListCompiled(range(10))
        )

        self.assertEqual(
            toList(x + 1 for x in range(10)), toListCompiled(x + 1 for x in range(10))
        )

        # check that iterating something that's not an interator throws an exception
        class HasBadIter:
            def __iter__(self):
                return 10

        for form in [toList, toListCompiled]:
            with self.assertRaisesRegex(TypeError, "iter.. returned non-iterator of type 'int'"):
                form(HasBadIter())

        def generator(x):
            for i in range(x):
                yield i

            raise Exception("Boo!")

        for form in [toList, toListCompiled]:
            with self.assertRaisesRegex(Exception, "Boo!"):
                form(generator(10))

    def test_bool_of_arbitrary(self):
        class C:
            def __bool__(self):
                return False

        @Compiled
        def f(x: object):
            if x:
                return True
            return False

        self.assertEqual(f([]), False)
        self.assertEqual(f([1]), True)
        self.assertEqual(f(C()), False)

    def test_int_of_arbitrary(self):
        class C:
            def __int__(self):
                return 123

        @Compiled
        def f(x: object):
            return int(x)

        self.assertEqual(f(10), 10)
        self.assertEqual(f(C()), 123)

    def test_float_of_arbitrary(self):
        class C:
            def __float__(self):
                return 123.5

        @Compiled
        def f(x: object):
            return float(x)

        self.assertEqual(f(10.5), 10.5)
        self.assertEqual(f(10), 10.0)
        self.assertEqual(f(C()), 123.5)

    def test_str_of_arbitrary(self):
        class C:
            def __str__(self):
                return "hihi"

        @Compiled
        def f(x: object):
            return str(x)

        self.assertEqual(f(10.5), "10.5")
        self.assertEqual(f([]), "[]")
        self.assertEqual(f(C()), "hihi")

    def test_bytes_of_arbitrary(self):
        class C:
            def __bytes__(self):
                return b"hihi"

        @Compiled
        def f(x: object):
            return bytes(x)

        self.assertEqual(f(b"10.5"), b"10.5")
        self.assertEqual(f(list(b"12")), b"12")
        self.assertEqual(f(C()), b"hihi")

    def test_exception_in_arbitrary_pyobj_conversion(self):
        class C:
            def __bool__(self):
                raise Exception("bad bool call")

            def __float__(self):
                raise Exception("bad float call")

            def __int__(self):
                raise Exception("bad int call")

            def __str__(self):
                raise Exception("bad str call")

            def __bytes__(self):
                raise Exception("bad bytes call")

        @Compiled
        def callBool(x: object):
            return bool(x)

        @Compiled
        def callInt(x: object):
            return int(x)

        @Compiled
        def callStr(x: object):
            return str(x)

        @Compiled
        def callFloat(x: object):
            return float(x)

        @Compiled
        def callBytes(x: object):
            return bytes(x)

        with self.assertRaisesRegex(Exception, "bad bool call"):
            callBool(C())

        with self.assertRaisesRegex(Exception, "bad int call"):
            callInt(C())

        with self.assertRaisesRegex(Exception, "bad str call"):
            callStr(C())

        with self.assertRaisesRegex(Exception, "bad bytes call"):
            callBytes(C())

        with self.assertRaisesRegex(Exception, "bad float call"):
            callFloat(C())

        @Entrypoint
        def callBoolTyped(x):
            return bool(x)

        @Entrypoint
        def callIntTyped(x):
            return int(x)

        @Entrypoint
        def callStrTyped(x):
            return str(x)

        @Entrypoint
        def callFloatTyped(x):
            return float(x)

        @Entrypoint
        def callBytesTyped(x):
            return bytes(x)

        with self.assertRaisesRegex(Exception, "bad bool call"):
            callBoolTyped(C())

        with self.assertRaisesRegex(Exception, "bad int call"):
            callIntTyped(C())

        with self.assertRaisesRegex(Exception, "bad str call"):
            callStrTyped(C())

        with self.assertRaisesRegex(Exception, "bad bytes call"):
            callBytesTyped(C())

        with self.assertRaisesRegex(Exception, "bad float call"):
            callFloatTyped(C())

    def test_invalid_return_value_in_arbitrary_pyobj_conversion(self):
        class C:
            def __bool__(self):
                return "not valid"

            def __float__(self):
                return "not valid"

            def __int__(self):
                return "not valid"

            def __str__(self):
                return 0

            def __bytes__(self):
                return 0

        @Compiled
        def callBool(x: object):
            return bool(x)

        @Compiled
        def callInt(x: object):
            return int(x)

        @Compiled
        def callStr(x: object):
            return str(x)

        @Compiled
        def callFloat(x: object):
            return float(x)

        @Compiled
        def callBytes(x: object):
            return bytes(x)

        with self.assertRaisesRegex(Exception, "__bool__ should return bool, returned str"):
            callBool(C())

        with self.assertRaisesRegex(Exception, "__int__ returned non-int"):
            callInt(C())

        with self.assertRaisesRegex(Exception, "__str__ returned non-string"):
            callStr(C())

        with self.assertRaisesRegex(Exception, "__bytes__ returned non-bytes"):
            callBytes(C())

        with self.assertRaisesRegex(Exception, "__float__ returned non-float"):
            callFloat(C())

    def test_check_is(self):
        @Entrypoint
        def g(x: object, y: object):
            return x is y

        self.assertEqual(g(None, None), True)

        aList = []

        self.assertEqual(g([], aList), False)
        self.assertEqual(g(aList, aList), True)

    def test_instantiate_python_class(self):
        class C():
            pass

        @Entrypoint
        def makeAC():
            return C()

        self.assertTrue(isinstance(makeAC(), C))

    def test_reverse_comparision_ops(self):
        @Entrypoint
        def ltOI(x: object, y: int):
            return x < y

        @Entrypoint
        def ltIO(x: int, y: object):
            return x < y

        assert ltOI(1, 2) == ltIO(1, 2)
        assert ltOI(2, 1) == ltIO(2, 1)

    def test_call_type_object_from_interpreter(self):
        @Entrypoint
        def callIt(x: object):
            return x()

        assert callIt(Dict(int, int)) == Dict(int, int)()

    def test_gil_contention(self):
        @NotCompiled
        def f(x: int) -> int:
            return x

        @Entrypoint
        def callInLoop(ct):
            y = 0
            for i in range(ct):
                y += f(i)
            return y

        t0 = time.time()
        callInLoop(100000)
        print(time.time() - t0, " to do 1mm single threaded")

        from typed_python import _types
        loop = threading.Thread(target=_types.gilReleaseThreadLoop, daemon=True)
        loop.start()

        t0 = time.time()
        t1 = threading.Thread(target=callInLoop, args=(1000000,), daemon=True)
        t2 = threading.Thread(target=callInLoop, args=(1000000,), daemon=True)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        print(time.time() - t0, " to do 2mm in two threads")

    def test_slice_object(self):
        @Entrypoint
        def sliceIt(x: object) -> str:
            return x[:1024]

        assert sliceIt("hi") == "hi"

    def test_type_of_object_in_compiled_code_accurate(self):
        @Entrypoint
        def typeOf(x: object):
            return type(x)

        assert typeOf(10) is int
        assert typeOf(object) is type

    def test_type_of_module_in_compiled_code_accurate(self):
        @Entrypoint
        def typeOf():
            return type(threading)

        assert typeOf() is type(threading)

    def test_type_of_global_fun_in_compiled_code_accurate(self):
        @Entrypoint
        def typeOf():
            return type(globalFun)

        assert typeOf() is type(globalFun)

    def test_type_of_print_in_compiled_code_accurate(self):
        @Entrypoint
        def typeOf():
            return type(print)

        assert typeOf() is type(print)

    def test_isinstance_on_objects(self):
        @Entrypoint
        def isinstanceC(o: object, t: object):
            return isinstance(o, t)

        assert isinstanceC(10, int)
        assert not isinstanceC(10, str)

    def test_isinstance_on_objects_with_known_type(self):
        @Entrypoint
        def isinstanceC(o: object, t):
            return isinstance(o, t)

        assert isinstanceC(10, int)
        assert not isinstanceC(10, str)

    def test_isinstance_on_objects_with_known_type_and_value(self):
        @Entrypoint
        def isinstanceC(o, t):
            return isinstance(o, t)

        assert isinstanceC(10, int)
        assert not isinstanceC(10, str)
