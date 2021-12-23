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
import unittest
import operator
import time
import gc
import math
from typed_python.test_util import currentMemUsageMb
from typed_python._types import is_default_constructible, canConvertToTrivially
from typed_python import (
    Int16, UInt64, Float32, ListOf, TupleOf, OneOf, NamedTuple, Class, Alternative,
    ConstDict, Member, _types, Forward, Final, Function, Entrypoint, Tuple, Value
)


class DefaultVal(Class, Final):
    x0 = Member(int)
    x1 = Member(int, 5)

    y0 = Member(bool)
    y1 = Member(bool, True)

    z0 = Member(float)
    z1 = Member(float, 3.14)

    b0 = Member(bytes)
    b1 = Member(bytes, b"abc")

    s0 = Member(str)
    s1 = Member(str, "abc")


class Interior(Class, Final):
    x = Member(int)
    y = Member(int)


class Exterior(Class, Final):
    x = Member(int)
    i = Member(Interior)
    iTup = Member(NamedTuple(x=Interior, y=Interior))

    def __init__(self):
        self.i = Interior()


ClassWithInit = Forward("ClassWithInit")


@ClassWithInit.define
class ClassWithInit(Class, Final):
    x = Member(int)
    y = Member(float)
    z = Member(str)
    cwi = Member(ClassWithInit)

    def __init__(self):
        pass

    def __init__(self, x=1, cwi=None):  # noqa: F811
        self.x = x
        if cwi is not None:
            self.cwi = cwi

    def __init__(self, x):  # noqa: F811
        self.x = x


class ClassWithComplexDispatch(Class):
    x = Member(int)

    def f(self, x) -> str:
        return 'x'

    def f(self, y) -> str:  # noqa: F811
        return 'y'


class NativeClassTypesTests(unittest.TestCase):
    def test_class_name(self):
        assert ClassWithInit.__name__ == 'ClassWithInit'
        assert ClassWithInit.__module__ == 'typed_python.class_types_test'
        assert ClassWithInit.__qualname__ == 'ClassWithInit'

    def test_member_default_value(self):
        c = DefaultVal()

        self.assertEqual(c.x0, 0)
        self.assertEqual(c.x1, 5)

        self.assertEqual(c.y0, False)
        self.assertEqual(c.y1, True)

        self.assertEqual(c.z0, 0.0)
        self.assertEqual(c.z1, 3.14)

        self.assertEqual(c.b0, b"")
        self.assertEqual(c.b1, b"abc")

        self.assertEqual(c.s0, "")
        self.assertEqual(c.s1, "abc")

    def test_class_dispatch_by_name(self):
        c = ClassWithComplexDispatch(x=200)

        self.assertEqual(c.f(10), 'x')
        self.assertEqual(c.f(x=10), 'x')
        self.assertEqual(c.f(y=10), 'y')

    def test_class_with_uninitializable(self):
        c = ClassWithInit()

        c.y = 20

        c.cwi = c

        self.assertEqual(c.cwi.cwi.cwi.cwi.cwi.cwi.cwi.y, 20)

        c.cwi = ClassWithInit(10)

        self.assertEqual(c.cwi.x, 10)

    def test_implied_init_fun(self):
        self.assertEqual(Interior().x, 0)
        self.assertEqual(Interior().y, 0)

        self.assertEqual(Interior(x=10).x, 10)
        self.assertEqual(Interior(x=10).y, 0)

        self.assertEqual(Interior(y=10).x, 0)
        self.assertEqual(Interior(y=10).y, 10)

        self.assertEqual(Interior(x=20, y=10).x, 20)
        self.assertEqual(Interior(x=20, y=10).y, 10)

    def executeInLoop(self, f, duration=.25):
        memUsage = currentMemUsageMb()

        t0 = time.time()

        while time.time() - t0 < duration:
            f()

        gc.collect()
        self.assertLess(currentMemUsageMb() - memUsage, 1.0)

    def test_typed_function_call_doesnt_leak(self):
        @Function
        def f(x, y):
            return x + y

        self.executeInLoop(lambda: f(1, 2))

    def test_typed_function_call_with_star_args_doesnt_leak(self):
        @Function
        def f(*args):
            return len(args)

        self.executeInLoop(lambda: f(1, 2))

    def test_class_create_doesnt_leak(self):
        self.executeInLoop(lambda: ClassWithInit(cwi=ClassWithInit()))

    def test_class_member_access_doesnt_leak(self):
        x = ClassWithInit(cwi=ClassWithInit())
        self.executeInLoop(lambda: x.cwi.z)

    def test_class(self):
        with self.assertRaises(TypeError):
            class A0(Class):
                x = Member((1, 2, 3))

        class A(Class, Final):
            x = Member(int)

            y = int  # not a member. Just a value.

            def f(self):
                return 10

        self.assertEqual(_types.bytecount(A), 8)
        self.assertTrue(A.y is int)

        a = A()

        with self.assertRaises(AttributeError):
            a.not_an_attribute

        self.assertEqual(a.x, 0)

        a.x = 10

        self.assertEqual(a.x, 10)

    def test_class_holding_class(self):
        e = Exterior()

        anI = e.i
        anI.x = 10
        self.assertEqual(e.i.x, 10)

        # verify we get proper referencing to underlying python objects
        anI2 = e.iTup.x
        anI2.x = 10
        self.assertEqual(e.iTup.x.x, 10)

    def test_class_stringification(self):
        self.assertEqual(Interior.__qualname__, "Interior")
        self.assertEqual(str(Interior()), "Interior(x=0, y=0)")

    def test_class_functions_work(self):
        class C(Class, Final):
            x = Member(int)

            def setX(self, x):
                self.x = x

        c = C()

        c.setX(20)

        self.assertEqual(c.x, 20)

    def test_class_functions_return_types(self):
        class C(Class, Final):
            def returnsInt(self, x) -> int:
                return x

            def returnsFloat(self, x) -> float:
                return x

        c = C()

        c.returnsInt(20)

        with self.assertRaises(TypeError):
            # this should throw because we cannot convert a string to an int
            c.returnsInt("hi")

        # this should throw because we are happy to convert int to float
        c.returnsFloat(1)

    def test_class_function_dispatch_on_arity(self):
        class C(Class, Final):
            def f(self):
                return 0

            def f(self, i):  # noqa: F811
                return 1

            def f(self, i, i2):  # noqa: F811
                return 2

            def f(self, i, i2, *args):  # noqa: F811
                return 2 + len(args)

            def f(self, i, i2, *args):  # noqa: F811
                return 2 + len(args)

        c = C()

        self.assertEqual(c.f(), 0)
        self.assertEqual(c.f(1), 1)
        self.assertEqual(c.f(1, 2), 2)
        self.assertEqual(c.f(1, 2, 3), 3)

    def test_class_function_exceptions(self):
        class C(Class, Final):
            def g(self, a, b):
                assert False

            def f(self, a, b):
                return 1

        c = C()

        with self.assertRaises(AssertionError):
            c.g(1, 2)

    def test_class_function_sends_args_to_right_place(self):
        def g(a, b, c=10, d=20, *args, **kwargs):
            return (a, b, d, args, kwargs)

        class C(Class, Final):
            def g(self, a, b, c=10, d=20, *args, **kwargs):
                return (a, b, d, args, kwargs)

        c = C()

        def assertSame(takesCallable):
            self.assertEqual(
                takesCallable(c.g),
                takesCallable(g)
            )

        assertSame(lambda formOfG: formOfG(1, 2))
        assertSame(lambda formOfG: formOfG(1, 2, 3))
        assertSame(lambda formOfG: formOfG(1, b=2))
        assertSame(lambda formOfG: formOfG(a=1, b=2))
        assertSame(lambda formOfG: formOfG(a=1, b=2, c=20))
        assertSame(lambda formOfG: formOfG(a=1, b=2, c=20, d=30))
        assertSame(lambda formOfG: formOfG(1, 2, 3, 4))
        assertSame(lambda formOfG: formOfG(1, 2, 3, 4, 5, 6))
        assertSame(lambda formOfG: formOfG(1, 2, 3, 4, 5, 6, q=20))

    def test_class_function_type_dispatch(self):
        class C(Class, Final):
            def f(self, a: float):
                return float

            def f(self, a):  # noqa: F811
                return "any"

            def f(self, *args: int):  # noqa: F811
                return "int list"

            def f(self, *args: str):  # noqa: F811
                return "string list"

            def f(self, **kwargs: TupleOf(int)):  # noqa: F811
                return "named tuple of ints"

        self.assertEqual(C().f(1.0), float)
        self.assertEqual(C().f(1), "any")
        self.assertEqual(C().f("asdf"), "any")
        self.assertEqual(C().f("asdf", "asdf2"), "string list")
        self.assertEqual(C().f(1, 2), "int list")
        with self.assertRaises(TypeError):
            C().f(1, "hi")
        self.assertEqual(C().f(x=(1, 2)), "named tuple of ints")

    def test_class_members_accessible(self):
        class C(Class, Final):
            x = 10
            y = Member(int)

        c = C()

        self.assertEqual(c.x, 10)
        self.assertEqual(c.y, 0)

        with self.assertRaisesRegex(AttributeError, "Cannot modify read-only class member"):
            c.x = 20

        with self.assertRaisesRegex(AttributeError, "cannot add attributes to instances of this type"):
            c.z = 20

        self.assertEqual(C.x, 10)
        self.assertEqual(C.y, Member(int))

    def test_static_methods(self):
        class C(Class, Final):
            @staticmethod
            def f(a: float):
                return float

            @staticmethod  # noqa: F811
            def f(a):  # noqa: F811
                return "any"

            @staticmethod  # noqa: F811
            def f(*args: int):  # noqa: F811
                return "int list"

            @staticmethod  # noqa: F811
            def f(*args: str):  # noqa: F811
                return "string list"

            @staticmethod  # noqa: F811
            def f(**kwargs: TupleOf(int)):  # noqa: F811
                return "named tuple of ints"

        for thing in [C(), C]:
            self.assertEqual(thing.f(1.0), float)
            self.assertEqual(thing.f("asdf"), "any")
            self.assertEqual(thing.f("asdf", "asdf2"), "string list")
            self.assertEqual(thing.f(1, 2), "int list")
            with self.assertRaises(TypeError):
                thing.f(1, "hi")
            self.assertEqual(thing.f(x=(1, 2)), "named tuple of ints")

    def test_python_objects_in_classes(self):
        class NormalPyClass:
            pass

        class NormalPySubclass(NormalPyClass):
            pass

        NT = NamedTuple(x=int, y=int)

        class NTSubclass(NT):
            def fun(self):
                return self.x + self.y

        class X(Class, Final):
            anything = Member(object)
            pyclass = Member(OneOf(None, NormalPyClass))
            pysubclass = Member(OneOf(None, NormalPySubclass))
            holdsNT = Member(NT)
            holdsNTSubclass = Member(NTSubclass)

            def f(self, x: NTSubclass):
                return "NTSubclass"

            def f(self, x: NormalPySubclass):  # noqa: F811
                return "NormalPySubclass"

            def f(self, x: NormalPyClass):  # noqa: F811
                return "NormalPyClass"

            def f(self, x):  # noqa: F811
                return "object"

        x = X()

        # x.anything
        x.anything = 10
        self.assertEqual(x.anything, 10)

        x.anything = NormalPyClass()
        self.assertIsInstance(x.anything, NormalPyClass)

        # x.pyclass
        x.pyclass = NormalPyClass()
        self.assertIsInstance(x.pyclass, NormalPyClass)
        with self.assertRaises(AssertionError):
            self.assertIsInstance(x.pyclass, NormalPySubclass)

        x.pyclass = NormalPySubclass()
        self.assertIsInstance(x.pyclass, NormalPyClass)
        self.assertIsInstance(x.pyclass, NormalPySubclass)

        x.pyclass = None
        self.assertIsInstance(x.pyclass, type(None))

        with self.assertRaises(TypeError):
            x.pyclass = 10

        # x.pysubclass
        x.pysubclass = NormalPySubclass()
        self.assertIsInstance(x.pysubclass, NormalPySubclass)
        self.assertIsInstance(x.pysubclass, NormalPyClass)

        with self.assertRaises(TypeError):
            x.pysubclass = NormalPyClass()

        self.assertEqual(x.f(NT()), "NTSubclass")
        self.assertEqual(x.f(NTSubclass()), "NTSubclass")
        self.assertEqual(x.f(NormalPySubclass()), "NormalPySubclass")
        self.assertEqual(x.f(NormalPyClass()), "NormalPyClass")
        self.assertEqual(x.f(10), "object")

    def test_class_with_getitem(self):
        class WithGetitem(Class, Final):
            def __getitem__(self, x: int):
                return "Int"

            def __getitem__(self, x: str):  # noqa: F811
                return "Str"

        self.assertEqual(WithGetitem()[0], "Int")
        self.assertEqual(WithGetitem()["hi"], "Str")

        with self.assertRaises(TypeError):
            WithGetitem()[None]

    def test_class_with_len(self):
        class WithLen(Class, Final):
            x = Member(int)

            def __init__(self, x):
                self.x = x

            def __len__(self):
                return self.x

        self.assertEqual(len(WithLen(10)), 10)
        self.assertEqual(len(WithLen(0)), 0)

        with self.assertRaises(ValueError):
            len(WithLen(-1))

        with self.assertRaises(ValueError):
            len(WithLen(-2))

    def test_class_unary_operators(self):
        class WithLotsOfOperators(Class, Final):
            def __neg__(self):
                return "neg"

            def __pos__(self):
                return "pos"

            def __abs__(self):
                return "abs"

            def __invert__(self):
                return "inv"

            def __int__(self):
                return 10203

            def __float__(self):
                return 123.5

            def __index__(self):
                return 2

        c = WithLotsOfOperators()

        self.assertEqual(abs(c), "abs")
        self.assertEqual(-c, "neg")
        self.assertEqual(+c, "pos")
        self.assertEqual(~c, "inv")
        self.assertEqual(int(c), 10203)
        self.assertEqual(float(c), 123.5)
        self.assertEqual([1, 2, 3][c], 3)

    def test_class_binary_operators(self):
        class WithLotsOfOperators(Class, Final):
            def __add__(self, other):
                return (self, "add", other)

            def __sub__(self, other):
                return (self, "sub", other)

            def __mul__(self, other):
                return (self, "mul", other)

            def __mod__(self, other):
                return (self, "mod", other)

            def __truediv__(self, other):
                return (self, "div", other)

            def __floordiv__(self, other):
                return (self, "floordiv", other)

            def __lshift__(self, other):
                return (self, "lshift", other)

            def __rshift__(self, other):
                return (self, "rshift", other)

            def __or__(self, other):
                return (self, "or", other)

            def __and__(self, other):
                return (self, "and", other)

            def __xor__(self, other):
                return (self, "xor", other)

            def __matmul__(self, other):
                return (self, "matmul", other)

        c = WithLotsOfOperators()

        self.assertEqual(c+0, (c, "add", 0))
        self.assertEqual(c-0, (c, "sub", 0))
        self.assertEqual(c*0, (c, "mul", 0))
        self.assertEqual(c/0, (c, "div", 0))
        self.assertEqual(c%0, (c, "mod", 0))
        self.assertEqual(c//0, (c, "floordiv", 0))
        self.assertEqual(c<<0, (c, "lshift", 0))
        self.assertEqual(c>>0, (c, "rshift", 0))
        self.assertEqual(c|0, (c, "or", 0))
        self.assertEqual(c&0, (c, "and", 0))
        self.assertEqual(c^0, (c, "xor", 0))
        self.assertEqual(c@0, (c, "matmul", 0))

    def test_class_binary_operators_reverse(self):
        class WithLotsOfOperators(Class, Final):
            def __radd__(self, other):
                return (self, "add", other)

            def __rsub__(self, other):
                return (self, "sub", other)

            def __rmul__(self, other):
                return (self, "mul", other)

            def __rmod__(self, other):
                return (self, "mod", other)

            def __rtruediv__(self, other):
                return (self, "div", other)

            def __rfloordiv__(self, other):
                return (self, "floordiv", other)

            def __rlshift__(self, other):
                return (self, "lshift", other)

            def __rrshift__(self, other):
                return (self, "rshift", other)

            def __ror__(self, other):
                return (self, "or", other)

            def __rand__(self, other):
                return (self, "and", other)

            def __rxor__(self, other):
                return (self, "xor", other)

            def __rmatmul__(self, other):
                return (self, "matmul", other)

        c = WithLotsOfOperators()

        self.assertEqual(0+c, (c, "add", 0))
        self.assertEqual(0-c, (c, "sub", 0))
        self.assertEqual(0*c, (c, "mul", 0))
        self.assertEqual(0/c, (c, "div", 0))
        self.assertEqual(0%c, (c, "mod", 0))
        self.assertEqual(0//c, (c, "floordiv", 0))
        self.assertEqual(0<<c, (c, "lshift", 0))
        self.assertEqual(0>>c, (c, "rshift", 0))
        self.assertEqual(0|c, (c, "or", 0))
        self.assertEqual(0&c, (c, "and", 0))
        self.assertEqual(0^c, (c, "xor", 0))
        self.assertEqual(0@c, (c, "matmul", 0))

    def test_class_binary_inplace_operators(self):
        class WithLotsOfOperators(Class, Final):
            def __iadd__(self, other):
                return (self, "iadd", other)

            def __isub__(self, other):
                return (self, "isub", other)

            def __imul__(self, other):
                return (self, "imul", other)

            def __imod__(self, other):
                return (self, "imod", other)

            def __itruediv__(self, other):
                return (self, "itruediv", other)

            def __ifloordiv__(self, other):
                return (self, "ifloordiv", other)

            def __ilshift__(self, other):
                return (self, "ilshift", other)

            def __irshift__(self, other):
                return (self, "irshift", other)

            def __ior__(self, other):
                return (self, "ior", other)

            def __iand__(self, other):
                return (self, "iand", other)

            def __ixor__(self, other):
                return (self, "ixor", other)

            def __imatmul__(self, other):
                return (self, "imatmul", other)

        c = WithLotsOfOperators()

        self.assertEqual(operator.iadd(c, 0), (c, "iadd", 0))
        self.assertEqual(operator.isub(c, 0), (c, "isub", 0))
        self.assertEqual(operator.imul(c, 0), (c, "imul", 0))
        self.assertEqual(operator.imod(c, 0), (c, "imod", 0))
        self.assertEqual(operator.itruediv(c, 0), (c, "itruediv", 0))
        self.assertEqual(operator.ifloordiv(c, 0), (c, "ifloordiv", 0))
        self.assertEqual(operator.ilshift(c, 0), (c, "ilshift", 0))
        self.assertEqual(operator.irshift(c, 0), (c, "irshift", 0))
        self.assertEqual(operator.ior(c, 0), (c, "ior", 0))
        self.assertEqual(operator.iand(c, 0), (c, "iand", 0))
        self.assertEqual(operator.ixor(c, 0), (c, "ixor", 0))
        self.assertEqual(operator.imatmul(c, 0), (c, "imatmul", 0))

    def test_class_dispatch_on_tuple_vs_list(self):
        class WithTwoFunctions(Class, Final):
            def f(self, x: TupleOf(int)):
                return "Tuple"

            def f(self, x: ListOf(int)):  # noqa: F811
                return "List"

        c = WithTwoFunctions()
        self.assertEqual(c.f(TupleOf(int)((1, 2, 3))), "Tuple")
        self.assertEqual(c.f(ListOf(int)((1, 2, 3))), "List")

    def test_class_comparison_operators(self):
        class ClassWithComparisons(Class, Final):
            x = Member(int)

            def __init__(self, x):
                self.x = x

            def __eq__(self, other):
                return self.x == other.x

            def __ne__(self, other):
                return self.x != other.x

            def __lt__(self, other):
                return self.x < other.x

            def __gt__(self, other):
                return self.x > other.x

            def __le__(self, other):
                return self.x <= other.x

            def __ge__(self, other):
                return self.x >= other.x

        for i in [0, 1, 2, 3]:
            for j in [0, 1, 2, 3]:
                self.assertEqual(
                    ClassWithComparisons(i) < ClassWithComparisons(j),
                    i < j
                )
                self.assertEqual(
                    ClassWithComparisons(i) > ClassWithComparisons(j),
                    i > j
                )
                self.assertEqual(
                    ClassWithComparisons(i) <= ClassWithComparisons(j),
                    i <= j
                )
                self.assertEqual(
                    ClassWithComparisons(i) >= ClassWithComparisons(j),
                    i >= j
                )
                self.assertEqual(
                    ClassWithComparisons(i) == ClassWithComparisons(j),
                    i == j
                )
                self.assertEqual(
                    ClassWithComparisons(i) != ClassWithComparisons(j),
                    i != j
                )

    def test_class_comparison_subclassing(self):
        class ClassWithComparisons(Class):
            x = Member(int)

        class SubclassWithComparisons(ClassWithComparisons, Final):
            def __lt__(self, other):
                return self.x < other.x

        with self.assertRaises(Exception):
            ClassWithComparisons(x=10) < ClassWithComparisons(x=20)

        self.assertLess(
            SubclassWithComparisons(x=9),
            ClassWithComparisons(x=10)
        )
        self.assertLess(
            SubclassWithComparisons(x=9),
            SubclassWithComparisons(x=10)
        )

    def test_class_repr_and_str_and_hash(self):
        class ClassWithReprAndStr(Class, Final):
            def __repr__(self):
                return "repr"

            def __str__(self):
                return "str"

            def __hash__(self):
                return 300

        self.assertEqual(hash(ClassWithReprAndStr()), 300)
        self.assertEqual(repr(ClassWithReprAndStr()), "repr")
        self.assertEqual(str(ClassWithReprAndStr()), "str")

    def test_class_missing_inplace_operators_fallback(self):

        class ClassWithoutInplaceOp(Class, Final):
            def __add__(self, other):
                return "worked"

            def __sub__(self, other):
                return "worked"

            def __mul__(self, other):
                return "worked"

            def __matmul__(self, other):
                return "worked"

            def __truediv__(self, other):
                return "worked"

            def __floordiv__(self, other):
                return "worked"

            def __mod__(self, other):
                return "worked"

            def __pow__(self, other):
                return "worked"

            def __lshift__(self, other):
                return "worked"

            def __rshift__(self, other):
                return "worked"

            def __and__(self, other):
                return "worked"

            def __or__(self, other):
                return "worked"

            def __xor__(self, other):
                return "worked"

        c = ClassWithoutInplaceOp()
        c += 10
        self.assertEqual(c, "worked")
        c = ClassWithoutInplaceOp()
        c -= 10
        self.assertEqual(c, "worked")
        c = ClassWithoutInplaceOp()
        c *= 10
        self.assertEqual(c, "worked")
        c = ClassWithoutInplaceOp()
        c @= 10
        self.assertEqual(c, "worked")
        c = ClassWithoutInplaceOp()
        c /= 10
        self.assertEqual(c, "worked")
        c = ClassWithoutInplaceOp()
        c //= 10
        self.assertEqual(c, "worked")
        c = ClassWithoutInplaceOp()
        c %= 10
        self.assertEqual(c, "worked")
        c = ClassWithoutInplaceOp()
        c **= 10
        self.assertEqual(c, "worked")
        c = ClassWithoutInplaceOp()
        c <<= 10
        self.assertEqual(c, "worked")
        c = ClassWithoutInplaceOp()
        c >>= 10
        self.assertEqual(c, "worked")
        c = ClassWithoutInplaceOp()
        c &= 10
        self.assertEqual(c, "worked")
        c = ClassWithoutInplaceOp()
        c |= 10
        self.assertEqual(c, "worked")
        c = ClassWithoutInplaceOp()
        c ^= 10
        self.assertEqual(c, "worked")

    def test_class_with_property(self):
        class ClassWithProperty(Class, Final):
            _x = Member(int)

            def __init__(self, x):
                self._x = x

            @property
            def x(self):
                return self._x + 1

        self.assertEqual(ClassWithProperty(10).x, 11)

    def test_class_with_bound_methods(self):
        class SomeClass:
            pass

        class SomeSubclass(SomeClass):
            def __init__(self, x):
                self.x = x

        class ClassWithBoundMethod(Class, Final):
            x = Member(OneOf(None, SomeClass))

            def __init__(self):
                self.x = None

            def increment(self, y):
                if self.x is None:
                    self.x = SomeSubclass(y)
                else:
                    self.x = SomeSubclass(self.x.x + y)

        c = ClassWithBoundMethod()
        for _ in range(1000000):
            c.increment(2)

        self.assertEqual(c.x.x, 2000000)

    def test_class_inheritance_basic(self):
        class BaseClass(Class):
            x = Member(int)

            def f(self, add) -> object:
                return self.x + add

        class ChildClass(BaseClass):
            y = Member(int)

            def f(self, add) -> object:
                return self.x + self.y + add

        b = BaseClass(x=10)
        c = ChildClass(x=10, y=20)

        self.assertEqual(b.x, 10)
        self.assertEqual(b.f(100), 110)

        self.assertEqual(c.x, 10)
        self.assertEqual(c.y, 20)
        self.assertEqual(c.f(100), 130)

    def test_class_inheritance_and_containers(self):
        class BaseClass(Class):
            pass

        class Child1(BaseClass):
            pass

        class Child2(BaseClass):
            pass

        clsList = ListOf(BaseClass)()

        clsList.append(Child1())
        clsList.append(Child2())

        self.assertIsInstance(clsList[0], Child1)
        self.assertIsInstance(clsList[1], Child2)

    def test_class_multiple_inheritence(self):
        class BaseA(Class):
            def f(self, x: int) -> str:
                return "int"

            def g(self) -> str:
                return "BaseA"

        class BaseB(Class):
            def f(self, x: float) -> str:
                return "float"

            def g(self) -> str:
                return "BaseB"

        class BaseBoth(BaseA, BaseB):
            pass

        x = BaseBoth()

        self.assertEqual(x.f(1), "int")
        self.assertEqual(x.f(1.0), "float")

        # because of the signature rules, the MRO dictates we'll always
        # get "BaseA" first.
        self.assertEqual(x.g(), "BaseA")

    def test_multiple_inheritance_with_members_in_both_children_fails(self):
        class BaseA(Class):
            x = Member(int)

            def f(self, x: int) -> str:
                return "int"

            def g(self) -> str:
                return "BaseA"

        class BaseB(Class):
            y = Member(int)

            def f(self, x: float) -> str:
                return "float"

            def g(self) -> str:
                return "BaseB"

        with self.assertRaisesRegex(
            TypeError,
            "Class BaseBoth can't have data members because its base classes BaseA and BaseB both have members."
        ):
            class BaseBoth(BaseA, BaseB):
                pass

    def test_member_order(self):
        class BaseClass(Class):
            x = Member(int)
            y = Member(int)

        class ChildClass(BaseClass):
            z = Member(int)

        self.assertEqual(ChildClass.MemberNames, ('x', 'y', 'z'))

    def test_final_classes(self):
        class BaseClass(Class):
            pass

        class ChildClass(Class, Final):
            pass

        self.assertFalse(BaseClass.IsFinal)
        self.assertTrue(ChildClass.IsFinal)

        with self.assertRaisesRegex(Exception, "Can't subclass ChildClass because it's marked 'Final'"):
            class BadClass(ChildClass):
                pass

    def test_callable_class(self):
        class CallableClass(Class, Final):
            x = Member(int)

            def __call__(self, x):
                return self.x + x

            def __call__(self):  # noqa: F811
                return -1

        class RegularClass(Class, Final):
            x = Member(int)

            def call(self, x):
                return self.x + x

        obj = CallableClass(x=42)
        self.assertEqual(obj(0), 42)
        self.assertEqual(obj(1), 43)
        self.assertEqual(obj(), -1 )

        exceptionMsg = "Cannot find a valid overload of '__call__' with arguments of type"
        with self.assertRaisesRegex(TypeError, exceptionMsg):
            obj(1, 2, 3)

        obj = RegularClass(x=42)
        self.assertEqual(obj.call(5), 47)
        with self.assertRaises(TypeError):
            obj()

    def test_recursive_classes_repr(self):
        A0 = Forward("A0")

        class ASelfRecursiveClass(Class, Final):
            x = Member(OneOf(None, A0))

        A0 = A0.define(ASelfRecursiveClass)

        a = ASelfRecursiveClass()
        a.x = a

        b = ASelfRecursiveClass()
        b.x = b

        print(repr(a))

    def test_dispatch_tries_without_conversion_first(self):
        class ClassWithForcedConversion(Class, Final):
            def f(self, x: float):
                return "float"

        class ClassWithBoth(Class, Final):
            def f(self, x: float):
                return "float"

            def f(self, x: int):  # noqa: F811
                return "int"

            def f(self, x: bool):  # noqa: F811
                return "bool"

        # swap the order
        class ClassWithBoth2(Class, Final):
            def f(self, x: bool):
                return "bool"

            def f(self, x: int):  # noqa: F811
                return "int"

            def f(self, x: float):  # noqa: F811
                return "float"

        self.assertEqual(ClassWithForcedConversion().f(10), "float")
        self.assertEqual(ClassWithForcedConversion().f(10.5), "float")
        self.assertEqual(ClassWithForcedConversion().f(True), "float")

        self.assertEqual(ClassWithBoth().f(10), "int")
        self.assertEqual(ClassWithBoth().f(10.5), "float")
        self.assertEqual(ClassWithBoth().f(True), "bool")

        self.assertEqual(ClassWithBoth2().f(10), "int")
        self.assertEqual(ClassWithBoth2().f(10.5), "float")
        self.assertEqual(ClassWithBoth2().f(True), "bool")

    def test_class_magic_methods(self):

        class AClass(Class, Final):
            _n = Member(str)

            def __init__(self, n=""):
                self._n = n
            __bool__ = lambda self: bool(self._n)
            __str__ = lambda self: "str"
            __repr__ = lambda self: "repr"
            __call__ = lambda self: "call"
            __len__ = lambda self: 42
            __contains__ = lambda self, item: not not item

            __add__ = lambda lhs, rhs: AClass("add")
            __sub__ = lambda lhs, rhs: AClass("sub")
            __mul__ = lambda lhs, rhs: AClass("mul")
            __matmul__ = lambda lhs, rhs: AClass("matmul")
            __truediv__ = lambda lhs, rhs: AClass("truediv")
            __floordiv__ = lambda lhs, rhs: AClass("floordiv")
            __mod__ = lambda lhs, rhs: AClass("mod")
            __pow__ = lambda lhs, rhs: AClass("pow")
            __lshift__ = lambda lhs, rhs: AClass("lshift")
            __rshift__ = lambda lhs, rhs: AClass("rshift")
            __and__ = lambda lhs, rhs: AClass("and")
            __or__ = lambda lhs, rhs: AClass("or")
            __xor__ = lambda lhs, rhs: AClass("xor")

            __neg__ = lambda self: AClass("neg")
            __pos__ = lambda self: AClass("pos")
            __invert__ = lambda self: AClass("invert")

            __abs__ = lambda self: AClass("abs")
            __int__ = lambda self: 123
            __float__ = lambda self: 234.5
            __index__ = lambda self: 124
            __complex__ = lambda self: complex(1, 2)
            __round__ = lambda self: 6
            __trunc__ = lambda self: 7
            __floor__ = lambda self: 8
            __ceil__ = lambda self: 9

            __bytes__ = lambda self: b'bytes'
            __format__ = lambda self, spec: "format"

        self.assertEqual(AClass().__bool__(), False)
        self.assertEqual(bool(AClass()), False)
        self.assertEqual(AClass("a").__bool__(), True)
        self.assertEqual(bool(AClass("a")), True)
        self.assertEqual(AClass().__str__(), "str")
        self.assertEqual(str(AClass()), "str")
        self.assertEqual(AClass().__repr__(), "repr")
        self.assertEqual(repr(AClass()), "repr")
        self.assertEqual(AClass().__call__(), "call")
        self.assertEqual(AClass()(), "call")
        self.assertEqual(AClass().__contains__(0), False)
        self.assertEqual(AClass().__contains__(1), True)
        self.assertEqual(0 in AClass(), False)
        self.assertEqual(1 in AClass(), True)
        self.assertEqual(AClass().__len__(), 42)
        self.assertEqual(len(AClass()), 42)

        self.assertEqual(AClass().__add__(AClass())._n, "add")
        self.assertEqual((AClass() + AClass())._n, "add")
        self.assertEqual(AClass().__sub__(AClass())._n, "sub")
        self.assertEqual((AClass() - AClass())._n, "sub")
        self.assertEqual((AClass() * AClass())._n, "mul")
        self.assertEqual((AClass() @ AClass())._n, "matmul")
        self.assertEqual((AClass() / AClass())._n, "truediv")
        self.assertEqual((AClass() // AClass())._n, "floordiv")
        self.assertEqual((AClass() % AClass())._n, "mod")
        self.assertEqual((AClass() ** AClass())._n, "pow")
        self.assertEqual((AClass() >> AClass())._n, "rshift")
        self.assertEqual((AClass() << AClass())._n, "lshift")
        self.assertEqual((AClass() & AClass())._n, "and")
        self.assertEqual((AClass() | AClass())._n, "or")
        self.assertEqual((AClass() ^ AClass())._n, "xor")
        self.assertEqual((+AClass())._n, "pos")
        self.assertEqual((-AClass())._n, "neg")
        self.assertEqual((~AClass())._n, "invert")
        self.assertEqual(abs(AClass())._n, "abs")
        self.assertEqual(int(AClass()), 123)
        self.assertEqual(float(AClass()), 234.5)
        self.assertEqual(range(1000)[1:AClass():2], range(1, 124, 2))
        self.assertEqual(complex(AClass()), 1+2j)
        self.assertEqual(round(AClass()), 6)
        self.assertEqual(math.trunc(AClass()), 7)
        self.assertEqual(math.floor(AClass()), 8)
        self.assertEqual(math.ceil(AClass()), 9)

        self.assertEqual(bytes(AClass()), b"bytes")
        self.assertEqual(format(AClass()), "format")

        d = dir(AClass())
        self.assertGreater(len(d), 50)

    def test_class_magic_methods_attr(self):

        A_attrs = {"q": "value-q", "z": "value-z"}

        def A_getattr(s, n):
            if n not in A_attrs:
                raise AttributeError(f"no attribute {n}")
            return A_attrs[n]

        def A_setattr(s, n, v):
            A_attrs[n] = v

        def A_delattr(s, n):
            A_attrs.pop(n, None)

        class AClass(Class, Final):
            __getattr__ = A_getattr
            __setattr__ = A_setattr
            __delattr__ = A_delattr

        self.assertEqual(AClass().q, "value-q")
        self.assertEqual(AClass().z, "value-z")
        AClass().q = "changedvalue for q"
        self.assertEqual(AClass().q, "changedvalue for q")
        with self.assertRaises(AttributeError):
            print(AClass().invalid)
        del AClass().z
        with self.assertRaises(AttributeError):
            print(AClass().z)
        AClass().MemberNames = "can't change MemberNames"
        self.assertEqual(AClass().MemberNames, tuple())

        A2_items = dict()

        def A2_setitem(self, i, v):
            A2_items[i] = v

        class AClass2(Class, Final):
            __getattribute__ = A_getattr
            __setattr__ = A_setattr
            __delattr__ = A_delattr
            __dir__ = lambda self: list(A_attrs.keys())
            __getitem__ = lambda self, i: A2_items.get(i, i)
            __setitem__ = A2_setitem

        self.assertEqual(AClass2().q, "changedvalue for q")
        AClass2().MemberNames = "can change MemberNames"
        self.assertEqual(AClass2().MemberNames, "can change MemberNames")
        self.assertEqual(dir(AClass2()), ["MemberNames", "q"])
        self.assertEqual(AClass2()[123], 123)
        AClass2()[123] = 7
        self.assertEqual(AClass2()[123], 7)

    def test_class_magic_methods_iter(self):

        class A_iter():
            def __init__(self):
                self._cur = 0

            def __iter__(self):
                return self

            def __next__(self):
                if self._cur >= 10:
                    raise StopIteration
                self._cur += 1
                return self._cur

        class A_reversed():
            def __init__(self):
                self._cur = 11

            def __iter__(self):
                return self

            def __next__(self):
                if self._cur <= 1:
                    raise StopIteration
                self._cur -= 1
                return self._cur

        class AClass(Class, Final):
            __iter__ = lambda self: A_iter()
            __reversed__ = lambda self: A_reversed()

        self.assertEqual([x for x in AClass()], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.assertEqual([x for x in reversed(AClass())], [10, 9, 8, 7, 6, 5, 4, 3, 2, 1])

    def test_class_magic_methods_as_iterator(self):

        class B_iter():
            def __init__(self):
                self._cur = 0

            def __iter__(self):
                return self

            def __next__(self):
                if self._cur >= 10:
                    raise StopIteration
                self._cur += 1
                return self._cur

        x = B_iter()

        class Iterator(Class, Final):
            __iter__ = lambda self: self
            __next__ = lambda self: x.__next__()

        A = Alternative("A", a={'a': int},
                        __iter__=lambda self: Iterator()
                        )
        # this is a one-time iterator
        self.assertEqual([x for x in A.a()], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.assertEqual([x for x in A.a()], [])

    def test_class_magic_methods_with(self):
        depth = 0

        def A_enter(s):
            nonlocal depth
            depth += 1
            return depth

        def A_exit(s, t, v, b):
            nonlocal depth
            depth -= 1
            return True

        class AClass(Class, Final):
            __enter__ = A_enter
            __exit__ = A_exit

        self.assertEqual(depth, 0)
        with AClass():
            self.assertEqual(depth, 1)
            with AClass():
                self.assertEqual(depth, 2)
        self.assertEqual(depth, 0)

    def test_class_comparison_methods(self):
        class C(Class, Final):
            x = Member(int)
            __eq__ = lambda self, other: self.x % 3 == other.x % 3
            __ne__ = lambda self, other: self.x % 3 != other.x % 3
            __lt__ = lambda self, other: self.x % 3 < other.x % 3
            __gt__ = lambda self, other: self.x % 3 > other.x % 3
            __le__ = lambda self, other: self.x % 3 <= other.x % 3
            __ge__ = lambda self, other: self.x % 3 >= other.x % 3
            __hash__ = lambda self: self.x % 3

        self.assertEqual(hash(C(x=1)), 1)
        self.assertEqual(hash(C(x=3)), 0)
        self.assertEqual(hash(C(x=5)), 2)
        self.assertEqual(C(x=1) == C(x=4), True)
        self.assertEqual(C(x=2) == C(x=6), False)
        self.assertEqual(C(x=2) != C(x=4), True)
        self.assertEqual(C(x=1) != C(x=4), False)
        self.assertEqual(C(x=6) < C(x=1), True)
        self.assertEqual(C(x=6) < C(x=3), False)
        self.assertEqual(C(x=2) > C(x=7), True)
        self.assertEqual(C(x=4) > C(x=1), False)
        self.assertEqual(C(x=6) <= C(x=0), True)
        self.assertEqual(C(x=5) <= C(x=3), False)
        self.assertEqual(C(x=2) >= C(x=7), True)
        self.assertEqual(C(x=0) >= C(x=1), False)

    def test_class_reverse_operators(self):
        class C(Class, Final):
            __radd__ = lambda lhs, rhs: "radd" + str(rhs)
            __rsub__ = lambda lhs, rhs: "rsub" + str(rhs)
            __rmul__ = lambda lhs, rhs: "rmul" + str(rhs)
            __rmatmul__ = lambda lhs, rhs: "rmatmul" + str(rhs)
            __rtruediv__ = lambda lhs, rhs: "rtruediv" + str(rhs)
            __rfloordiv__ = lambda lhs, rhs: "rfloordiv" + str(rhs)
            __rmod__ = lambda lhs, rhs: "rmod" + str(rhs)
            __rpow__ = lambda lhs, rhs: "rpow" + str(rhs)
            __rlshift__ = lambda lhs, rhs: "rlshift" + str(rhs)
            __rrshift__ = lambda lhs, rhs: "rrshift" + str(rhs)
            __rand__ = lambda lhs, rhs: "rand" + str(rhs)
            __rxor__ = lambda lhs, rhs: "rxor" + str(rhs)
            __ror__ = lambda lhs, rhs: "ror" + str(rhs)

        values = [1, Int16(1), UInt64(1), 1.234, Float32(1.234), True, "abc",
                  ListOf(int)((1, 2)), ConstDict(str, str)({"a": "1"})]
        for v in values:
            self.assertEqual(v + C(), "radd" + str(v))
            self.assertEqual(v - C(), "rsub" + str(v))
            self.assertEqual(v * C(), "rmul" + str(v))
            self.assertEqual(v @ C(), "rmatmul" + str(v))
            self.assertEqual(v / C(), "rtruediv" + str(v))
            self.assertEqual(v // C(), "rfloordiv" + str(v))
            if type(v) != str:
                self.assertEqual(v % C(), "rmod" + str(v))
            self.assertEqual(v ** C(), "rpow" + str(v))
            self.assertEqual(v << C(), "rlshift" + str(v))
            self.assertEqual(v >> C(), "rrshift" + str(v))
            self.assertEqual(v & C(), "rand" + str(v))
            self.assertEqual(v ^ C(), "rxor" + str(v))
            self.assertEqual(v | C(), "ror" + str(v))
            with self.assertRaises(TypeError):
                C() + v
            with self.assertRaises(TypeError):
                C() - v
            with self.assertRaises(TypeError):
                C() * v
            with self.assertRaises(TypeError):
                C() @ v
            with self.assertRaises(TypeError):
                C() / v
            with self.assertRaises(TypeError):
                C() // v
            with self.assertRaises(TypeError):
                C() % v
            with self.assertRaises(TypeError):
                C() ** v
            with self.assertRaises(TypeError):
                C() << v
            with self.assertRaises(TypeError):
                C() >> v
            with self.assertRaises(TypeError):
                C() & v
            with self.assertRaises(TypeError):
                C() ^ v
            with self.assertRaises(TypeError):
                C() | v

    def test_class_default_constructible(self):
        # classes without 'init' are always default constructible,
        # and we'll initialize any non-class members that are default
        # constructible within it
        class C(Class):
            m = Member(int)

        self.assertTrue(is_default_constructible(C))

        T = Tuple(C)

        self.assertEqual(T()[0].m, 0)

        # adding any __init__ at all makes a class non-default-constructible
        class NotDC(Class):
            m = Member(int)

            def __init__(self):
                pass

        self.assertFalse(is_default_constructible(NotDC))

    def test_class_isinstance(self):
        class BaseClass(Class):
            pass

        class Subclass(BaseClass):
            pass

        def check():
            assert issubclass(Subclass, Subclass)
            assert issubclass(Subclass, BaseClass)
            assert issubclass(Subclass, Class)
            assert isinstance(Subclass(), Subclass)
            assert isinstance(Subclass(), BaseClass)
            assert isinstance(Subclass(), Class)

        check()
        Entrypoint(check)()

    def test_class_delitem(self):
        class C(Class, Final):
            deleted = Member(int)

            def __delitem__(self, k):
                self.deleted = k

        c = C()
        del c[10]

        assert c.deleted == 10

    def test_class_len_leak(self):
        class C(Class, Final):
            def __len__(self):
                return 123 + 123123123

        c = C()

        m0 = currentMemUsageMb()
        t0 = time.time()
        t1 = time.time()
        iterations = 0
        while time.time() - t0 < 2.0:
            iterations += 1

            if c:
                pass
            len(c)

            if time.time() - t1 > 1.0:
                print(iterations, currentMemUsageMb() - m0)
                t1 = time.time()
        assert currentMemUsageMb() - m0 < 10.0

    def test_class_inquiry_method_exception_resolution(self):
        class C(Class, Final):
            def __len__(self):
                return None

            def __contains__(self, other):
                return bool("is true?")

            def __delattr__(self, attr):
                return bool("is true?")

            def __setattr__(self, attr, val):
                return bool("is true?")

            def __setitem__(self, ix, thing):
                return bool("is true?")

        c = C()

        with self.assertRaisesRegex(Exception, "Cannot construct a new int from an instance of NoneType"):
            len(c)

        with self.assertRaisesRegex(Exception, "Cannot construct the value None from an instance of bool"):
            del c.x

        with self.assertRaisesRegex(Exception, "Cannot construct the value None from an instance of bool"):
            c.x = 10

        with self.assertRaisesRegex(Exception, "Cannot construct the value None from an instance of bool"):
            c[10] = 20

    def test_class_delattr(self):
        class C(Class, Final):
            x = Member(str)

        c = C(x="hi")

        assert c.x == "hi"

        del c.x

        with self.assertRaises(AttributeError):
            c.x

        assert not hasattr(c, 'x')

        c.x = "bye"

        assert c.x == "bye"

    def test_class_nonempty(self):
        class CNonempty(Class, Final):
            x = Member(str, nonempty=True)

            def __init__(self):
                pass

        class CEmpty(Class, Final):
            x = Member(str)

            def __init__(self):
                pass

        assert CNonempty().x == ""

        cN = CNonempty()
        cE = CEmpty()

        with self.assertRaisesRegex(Exception, "Attribute 'x' cannot be deleted"):
            del cN.x

        assert hasattr(cE, 'x')
        del cE.x

        with self.assertRaisesRegex(Exception, "Attribute 'x' is not initialized"):
            del cE.x

        assert not hasattr(cE, 'x')
        cE.x = "hihi"
        assert hasattr(cE, 'x')

    def test_class_nonempty_forces_construction(self):
        class SomeOtherClass(Class):
            pass

        class CNonempty(Class, Final):
            x = Member(SomeOtherClass, nonempty=True)

            def __init__(self):
                pass

        class CEmpty(Class, Final):
            x = Member(SomeOtherClass)

            def __init__(self):
                pass

        assert hasattr(CNonempty(), 'x')
        assert not hasattr(CEmpty(), 'x')

    def test_class_mro_index_is_zero_for_self(self):
        class BaseClass(Class):
            pass

        class ChildClass(BaseClass, Final):
            pass

        assert _types.classGetDispatchIndex(BaseClass, ChildClass) == 1
        assert _types.classGetDispatchIndex(BaseClass, BaseClass) == 0
        assert _types.classGetDispatchIndex(ChildClass, ChildClass) == 0

    def test_method_return_type_overrides_are_sensible(self):
        def makeClassReturning(Base, T, value):
            if not T:
                class Cls(Class, *([Base] if Base else [])):
                    def f(self):
                        return value
            else:
                class Cls(Class, *([Base] if Base else [])):
                    def f(self) -> T:
                        return value

            return Cls

        def makeOverrideSequence(*typesAndValues):
            C = makeClassReturning(None, typesAndValues[0][0], typesAndValues[0][1])
            classes = [C]

            for T, value in typesAndValues[1:]:
                C = makeClassReturning(C, T, value)
                classes.append(C)

            return classes

        def assertSequenceValid(*typesAndValues):
            classes = makeOverrideSequence(*typesAndValues)

            for i in range(len(classes)):
                instance = classes[i]()

                # this shouldn't throw
                instance.f()

                for priorI in range(i + 1):
                    if typesAndValues[priorI][0] and typesAndValues[i][0]:
                        assert canConvertToTrivially(typesAndValues[i][0], typesAndValues[priorI][0])

        def assertSequenceInvalid(*typesAndValues):
            classes = makeOverrideSequence(*typesAndValues)

            with self.assertRaises(TypeError):
                classes[-1]().f()

        # assert that a sequence of (T, value) pairs in base->child class relationships
        # is valid. None encodes that we have no assumptions about the type.
        assertSequenceValid((None, 0))
        assertSequenceValid((int, 0))
        assertSequenceValid((int, 0), (int, 1))
        assertSequenceValid((OneOf(int, float), 0.5), (int, 1))
        assertSequenceValid((None, 0), (int, 1))
        assertSequenceValid((int, 0), (None, 1))
        assertSequenceValid((int, 0), (Value(1), 1))

        assertSequenceInvalid((int, 0), (float, 1.0))
        assertSequenceInvalid((int, 0), (None, 0), (float, 1.0))
        assertSequenceInvalid((int, 0), (int, 0), (float, 1.0))

    def test_method_return_types_depend_on_specialization(self):
        class BaseClass(Class):
            def f(self, x: OneOf(int, None)) -> int:
                return x or 0

        class ChildClass(BaseClass):
            def f(self, x: OneOf(str, None)) -> str:
                return x or ""

        assert ChildClass().f("1.5") == "1.5"
        assert ChildClass().f(0) == 0

        # in this case, it's ambiguous what we should be returning
        with self.assertRaisesRegex(TypeError, "promise"):
            ChildClass().f(None)

    def test_compiled_dispatch_on_class(self):
        class A(Class):
            pass

        class B(Class):
            pass

        class Both(A, B):
            pass

        class TestClass(Class):
            def f(self, x: A) -> str:
                return "A"

            def f(self, x: B) -> str:  # noqa
                return "B"

        # interpreter can tell the difference between the two of these
        assert TestClass().f(A()) == "A"
        assert TestClass().f(B()) == "B"

        # and 'Both' should match 'A' first
        assert TestClass().f(Both()) == "A"

        def callItAs(T):
            @Entrypoint
            def callIt(c: T, x):
                return c.f(x)

            return callIt

        @Entrypoint
        def callIt(c, x):
            return c.f(x)

        assert callIt(TestClass(), A()) == "A"
        assert callIt(TestClass(), B()) == "B"
        assert callIt(TestClass(), Both()) == "A"

        class Subclass(TestClass):
            def f(self, x: Both):
                return "Both"

        assert callItAs(TestClass)(Subclass(), A()) == "A"
        assert callItAs(TestClass)(Subclass(), B()) == "B"
        assert callItAs(TestClass)(Subclass(), Both()) == "Both"

    def test_dispatch_on_class_infers_type_correctly(self):
        class A(Class):
            pass

        class B(Class):
            pass

        class C(Class):
            pass

        class TestClass(Class):
            def f(self, x: A) -> int:
                return "A"

            def f(self, x: B) -> float:  # noqa
                return "B"

        @Entrypoint
        def callIt(c, x):
            return c.f(x)

        assert callIt.resultTypeFor(TestClass, A).typeRepresentation is int
        assert callIt.resultTypeFor(TestClass, B).typeRepresentation is OneOf(float, int)
        assert callIt.resultTypeFor(TestClass, C).typeRepresentation is OneOf(float, int)

    def test_cant_override_class_members(self):
        class A(Class):
            X = int

        with self.assertRaisesRegex(TypeError, "Class B can't redefine classmember X"):
            class B(A):
                X = float

    def test_mro_equivalent_to_python_diamond(self):
        def makeClasses(base):
            class B(base):
                pass

            class C(B):
                pass

            class D(B):
                pass

            class E(C, D):
                pass

            return E

        EPy = makeClasses(object)
        ETP = makeClasses(Class)

        mroPy = [T.__name__ for T in EPy.__mro__][:4]
        mroTP = [T.__name__ for T in ETP.MRO][:4]

        assert mroPy == mroTP

    def test_super(self):
        class B(Class):
            def f(self, x):
                return x

        class C(B):
            def f(self, x):
                return super().f(x) + 1

        assert C().f(10) == 11

    def test_classmethod_has_methodOf(self):
        class B(Class):
            @classmethod
            def method(cls, x) -> int:
                return x

            @staticmethod
            def staticMeth(x) -> int:
                return x

        assert B.method.__func__.overloads[0].methodOf.Class is B
        assert B.staticMeth.overloads[0].methodOf.Class is B
