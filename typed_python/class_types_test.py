#   Copyright 2018 Braxton Mckee
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
import time
import gc
from typed_python.test_util import currentMemUsageMb

from typed_python import (
    Int8, NoneType, TupleOf, OneOf, Tuple, NamedTuple, ConstDict,
    Alternative, serialize, deserialize, Value, Class, Member, _types
)

class DefaultVal(Class):
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


class Interior(Class):
    x = Member(int)
    y = Member(int)


class Exterior(Class):
    x = Member(int)
    i = Member(Interior)
    iTup = Member(NamedTuple(x=Interior, y=Interior))

    def __init__(self):
        self.i = Interior()

class ClassWithInit(Class):
    x = Member(int)
    y = Member(float)
    z = Member(str)
    cwi = Member(lambda: ClassWithInit)

    def __init__(self):
        pass

    def __init__(self,x=1,cwi=None):
        self.x = x
        if cwi is not None:
            self.cwi = cwi

    def __init__(self,x):
        self.x = x

class ClassWithComplexDispatch(Class):
    x = Member(int)

    def f(self, x):
        return 'x'

    def f(self, y):
        return 'y'

class NativeClassTypesTests(unittest.TestCase):
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

        self.assertEqual(Interior(x=20,y=10).x, 20)
        self.assertEqual(Interior(x=20,y=10).y, 10)

    def executeInLoop(self, f, duration=.25):
        memUsage = currentMemUsageMb()

        t0 = time.time()

        while time.time() - t0 < duration:
            f()

        gc.collect()
        self.assertLess(currentMemUsageMb() - memUsage, 1.0)

    def test_class_create_doesnt_leak(self):
        self.executeInLoop(lambda: ClassWithInit(cwi=ClassWithInit()))

    def test_class_member_access_doesnt_leak(self):
        x = ClassWithInit(cwi=ClassWithInit())
        self.executeInLoop(lambda: x.cwi.z)

    def test_class(self):
        with self.assertRaises(TypeError):
            class A(Class):
                x = Member((1,2,3))

        class A(Class):
            x = Member(int)

            y = int #not a member. Just a value.

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

        #verify we get proper referencing to underlying python objects
        anI2 = e.iTup.x
        anI2.x = 10
        self.assertEqual(e.iTup.x.x, 10)

    def test_class_stringification(self):
        i = Interior()

        self.assertEqual(Interior.__qualname__, "Interior")
        self.assertEqual(str(Interior()), "Interior(x=0, y=0)")

    def test_class_functions_work(self):
        class C(Class):
            x = Member(int)

            def setX(self, x):
                self.x = x

        c = C()

        c.setX(20)

        self.assertEqual(c.x, 20)

    def test_class_functions_return_types(self):
        class C(Class):
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
        class C(Class):
            def f(self):
                return 0
            def f(self, i):
                return 1
            def f(self, i, i2):
                return 2
            def f(self, i, i2, *args):
                return 2 + len(args)
            def f(self, i, i2, *args):
                return 2 + len(args)

        c = C()

        self.assertEqual(c.f(), 0)
        self.assertEqual(c.f(1), 1)
        self.assertEqual(c.f(1, 2), 2)
        self.assertEqual(c.f(1, 2, 3), 3)

    def test_class_function_exceptions(self):
        class C(Class):
            def g(self, a,b):
                assert False
            def f(self, a,b):
                return 1

        c = C()

        with self.assertRaises(AssertionError):
            c.g(1,2)


    def test_class_function_sends_args_to_right_place(self):
        def g(a, b, c=10, *args, d=20, **kwargs):
            return (a, b, args, kwargs)

        class C(Class):
            def g(self, a, b, c=10, *args, d=20, **kwargs):
                return (a, b, args, kwargs)

        c = C()

        def assertSame(takesCallable):
            self.assertEqual(
                takesCallable(c.g),
                takesCallable(g)
            )

        assertSame(lambda formOfG: formOfG(1,2))
        assertSame(lambda formOfG: formOfG(1,2,3))
        assertSame(lambda formOfG: formOfG(1,b=2))
        assertSame(lambda formOfG: formOfG(a=1,b=2))
        assertSame(lambda formOfG: formOfG(a=1,b=2,c=20))
        assertSame(lambda formOfG: formOfG(a=1,b=2,c=20,d=30))
        assertSame(lambda formOfG: formOfG(1,2,3,4))
        assertSame(lambda formOfG: formOfG(1,2,3,4,5,6))
        assertSame(lambda formOfG: formOfG(1,2,3,4,5,6,d=10,q=20))


    def test_class_function_type_dispatch(self):
        class C(Class):
            def f(self, a: float):
                return float
            def f(self, a):
                return "any"

            def f(self, *args: int):
                return "int list"
            def f(self, *args: str):
                return "string list"
            def f(self, **kwargs: TupleOf(int)):
                return "named tuple of ints"

        self.assertEqual(C().f(1), float)
        self.assertEqual(C().f("asdf"), "any")
        self.assertEqual(C().f("asdf", "asdf2"), "string list")
        self.assertEqual(C().f(1,2), "int list")
        with self.assertRaises(TypeError):
            C().f(1,"hi")
        self.assertEqual(C().f(x=(1,2)), "named tuple of ints")

    def test_class_members_accessible(self):
        class C(Class):
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
        class C(Class):
            @staticmethod
            def f(a: float):
                return float

            @staticmethod
            def f(a):
                return "any"

            @staticmethod
            def f(*args: int):
                return "int list"

            @staticmethod
            def f(*args: str):
                return "string list"

            @staticmethod
            def f(**kwargs: TupleOf(int)):
                return "named tuple of ints"

        for thing in [C(), C]:
            self.assertEqual(thing.f(1), float)
            self.assertEqual(thing.f("asdf"), "any")
            self.assertEqual(thing.f("asdf", "asdf2"), "string list")
            self.assertEqual(thing.f(1,2), "int list")
            with self.assertRaises(TypeError):
                thing.f(1,"hi")
            self.assertEqual(thing.f(x=(1,2)), "named tuple of ints")

    def test_python_objects_in_classes(self):
        class NormalPyClass(object):
            pass

        class NormalPySubclass(NormalPyClass):
            pass

        NT = NamedTuple(x=int, y=int)

        class NTSubclass(NT):
            def fun(self):
                return self.x + self.y

        class X(Class):
            anything = Member(object)
            pyclass = Member(OneOf(None, NormalPyClass))
            pysubclass = Member(OneOf(None, NormalPySubclass))
            holdsNT = Member(NT)
            holdsNTSubclass = Member(NTSubclass)

            def f(self, x: NTSubclass):
                return "NTSubclass"

            def f(self, x: NormalPySubclass):
                return "NormalPySubclass"

            def f(self, x: NormalPyClass):
                return "NormalPyClass"

            def f(self, x):
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


