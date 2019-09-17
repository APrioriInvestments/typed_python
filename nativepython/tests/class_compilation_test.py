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

from typed_python import Function, Class, TupleOf, ListOf, Member, OneOf, Int64, String
import typed_python._types as _types
from nativepython.runtime import Runtime, SpecializedEntrypoint
import unittest
import time


def Compiled(f):
    f = Function(f)
    return Runtime.singleton().compile(f)


def resultType(f, **kwargs):
    return Runtime.singleton().resultTypes(f, kwargs)


class AClass(Class):
    x = Member(int)
    y = Member(float)
    z = Member(TupleOf(int))

    def f(self):
        return self.x + self.y

    def f(self, arg):  # noqa
        return self.x + self.y + arg

    def g(self) -> float:
        return 100

    def add(self, x) -> float:
        return 100 + x

    def loop(self, count: int):
        i = 0
        res = self.y
        while i < count:
            res = res + self.y
            i = i + 1

        return res


class AChildClass(AClass):
    def g(self) -> float:
        return 1234

    def add(self, x) -> float:
        if isinstance(x, int):
            return 0.2

        return 1234 + x


class AClassWithAnotherClass(Class):
    x = Member(int)
    y = Member(float)
    ac = Member(AClass)


class AClassWithDefaults(Class):
    x = Member(int, 123)
    y = Member(int)


class AClassWithInit(Class):
    x = Member(int)
    y = Member(float)

    def __init__(self):
        self.x = 100
        self.y = 100.0

    def __init__(self, x):  # noqa: F811
        self.x = x
        self.y = 100.0

    def __init__(self, x, y):  # noqa: F811
        self.x = x
        self.y = y


class TestClassCompilationCompilation(unittest.TestCase):
    def test_class_attribute(self):
        a = AClass(x=10, y=20.5, z=(1, 2, 3))

        @Compiled
        def getX(a: AClass) -> int:
            return a.x

        @Compiled
        def getY(a: AClass) -> float:
            return a.y

        @Compiled
        def getZ(a: AClass) -> TupleOf(int):
            return a.z

        self.assertEqual(getX(a), a.x)
        self.assertEqual(getY(a), a.y)
        self.assertEqual(getZ(a), a.z)

    def test_class_set_attribute(self):
        a = AClass()

        aTupleOfInt = TupleOf(int)((1, 2, 3))

        @Compiled
        def setX(a: AClass, x: int) -> None:
            a.x = x

        @Compiled
        def setY(a: AClass, y: float) -> None:
            a.y = y

        @Compiled
        def setZ(a: AClass, z: TupleOf(int)) -> None:
            a.z = z

        setX(a, 20)
        setY(a, 20.5)
        setZ(a, aTupleOfInt)

        self.assertEqual(a.x, 20)
        self.assertEqual(a.y, 20.5)
        self.assertEqual(a.z, aTupleOfInt)

        self.assertEqual(_types.refcount(aTupleOfInt), 2)

        a.z = (1, 2, 3, 4)

        self.assertEqual(_types.refcount(aTupleOfInt), 1)

        a.z = aTupleOfInt

        self.assertEqual(_types.refcount(aTupleOfInt), 2)

        a = None

        self.assertEqual(_types.refcount(aTupleOfInt), 1)

    def test_class_uninitialized_attribute(self):
        @Compiled
        def set(ac: AClassWithAnotherClass, a: AClass) -> None:
            ac.ac = a

        @Compiled
        def get(ac: AClassWithAnotherClass) -> AClass:
            return ac.ac

        ac1 = AClass(x=1)
        ac2 = AClass(x=2)

        anAWithAClass = AClassWithAnotherClass(ac=ac1)

        self.assertEqual(_types.refcount(ac1), 2)
        self.assertEqual(_types.refcount(ac2), 1)
        self.assertEqual(anAWithAClass.ac.x, 1)

        set(anAWithAClass, ac2)
        self.assertEqual(_types.refcount(ac1), 1)
        self.assertEqual(_types.refcount(ac2), 2)
        self.assertEqual(anAWithAClass.ac.x, 2)

        anAWithAClass = AClassWithAnotherClass()
        self.assertEqual(_types.refcount(ac1), 1)
        self.assertEqual(_types.refcount(ac2), 1)

        with self.assertRaises(Exception):
            get(anAWithAClass)

        set(anAWithAClass, ac1)
        self.assertEqual(_types.refcount(ac1), 2)
        self.assertEqual(_types.refcount(ac2), 1)

        self.assertEqual(get(anAWithAClass).x, 1)
        self.assertEqual(get(anAWithAClass).x, 1)
        self.assertEqual(get(anAWithAClass).x, 1)
        self.assertEqual(get(anAWithAClass).x, 1)
        self.assertEqual(_types.refcount(ac1), 2)

        set(anAWithAClass, ac2)
        self.assertEqual(_types.refcount(ac1), 1)
        self.assertEqual(_types.refcount(ac2), 2)
        self.assertEqual(get(anAWithAClass).x, 2)

    def test_call_method_basic(self):
        @Compiled
        def g(c: AClass):
            return c.g()

        c = AClass()
        c2 = AChildClass()

        self.assertEqual(g(c), c.g())
        self.assertEqual(g(c2), c2.g())

    def test_call_method_dispatch_perf(self):
        @Compiled
        def addCaller(c: AClass, count: int):
            res = c.add(1) + c.add(2.5)

            for i in range(count - 1):
                res += c.add(1) + c.add(2.5)

            return res

        c = AClass()
        c2 = AChildClass()

        t0 = time.time()
        addCaller(c, 200 * 1000000)
        t1 = time.time()
        addCaller(c2, 200 * 1000000)
        t2 = time.time()

        elapsed1 = t1 - t0
        elapsed2 = t2 - t1

        print("Times were ", elapsed1, elapsed2)
        # they should take the same amount of time. Takes about 1 second for me to
        # tun this test. Note that we generate two entrypoints (one for float, one for int)
        self.assertTrue(.5 <= elapsed1 / elapsed2 <= 2.0, elapsed1 / elapsed2)

    def test_compile_class_method(self):
        c = AClass(x=20)

        t0 = time.time()
        uncompiled_res = c.loop(1000000)
        uncompiled_time = time.time() - t0

        Runtime.singleton().compile(AClass.loop)

        t0 = time.time()
        compiled_res = c.loop(1000000)
        compiled_time = time.time() - t0

        speedup = uncompiled_time / compiled_time

        self.assertGreater(speedup, 20)
        self.assertEqual(compiled_res, uncompiled_res)

        print("speedup is ", speedup)  # I get about 75

    def test_dispatch_up_to_class_method(self):
        class TestClass(Class):
            def f(self, x: OneOf(int, float)):
                return x + 1

        @Compiled
        def compiled(c: TestClass, x: float):
            return c.f(x)

        self.assertEqual(compiled(TestClass(), 123), 124.0)
        self.assertEqual(compiled(TestClass(), 123.5), 124.5)

    def test_compile_class_init(self):
        @Compiled
        def f(x: int) -> AClassWithInit:
            return AClassWithInit(x, 22.0)

        self.assertEqual(f(10).x, 10)
        self.assertEqual(f(10).y, 22.0)

    def test_compile_class_init_with_defaults(self):
        @Compiled
        def f() -> AClassWithDefaults:
            return AClassWithDefaults()

        self.assertEqual(f().x, 123)
        self.assertEqual(f().y, 0)

    def test_compile_class_repr_and_str(self):
        class ClassWithReprAndStr(Class):
            def __repr__(self):
                return "repr"

            def __str__(self):
                return "str"

        self.assertEqual(str(ListOf(ClassWithReprAndStr)([ClassWithReprAndStr()])), "[str]")
        self.assertEqual(repr(ListOf(ClassWithReprAndStr)([ClassWithReprAndStr()])), "[repr]")

        @SpecializedEntrypoint
        def callRepr(x):
            return repr(x)

        @SpecializedEntrypoint
        def callStr(x):
            return str(x)

        self.assertEqual(callRepr(ClassWithReprAndStr()), "repr")
        self.assertEqual(callStr(ClassWithReprAndStr()), "str")

    def test_compiled_class_subclass_layout(self):
        class BaseClass(Class):
            x = Member(int)
            y = Member(int)

        class ChildClass(BaseClass):
            z = Member(int)

        def f(x: BaseClass):
            return x.x + x.y

        fCompiled = Compiled(f)

        c = ChildClass(x=10, y=20, z=30)

        self.assertEqual(fCompiled(c), f(c))

    def test_class_subclass_destructors(self):
        class BaseClass(Class):
            pass

        class ChildClass(BaseClass):
            x = Member(ListOf(int))

        aListOfInt = ListOf(int)()

        aListOfBase = ListOf(BaseClass)()
        aListOfBase.append(ChildClass(x=aListOfInt))

        self.assertEqual(_types.refcount(aListOfInt), 2)

        aListOfBase.clear()

        self.assertEqual(_types.refcount(aListOfInt), 1)

    def test_class_subclass_destructors_compiled(self):
        class BaseClass(Class):
            pass

        class ChildClass(BaseClass):
            x = Member(ListOf(int))

        aListOfInt = ListOf(int)()

        aListOfBase = ListOf(BaseClass)()

        @SpecializedEntrypoint
        def clearList(l):
            l.clear()

        aListOfBase.append(ChildClass(x=aListOfInt))

        self.assertEqual(_types.refcount(aListOfInt), 2)

        clearList(aListOfBase)

        self.assertEqual(_types.refcount(aListOfInt), 1)

    def test_class_inheritance_method_signatures(self):
        class Base(Class):
            def f(self) -> int:
                return 0

        # if you override a method you can't change its output type.
        with self.assertRaisesRegex(Exception, "Overloads of 'f' don't have the same return type"):
            class BadChild(Base):
                def f(self) -> float:
                    pass

        # but if you have a non-overlapping signature it's OK because
        # the compiler wouldn't have a problem figuring out which one
        # to dispatch to.
        class GoodChild(Base):
            def f(self, x) -> float:
                pass

    def test_dispatch_with_multiple_overloads(self):
        # check that we can dispatch appropriately
        class TestClass(Class):
            def f(self, x: int) -> int:
                return x + 1

            def f(self, y: str) -> str:  # noqa
                return y + "hi"

        self.assertEqual(resultType(lambda c, a: c.f(a), c=TestClass, a=int), Int64)
        self.assertEqual(resultType(lambda c, a: c.f(a), c=TestClass, a=str), String)
        self.assertEqual(set(resultType(lambda c, a: c.f(a), c=TestClass, a=OneOf(str, int)).Types), set(OneOf(str, int).Types))
        self.assertEqual(set(resultType(lambda c, a: c.f(a), c=TestClass, a=object).Types), set(OneOf(str, int).Types))

        @Compiled
        def callWithStr(c: TestClass, x: str):
            return c.f(x)

        @Compiled
        def callWithInt(c: TestClass, x: int):
            return c.f(x)

        self.assertEqual(callWithInt(TestClass(), 1), TestClass().f(1))
        self.assertEqual(callWithStr(TestClass(), ""), TestClass().f(""))
        self.assertEqual(callWithStr(TestClass(), "hi"), TestClass().f("hi"))
