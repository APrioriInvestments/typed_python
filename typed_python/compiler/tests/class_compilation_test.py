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

import time
import psutil
from math import trunc, floor, ceil
import unittest
from flaky import flaky
from typed_python import (
    Class,
    Dict,
    ConstDict,
    TupleOf,
    ListOf,
    Member,
    OneOf,
    UInt64,
    Int16,
    Float32,
    Final,
    makeNamedTuple,
    Compiled,
    Function,
    Value,
    pointerTo,
    refTo,
    Forward,
    TypeFunction,
    typeKnownToCompiler,
    SubclassOf
)
import typed_python._types as _types
from typed_python.compiler.runtime import Entrypoint, Runtime, CountCompilationsVisitor
from typed_python.compiler.type_wrappers.class_wrapper import classCouldBeInstanceOf


def resultType(f, **kwargs):
    return Runtime.singleton().resultTypes(f, kwargs)


def result_or_exception(f, *p):
    try:
        return f(*p)
    except Exception as e:
        return type(e)


class ShouldInitializeMembers(Class):
    x = Member(OneOf(None, ListOf(int)))

    def __init__(self):
        pass


@Function
def moduleLevelIdentityFun(x):
    return x


class AClass(Class):
    x = Member(int)
    y = Member(float)
    z = Member(TupleOf(int))

    def f(self) -> float:
        return self.x + self.y

    def f(self, arg) -> float:  # noqa
        return self.x + self.y + arg

    def g(self) -> float:
        return 100

    def add(self, x) -> float:
        return 100 + x

    def loop(self, count: int) -> float:
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
    def test_initializes_correctly(self):
        ShouldInitializeMembers().x

        @Entrypoint
        def f():
            return ShouldInitializeMembers()

        f().x

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

    @flaky(max_runs=3, min_passes=1)
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
        addCaller(c, 200 * 100000)
        t1 = time.time()
        addCaller(c2, 200 * 100000)
        t2 = time.time()

        elapsed1 = t1 - t0
        elapsed2 = t2 - t1

        print("Times were ", elapsed1, elapsed2)
        # they should take the same amount of time. Takes about 1 second for me to
        # tun this test. Note that we generate two entrypoints (one for float, one for int)
        self.assertTrue(0.5 <= elapsed1 / elapsed2 <= 2.0, elapsed1 / elapsed2)

    def test_compile_class_method(self):
        c = AClass(x=20)

        t0 = time.time()
        uncompiled_res = c.loop(1000000)
        uncompiled_time = time.time() - t0

        self.assertEqual(AClass.loop.resultTypeFor(AClass, int).typeRepresentation, float)

        t0 = time.time()
        compiled_res = c.loop(1000000)
        compiled_time = time.time() - t0

        speedup = uncompiled_time / compiled_time

        self.assertGreater(speedup, 15)
        self.assertEqual(compiled_res, uncompiled_res)

        print("speedup is ", speedup)  # I get about 75

    def test_dispatch_up_to_class_method(self):
        class TestClass(Class, Final):
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
        class RegularClassWithReprAndString:
            def __repr__(self):
                return "repr"

            def __str__(self):
                return "str"

        class ClassWithReprAndStr(Class):
            def __repr__(self):
                return "repr"

            def __str__(self):
                return "str"

        self.assertEqual(str(ListOf(ClassWithReprAndStr)([ClassWithReprAndStr()])), "[repr]")
        self.assertEqual(repr(ListOf(ClassWithReprAndStr)([ClassWithReprAndStr()])), "[repr]")

        self.assertEqual(str([RegularClassWithReprAndString()]), "[repr]")
        self.assertEqual(repr([RegularClassWithReprAndString()]), "[repr]")

        @Entrypoint
        def callRepr(x):
            return repr(x)

        @Entrypoint
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

        @Entrypoint
        def clearList(l):
            l.clear()

        aListOfBase.append(ChildClass(x=aListOfInt))

        self.assertEqual(_types.refcount(aListOfInt), 2)

        clearList(aListOfBase)

        self.assertEqual(_types.refcount(aListOfInt), 1)

    def test_dispatch_with_multiple_overloads(self):
        # check that we can dispatch appropriately
        class TestClass(Class):
            def f(self, x: int) -> int:
                return x + 1

            def f(self, y: str) -> str:  # noqa
                return y + "hi"

        self.assertEqual(Function(lambda c, a: c.f(a)).resultTypeFor(TestClass, int).typeRepresentation, int)
        self.assertEqual(Function(lambda c, a: c.f(a)).resultTypeFor(TestClass, str).typeRepresentation, str)
        self.assertEqual(
            set(
                Function(lambda c, a: c.f(a))
                .resultTypeFor(TestClass, OneOf(str, int))
                .typeRepresentation.Types
            ),
            set(OneOf(str, int).Types),
        )
        self.assertEqual(
            set(Function(lambda c, a: c.f(a)).resultTypeFor(TestClass, object).typeRepresentation.Types),
            set(OneOf(str, int).Types),
        )

        @Compiled
        def callWithStr(c: TestClass, x: str):
            return c.f(x)

        @Compiled
        def callWithInt(c: TestClass, x: int):
            return c.f(x)

        self.assertEqual(callWithInt(TestClass(), 1), TestClass().f(1))
        self.assertEqual(callWithStr(TestClass(), ""), TestClass().f(""))
        self.assertEqual(callWithStr(TestClass(), "hi"), TestClass().f("hi"))

    def test_multiple_child_dispatches(self):
        # child specializations should get to look at an argument
        # of type 'OneOf(int, float)' and get to decide if they want to handle it.

        class BaseClass(Class):
            def f(self, x: OneOf(int, float)) -> str:
                if type(x) is int:
                    return "base: int"
                else:
                    return "base: float"

        class ChildClass(BaseClass):
            def f(self, x: float) -> str:
                return "child: float"

            def f(self, x: int) -> str:  # noqa
                return "child: int"

        @Compiled
        def call(c: BaseClass, x: OneOf(int, float)):
            return c.f(x)

        self.assertEqual(call(BaseClass(), 1.0), "base: float")
        self.assertEqual(call(BaseClass(), 1), "base: int")
        self.assertEqual(call(ChildClass(), 1), "child: int")
        self.assertEqual(call(ChildClass(), 1.0), "child: float")

    def test_multiple_child_dispatches_with_arg_dispatch(self):
        # the child specializations should get to look at an argument
        # of type 'OneOf(int, float)' and get to decide if they want to handle it.
        class BaseClass(Class):
            def f(self, x) -> int:
                return 1

            def f(self, x, y) -> int:  # noqa
                return 2

        class ChildClass(BaseClass):
            def f(self, x: int) -> int:
                return 11

            def f(self, x: int, y: int) -> int:  # noqa
                return 12

        @Compiled
        def call1(c: BaseClass, x: int):
            return c.f(x)

        @Compiled
        def call2(c: BaseClass, x: int, y: int):
            return c.f(x, y)

        self.assertEqual(call1(BaseClass(), 1), 1)
        self.assertEqual(call2(BaseClass(), 1, 2), 2)
        self.assertEqual(call1(ChildClass(), 1), 11)
        self.assertEqual(call2(ChildClass(), 1, 2), 12)

    def test_dispatch_to_none_works(self):
        class BaseClass(Class):
            def f(self, x) -> int:
                return 1

        class ChildClass(BaseClass):
            def f(self, x) -> int:  # noqa
                return 2

        @Compiled
        def call(c: BaseClass, x: int):
            return c.f(x)

        self.assertEqual(call(BaseClass(), 1), 1)
        self.assertEqual(call(ChildClass(), 1), 2)

    def test_dispatch_to_subclass_from_list(self):
        class BaseClass(Class):
            def f(self) -> int:
                return 1

        class ChildClass1(BaseClass):
            def f(self) -> int:
                return 2

        class ChildClass2(BaseClass):
            def f(self) -> int:
                return 3

        class ChildChildClass(ChildClass1, ChildClass2):
            def f(self) -> int:
                return 3

        def addFsUncompiled(c: ListOf(BaseClass), times: int):
            res = 0
            for t in range(times):
                for i in c:
                    res += i.f()
            return res

        addFs = Compiled(addFsUncompiled)

        aList = ListOf(BaseClass)()

        aList.append(BaseClass())
        self.assertEqual(addFs(aList, 1), sum(c.f() for c in aList))

        aList.append(ChildClass1())
        self.assertEqual(addFs(aList, 1), sum(c.f() for c in aList))

        aList.append(ChildClass2())
        self.assertEqual(addFs(aList, 1), sum(c.f() for c in aList))

        aList.append(ChildChildClass())
        self.assertEqual(addFs(aList, 1), sum(c.f() for c in aList))

        count = 100000

        t0 = time.time()
        val1 = addFs(aList, count)
        t1 = time.time()
        val2 = addFsUncompiled(aList, count)
        t2 = time.time()

        self.assertEqual(val1, val2)

        elapsedCompiled = t1 - t0
        elapsedUncompiled = t2 - t1
        speedup = elapsedUncompiled / elapsedCompiled

        print(f"speedup is {speedup}. Compiled took {elapsedCompiled} to do {count * len(aList)}")

        self.assertGreater(speedup, 20)

    def test_convert_up_in_compiled_code(self):
        class BaseClass(Class):
            def f(self) -> int:
                return 1

        class ChildClass(BaseClass):
            def f(self) -> int:
                return 2

        class ChildChildClass(ChildClass):
            def f(self) -> int:
                return 3

        class DispatchCls(Class, Final):
            def fOnBase(self, x: BaseClass):
                return x.f()

        @Compiled
        def dispatchUp(x: ChildClass):
            return DispatchCls().fOnBase(x)

        self.assertEqual(dispatchUp(ChildClass()), ChildClass().f())
        self.assertEqual(dispatchUp(ChildChildClass()), ChildChildClass().f())

    @flaky(max_runs=3, min_passes=1)
    def test_perf_of_dispatch_with_final_is_fast(self):
        class BaseClass(Class):
            def f(self) -> float:
                return 1.0

        class ChildClass(BaseClass, Final):
            pass

        @Entrypoint
        def addFs(c, times: int):
            res = 0.0
            for t in range(times):
                res += c.f()
            return res

        addFs(ChildClass(), 1)
        addFs(BaseClass(), 1)

        passes = 1e7
        t0 = time.time()
        addFs(BaseClass(), passes)
        t1 = time.time()
        addFs(ChildClass(), passes)
        t2 = time.time()

        elapsedDispatch = t1 - t0
        elapsedNoDispatch = t2 - t1
        speedup = elapsedDispatch / elapsedNoDispatch

        print(f"speedup is {speedup}. {elapsedNoDispatch} to do {passes} without dispatch.")

        self.assertGreater(speedup, 1.2)

    def test_dispatch_with_different_types(self):
        class BaseClass(Class, Final):
            def f(self, x: int) -> str:
                return "int"

            def f(self, y: float) -> str:  # noqa
                return "float"

        @Compiled
        def f(c: BaseClass, x: OneOf(int, float)):
            return c.f(x)

        self.assertEqual(f(BaseClass(), 0), "int")
        self.assertEqual(f(BaseClass(), 1.0), "float")

    def test_dispatch_with_different_output_types(self):
        class BaseClass(Class, Final):
            def f(self, x: int) -> int:
                return x + 1

            def f(self, x: float) -> float:  # noqa
                return x * 2

        @Function
        def f(x):
            return BaseClass().f(x)

        self.assertEqual(f.resultTypeFor(int).typeRepresentation, int)
        self.assertEqual(f.resultTypeFor(float).typeRepresentation, float)
        self.assertEqual(f.resultTypeFor(OneOf(int, float)).typeRepresentation, OneOf(float, int))
        self.assertEqual(f.resultTypeFor(object).typeRepresentation, OneOf(float, int))

    def test_dispatch_with_no_specified_output_types(self):
        class BaseClass(Class):
            def f(self, x: int) -> object:
                return x + 1

            def f(self, x: float) -> object:  # noqa
                return x * 2

        class BaseClassFinal(Class, Final):
            def f(self, x: int):
                return x + 1

            def f(self, x: float):  # noqa
                return x * 2

        @Function
        def f(x):
            return BaseClass().f(x)

        @Function
        def fFinal(x):
            return BaseClassFinal().f(x)

        self.assertEqual(f.resultTypeFor(OneOf(int, float)).typeRepresentation, object)
        self.assertEqual(fFinal.resultTypeFor(OneOf(int, float)).typeRepresentation, OneOf(float, int))

    def test_classes_with_lots_of_members(self):
        class BaseClass(Class):
            x00 = Member(int)
            x01 = Member(int)
            x02 = Member(int)
            x03 = Member(int)
            x04 = Member(int)
            x05 = Member(int)
            x06 = Member(int)
            x07 = Member(int)

        class ChildClass(BaseClass):
            x10 = Member(int)
            x11 = Member(int)
            x12 = Member(int)
            x13 = Member(int)
            x14 = Member(int)
            x15 = Member(int)
            x16 = Member(int)
            x17 = Member(int)

        @Compiled
        def f(c: BaseClass):
            return c.x02

        c = BaseClass()
        c.x02 = 10
        self.assertEqual(f(c), 10)

        c = ChildClass()
        c.x02 = 10
        self.assertEqual(f(c), 10)

    @flaky(max_runs=3, min_passes=1)
    def test_class_member_perf_with_final(self):
        class BaseClass(Class):
            x0 = Member(float)

        class ChildClass(BaseClass, Final):
            pass

        @Compiled
        def sumX0(x: BaseClass, t: int):
            res = 0.0
            for i in range(t):
                res += x.x0
            return res

        @Compiled
        def sumX0Child(x: ChildClass, t: int):
            res = 0.0
            for i in range(t):
                res += x.x0
            return res

        passes = 1e7
        t0 = time.time()
        sumX0(BaseClass(), passes)
        t1 = time.time()
        sumX0Child(ChildClass(), passes)
        t2 = time.time()

        elapsedDispatch = t1 - t0
        elapsedNoDispatch = t2 - t1
        speedup = elapsedDispatch / elapsedNoDispatch

        self.assertTrue(0.7 < speedup < 1.3, speedup)

    def test_class_return_self(self):
        class C(Class):
            def e(self) -> object:
                return self

        def f(a: C) -> object:
            return a

        # verify that we don't segfault
        c_f = Compiled(f)
        self.assertIsInstance(c_f(C()), C)

    def test_compile_class_magic_methods(self):
        class C(Class, Final):
            s = Member(str)

            def __init__(self, label=""):
                self.s = label

            def __eq__(self, other):
                return self.s == other.s

            __bool__ = lambda self: False
            __str__ = lambda self: "my str"
            __repr__ = lambda self: "my repr"
            __call__ = lambda self, i: "my call"
            __len__ = lambda self: 42
            __contains__ = lambda self, item: item == 1
            __bytes__ = lambda self: b"my bytes"

            __int__ = lambda self: 43
            __float__ = lambda self: 44.44
            __complex__ = lambda self: 3 + 4j

            __add__ = lambda self, other: C("add")
            __sub__ = lambda self, other: C("sub")
            __mul__ = lambda self, other: C("mul")
            __matmul__ = lambda self, other: C("matmul")
            __truediv__ = lambda self, other: C("truediv")
            __floordiv__ = lambda self, other: C("floordiv")
            __divmod__ = lambda self, other: C("divmod")
            __mod__ = lambda self, other: C("mod")
            __pow__ = lambda self, other: C("pow")
            __lshift__ = lambda self, other: C("lshift")
            __rshift__ = lambda self, other: C("rshift")
            __and__ = lambda self, other: C("and")
            __or__ = lambda self, other: C("or")
            __xor__ = lambda self, other: C("xor")

            __iadd__ = lambda self, other: C("iadd")
            __isub__ = lambda self, other: C("isub")
            __imul__ = lambda self, other: C("imul")
            __imatmul__ = lambda self, other: C("imatmul")
            __itruediv__ = lambda self, other: C("itruediv")
            __ifloordiv__ = lambda self, other: C("ifloordiv")
            __imod__ = lambda self, other: C("imod")
            __ipow__ = lambda self, other: C("ipow")
            __ilshift__ = lambda self, other: C("ilshift")
            __irshift__ = lambda self, other: C("irshift")
            __iand__ = lambda self, other: C("iand")
            __ior__ = lambda self, other: C("ior")
            __ixor__ = lambda self, other: C("ixor")

            __neg__ = lambda self: C("neg")
            __pos__ = lambda self: C("pos")
            __invert__ = lambda self: C("invert")

            __abs__ = lambda self: C("abs")

        def f_bool(x: C):
            return bool(x)

        def f_str(x: C):
            return str(x)

        def f_repr(x: C):
            return repr(x)

        def f_call(x: C):
            return x(1)

        def f_1in(x: C):
            return 1 in x

        def f_0in(x: C):
            return 0 in x

        def f_len(x: C):
            return len(x)

        def f_int(x: C):
            return int(x)

        def f_float(x: C):
            return float(x)

        def f_add(x: C):
            return x + C("")

        def f_sub(x: C):
            return x - C("")

        def f_mul(x: C):
            return x * C("")

        def f_div(x: C):
            return x / C("")

        def f_floordiv(x: C):
            return x // C("")

        def f_matmul(x: C):
            return x @ C("")

        def f_mod(x: C):
            return x % C("")

        def f_and(x: C):
            return x & C("")

        def f_or(x: C):
            return x | C("")

        def f_xor(x: C):
            return x ^ C("")

        def f_rshift(x: C):
            return x >> C("")

        def f_lshift(x: C):
            return x << C("")

        def f_pow(x: C):
            return x ** C("")

        def f_neg(x: C):
            return -x

        def f_pos(x: C):
            return +x

        def f_invert(x: C):
            return ~x

        def f_abs(x: C):
            return abs(x)

        def f_iadd(x: C):
            x += C("")
            return x

        def f_isub(x: C):
            x -= C("")
            return x

        def f_imul(x: C):
            x *= C("")
            return x

        def f_idiv(x: C):
            x /= C("")
            return x

        def f_ifloordiv(x: C):
            x //= C("")
            return x

        def f_imatmul(x: C):
            x @= C("")
            return x

        def f_imod(x: C):
            x %= C("")
            return x

        def f_iand(x: C):
            x &= C("")
            return x

        def f_ior(x: C):
            x |= C("")
            return x

        def f_ixor(x: C):
            x ^= C("")
            return x

        def f_irshift(x: C):
            x >>= C("")
            return x

        def f_ilshift(x: C):
            x <<= C("")
            return x

        def f_ipow(x: C):
            x **= C("")
            return x

        test_cases = [
            f_int,
            f_float,
            f_bool,
            f_str,
            f_repr,
            f_call,
            f_0in,
            f_1in,
            f_len,
            f_add,
            f_sub,
            f_mul,
            f_div,
            f_floordiv,
            f_matmul,
            f_mod,
            f_and,
            f_or,
            f_xor,
            f_rshift,
            f_lshift,
            f_pow,
            f_neg,
            f_pos,
            f_invert,
            f_abs,
            f_iadd,
            f_isub,
            f_imul,
            f_idiv,
            f_ifloordiv,
            f_imatmul,
            f_imod,
            f_iand,
            f_ior,
            f_ixor,
            f_irshift,
            f_ilshift,
            f_ipow,
        ]

        for f in test_cases:
            compiled_f = Compiled(f)
            r1 = f(C(""))
            r2 = compiled_f(C(""))
            self.assertEqual(r1, r2)

    def test_compile_class_reverse_methods(self):
        class C(Class, Final):
            s = Member(str)
            __radd__ = lambda self, other: "radd" + repr(other)
            __rsub__ = lambda self, other: "rsub" + repr(other)
            __rmul__ = lambda self, other: "rmul" + repr(other)
            __rmatmul__ = lambda self, other: "rmatmul" + repr(other)
            __rtruediv__ = lambda self, other: "rtruediv" + repr(other)
            __rfloordiv__ = lambda self, other: "rfloordiv" + repr(other)
            __rmod__ = lambda self, other: "rmod" + repr(other)
            __rpow__ = lambda self, other: "rpow" + repr(other)
            __rlshift__ = lambda self, other: "rlshift" + repr(other)
            __rrshift__ = lambda self, other: "rrshift" + repr(other)
            __rand__ = lambda self, other: "rand" + repr(other)
            __rxor__ = lambda self, other: "rxor" + repr(other)
            __ror__ = lambda self, other: "ror" + repr(other)

        values = [
            1,
            Int16(1),
            UInt64(1),
            1.234,
            Float32(1.234),
            True,
            "abc",
            ListOf(int)((1, 2)),
            ConstDict(str, str)({"a": "1"}),
        ]
        for v in values:
            T = type(v)

            def f_radd(v: T, x: C):
                return v + x

            def f_rsub(v: T, x: C):
                return v - x

            def f_rmul(v: T, x: C):
                return v * x

            def f_rmatmul(v: T, x: C):
                return v @ x

            def f_rtruediv(v: T, x: C):
                return v * x

            def f_rfloordiv(v: T, x: C):
                return v * x

            def f_rmod(v: T, x: C):
                return v * x

            def f_rpow(v: T, x: C):
                return v * x

            def f_rlshift(v: T, x: C):
                return v * x

            def f_rrshift(v: T, x: C):
                return v * x

            def f_rand(v: T, x: C):
                return v * x

            def f_rxor(v: T, x: C):
                return v * x

            def f_ror(v: T, x: C):
                return v * x

            for f in [
                f_radd,
                f_rsub,
                f_rmul,
                f_rmatmul,
                f_rtruediv,
                f_rfloordiv,
                f_rmod,
                f_rpow,
                f_rlshift,
                f_rrshift,
                f_rand,
                f_rxor,
                f_ror,
            ]:
                r1 = f(v, C())
                compiled_f = Compiled(f)
                r2 = compiled_f(v, C())
                self.assertEqual(r1, r2)

    def test_compile_class_format(self):
        class C1(Class, Final):
            pass

        class C2(Class, Final):
            __str__ = lambda self: "my str"

        class C3(Class, Final):
            __format__ = lambda self, spec="": "my format " + spec

        def format1(x: C1):
            return format(x)

        def format2(x: C2):
            return format(x)

        def format3(x: C3):
            return format(x)

        def format3_spec(x: C3):
            return format(x, "spec")

        r1 = format1(C1())
        c_format1 = Compiled(format1)
        r2 = c_format1(C1())
        self.assertEqual(r1, r2)

        r1 = format2(C2())
        c_format2 = Compiled(format2)
        r2 = c_format2(C2())
        self.assertEqual(r1, r2)

        r1 = format3(C3())
        c_format3 = Compiled(format3)
        r2 = c_format3(C3())
        self.assertEqual(r1, r2)

        r1 = format3_spec(C3())
        c_format3_spec = Compiled(format3_spec)
        r2 = c_format3_spec(C3())
        self.assertEqual(r1, r2)

        @Entrypoint
        def specialized_format(x):
            return format(x)

        test_values = [C1(), C2(), C3()]
        for v in test_values:
            r1 = format(v)
            r2 = specialized_format(v)
            self.assertEqual(r1, r2)

    def test_compile_class_bytes(self):
        class C(Class, Final):
            __bytes__ = lambda self: b"my bytes"

        def f_bytes(x: C):
            return bytes(x)

        v = C()
        r1 = f_bytes(v)
        c_f = Compiled(f_bytes)
        r2 = c_f(v)
        self.assertEqual(r1, r2)

    def test_compile_class_delitem(self):
        class C(Class, Final):
            deleted = Member(int)

            def __delitem__(self, k):
                self.deleted = k

        @Entrypoint
        def delIt(c, k):
            del c[k]

        c = C()
        delIt(c, 10)
        assert c.deleted == 10

    def test_compile_class_attr(self):
        class C(Class, Final):
            d = Member(Dict(str, str))
            i = Member(int)

            def __getattr__(self, n):
                return self.d[n]

            def __setattr__(self, n, v):
                self.d[n] = v

            def __delattr__(self, n):
                del self.d[n]

        def f_getattr1(x: C):
            return x.q

        def f_getattr2(x: C):
            return x.z

        def f_setattr1(x: C, s: str):
            x.q = s

        def f_setattr2(x: C, s: str):
            x.z = s

        def f_delattr1(x: C):
            del x.q

        def f_delattr2(x: C):
            del x.z

        c_getattr1 = Compiled(f_getattr1)
        c_getattr2 = Compiled(f_getattr2)
        c_setattr1 = Compiled(f_setattr1)
        c_setattr2 = Compiled(f_setattr2)
        c_delattr1 = Compiled(f_delattr1)
        c_delattr2 = Compiled(f_delattr2)
        for v in [C()]:
            f_setattr1(v, "0")
            f_setattr2(v, "0")
            self.assertEqual(f_getattr1(v), "0")
            self.assertEqual(f_getattr1(v), c_getattr1(v))
            self.assertEqual(f_getattr2(v), "0")
            self.assertEqual(f_getattr2(v), c_getattr2(v))
            f_setattr1(v, "1")
            self.assertEqual(f_getattr1(v), "1")
            self.assertEqual(f_getattr1(v), c_getattr1(v))
            self.assertEqual(f_getattr2(v), "0")
            self.assertEqual(f_getattr2(v), c_getattr2(v))
            c_setattr1(v, "2")
            self.assertEqual(f_getattr1(v), "2")
            self.assertEqual(f_getattr1(v), c_getattr1(v))
            self.assertEqual(f_getattr2(v), "0")
            self.assertEqual(f_getattr2(v), c_getattr2(v))
            f_setattr2(v, "3")
            self.assertEqual(f_getattr1(v), "2")
            self.assertEqual(f_getattr1(v), c_getattr1(v))
            self.assertEqual(f_getattr2(v), "3")
            self.assertEqual(f_getattr2(v), c_getattr2(v))
            c_setattr2(v, "4")
            self.assertEqual(f_getattr1(v), "2")
            self.assertEqual(f_getattr1(v), c_getattr1(v))
            self.assertEqual(f_getattr2(v), "4")
            self.assertEqual(f_getattr2(v), c_getattr2(v))
            f_delattr1(v)
            with self.assertRaises(KeyError):
                f_getattr1(v)
            with self.assertRaises(KeyError):
                c_getattr1(v)
            self.assertEqual(f_getattr2(v), "4")
            self.assertEqual(f_getattr2(v), c_getattr2(v))
            f_delattr2(v)
            with self.assertRaises(KeyError):
                f_getattr1(v)
            with self.assertRaises(KeyError):
                c_getattr1(v)
            with self.assertRaises(KeyError):
                f_getattr2(v)
            with self.assertRaises(KeyError):
                c_getattr2(v)
            f_setattr1(v, "5")
            f_setattr2(v, "6")
            c_delattr1(v)
            with self.assertRaises(KeyError):
                f_getattr1(v)
            with self.assertRaises(KeyError):
                c_getattr1(v)
            self.assertEqual(f_getattr2(v), "6")
            self.assertEqual(f_getattr2(v), c_getattr2(v))
            c_delattr2(v)
            with self.assertRaises(KeyError):
                f_getattr1(v)
            with self.assertRaises(KeyError):
                c_getattr1(v)
            with self.assertRaises(KeyError):
                f_getattr2(v)
            with self.assertRaises(KeyError):
                c_getattr2(v)

    def test_compile_class_float_methods(self):
        # if __float__ is defined, then floor() and ceil() are based off this conversion,
        # when __floor__ and __ceil__ are not defined
        class C(Class, Final):
            __float__ = lambda self: 1234.5

        def f_floor(x: C):
            return floor(x)

        def f_ceil(x: C):
            return ceil(x)

        test_cases = [f_floor, f_ceil]
        for f in test_cases:
            r1 = f(C())
            compiled_f = Compiled(f)
            r2 = compiled_f(C())
            self.assertEqual(r1, r2)

        class C2(Class, Final):
            __round__ = lambda self, n: 1234 + n
            __trunc__ = lambda self: 1
            __floor__ = lambda self: 2
            __ceil__ = lambda self: 3

        def f_round0(x: C2):
            return round(x, 0)

        def f_round1(x: C2):
            return round(x, 1)

        def f_round2(x: C2):
            return round(x, 2)

        def f_round_1(x: C2):
            return round(x, -1)

        def f_round_2(x: C2):
            return round(x, -2)

        def f_trunc(x: C2):
            return trunc(x)

        def f_floor(x: C2):
            return floor(x)

        def f_ceil(x: C2):
            return ceil(x)

        test_cases = [f_round0, f_round1, f_round2, f_round_1, f_round_2, f_trunc, f_floor, f_ceil]
        for f in test_cases:
            r1 = f(C2())
            compiled_f = Compiled(f)
            r2 = compiled_f(C2())
            self.assertEqual(r1, r2)

    def test_compile_class_dir(self):
        # The interpreted dir() calls __dir__() and sorts the result.
        # I expected the compiled dir() to do the same thing, but it doesn't sort.
        # So if you append these elements out of order, the test will fail.

        class C0(Class, Final):
            i = Member(int)

            def one(self):
                pass

            def two(self):
                pass

        class C(Class, Final):
            i = Member(int)

            def __dir__(self):
                x = ListOf(str)()
                x.append("x")
                x.append("y")
                x.append("z")
                return x

        def f_dir0(x: C0):
            return dir(x)

        def f_dir(x: C):
            return dir(x)

        for f in [f_dir0]:
            compiled_f = Compiled(f)
            r1 = f(C0())
            r2 = compiled_f(C0())
            self.assertEqual(r1, r2)

        for f in [f_dir]:
            compiled_f = Compiled(f)
            r1 = f(C())
            r2 = compiled_f(C())
            self.assertEqual(r1, r2)

        c0 = Compiled(f_dir0)
        c = Compiled(f_dir)
        initMem = psutil.Process().memory_info().rss / 1024 ** 2

        for i in range(10000):
            c0(C0(i=i))
            c(C(i=i))

        finalMem = psutil.Process().memory_info().rss / 1024 ** 2

        self.assertTrue(finalMem < initMem + 2)

    def test_compile_class_comparison_defaults(self):
        class C(Class, Final):
            i = Member(int)
            s = Member(str)

        def f_eq(x: C, y: C):
            return x == y

        def f_ne(x: C, y: C):
            return x != y

        def f_lt(x: C, y: C):
            return x < y

        def f_gt(x: C, y: C):
            return x > y

        def f_le(x: C, y: C):
            return x <= y

        def f_ge(x: C, y: C):
            return x >= y

        def f_hash(x: C):
            return hash(x)

        values = [C(i=0), C(i=1), C(s="a"), C(s="b")]
        for f in [f_eq, f_ne, f_lt, f_gt, f_le, f_ge]:
            for v1 in values:
                for v2 in values:
                    compiled_f = Compiled(f)
                    r1 = result_or_exception(f, v1, v2)
                    r2 = result_or_exception(compiled_f, v1, v2)
                    self.assertEqual(r1, r2, (f, v1, v2))

        for f in [f_hash]:
            for v in values:
                compiled_f = Compiled(f)
                r1 = result_or_exception(f, v)
                r2 = result_or_exception(compiled_f, v)
                self.assertEqual(r1, r2)

    def test_compile_class_comparison_to_other_types(self):
        class C(Class, Final):
            x = Member(int)

            def __eq__(self, other):
                return self.x == other.x

        @Entrypoint
        def isEqObj(c: C, o: object):
            return c == o

        self.assertTrue(isEqObj(C(x=10), C(x=10)))
        self.assertTrue(isEqObj(C(x=10), makeNamedTuple(x=10)))

    def test_compile_class_comparison_methods(self):
        class C(Class, Final):
            i = Member(int)
            s = Member(str)

            def __eq__(x, y):
                return x.i == y.i

            def __ne__(x, y):
                return x.i != y.i

            def __lt__(x, y):
                return x.s < y.s

            def __gt__(x, y):
                return x.s > y.s

            def __le__(x, y):
                return x.i <= y.i

            def __ge__(x, y):
                return x.i >= y.i

            def __hash__(self):
                return 123

        def f_eq(x: C, y: C):
            return x == C()

        def f_ne(x: C, y: C):
            return x != C()

        def f_lt(x: C, y: C):
            return x < C()

        def f_gt(x: C, y: C):
            return x > C()

        def f_le(x: C, y: C):
            return x <= C()

        def f_ge(x: C, y: C):
            return x >= C()

        def f_hash(x: C):
            return hash(x)

        values = [C(i=0), C(i=1), C(s="a"), C(s="b")]
        for f in [f_eq, f_ne, f_lt, f_gt, f_le, f_ge]:
            compiled_f = Compiled(f)
            for v1 in values:
                for v2 in values:
                    r1 = f(v1, v2)
                    r2 = compiled_f(v1, v2)
                    self.assertEqual(r1, r2)
        for f in [f_hash]:
            compiled_f = Compiled(f)
            for v in values:
                r1 = f(v)
                r2 = compiled_f(v)
                self.assertEqual(r1, r2)

    def test_compile_class_hash_special_value(self):
        class C(Class, Final):
            i = Member(int)

            def __hash__(self):
                return self.i

        def f_hash(x: C):
            return hash(x)

        c_hash = Compiled(f_hash)
        self.assertEqual(f_hash(C(i=123)), 123)
        self.assertEqual(c_hash(C(i=123)), 123)
        self.assertEqual(f_hash(C(i=-1)), -2)
        self.assertEqual(c_hash(C(i=-1)), -2)

    def test_compile_class_getsetitem(self):
        class C(Class, Final):
            d = Member(Dict(int, int))

            def __getitem__(self, i):
                if i not in self.d:
                    return i
                return self.d[i]

            def __setitem__(self, i, v):
                self.d[i] = v

        def f_getitem(c: C, i: int) -> int:
            return c[i]

        def f_setitem(c: C, i: int, v: int):
            c[i] = v

        c_getitem = Compiled(f_getitem)
        c_setitem = Compiled(f_setitem)

        c = C()
        c[123] = 7
        self.assertEqual(c[123], 7)
        for i in range(10, 20):
            self.assertEqual(f_getitem(c, i), i)
            self.assertEqual(c_getitem(c, i), i)
            f_setitem(c, i, i + 100)
            self.assertEqual(f_getitem(c, i), i + 100)
            self.assertEqual(c_getitem(c, i), i + 100)
            c_setitem(c, i, i + 200)
            self.assertEqual(f_getitem(c, i), i + 200)
            self.assertEqual(c_getitem(c, i), i + 200)

    def test_compile_class_float_conv(self):
        class C0(Class, Final):
            __int__ = lambda self: 123
            __float__ = lambda self: 1234.5

        class C(Class, Final):
            __int__ = lambda self: 123
            __float__ = lambda self: 1234.5

        def f(x: float):
            return x

        def g(x: int):
            return x

        c_f = Compiled(f)
        c_g = Compiled(g)
        with self.assertRaises(TypeError):
            c_f(C())
        with self.assertRaises(TypeError):
            c_f(C0())
        with self.assertRaises(TypeError):
            c_g(C())
        with self.assertRaises(TypeError):
            c_g(C0())

    def test_compile_class_missing_inplace_fallback(self):
        class ClassWithoutInplaceOp(Class, Final):
            s = Member(str)

            def __add__(self, other):
                self.s += " add" + other.s
                return self

            def __sub__(self, other):
                self.s += " sub" + other.s
                return self

            def __mul__(self, other):
                self.s += " mul" + other.s
                return self

            def __matmul__(self, other):
                self.s += " matmul" + other.s
                return self

            def __truediv__(self, other):
                self.s += " truediv" + other.s
                return self

            def __floordiv__(self, other):
                self.s += " floordiv" + other.s
                return self

            def __mod__(self, other):
                self.s += " mod" + other.s
                return self

            def __pow__(self, other):
                self.s += " pow" + other.s
                return self

            def __lshift__(self, other):
                self.s += " lshift" + other.s
                return self

            def __rshift__(self, other):
                self.s += " rshift" + other.s
                return self

            def __and__(self, other):
                self.s += " and" + other.s
                return self

            def __or__(self, other):
                self.s += " or" + other.s
                return self

            def __xor__(self, other):
                self.s += " xor" + other.s
                return self

        def inplace(x: ClassWithoutInplaceOp):
            x += ClassWithoutInplaceOp()
            x -= ClassWithoutInplaceOp()
            x *= ClassWithoutInplaceOp()
            x @= ClassWithoutInplaceOp()
            x /= ClassWithoutInplaceOp()
            x //= ClassWithoutInplaceOp()
            x %= ClassWithoutInplaceOp()
            x **= ClassWithoutInplaceOp()
            x <<= ClassWithoutInplaceOp()
            x >>= ClassWithoutInplaceOp()
            x &= ClassWithoutInplaceOp()
            x |= ClassWithoutInplaceOp()
            x ^= ClassWithoutInplaceOp()
            return x

        expected = ClassWithoutInplaceOp(
            s="start add sub mul matmul truediv floordiv mod pow lshift rshift and or xor"
        )
        v = ClassWithoutInplaceOp(s="start")
        r1 = inplace(v)
        self.assertEqual(r1.s, expected.s)
        v = ClassWithoutInplaceOp(s="start")
        r2 = Compiled(inplace)(v)
        self.assertEqual(r2.s, expected.s)

    def test_defining_float_doesnt_allow_implicit_conversion(self):
        aList = ListOf(float)()

        class C(Class, Final):
            def __float__(self):
                return 1.0

        with self.assertRaisesRegex(TypeError, "Cannot construct a new float from an instance of C"):
            aList.append(C())

        @Function
        def requiresAFloat(f: float):
            return f

        with self.assertRaisesRegex(TypeError, "annot find a valid overload"):
            requiresAFloat(C())

        @Compiled
        def tryToAppend(aList: ListOf(float)):
            c = C()
            aList.append(c)

        with self.assertRaisesRegex(TypeError, "Couldn't initialize type float from C"):
            tryToAppend(aList)

        @Compiled
        def tryToCallRequiresAFloat():
            requiresAFloat(C())

        with self.assertRaisesRegex(TypeError, "cannot find a valid overload"):
            tryToCallRequiresAFloat()

    def compileCheck(self, f):
        self.assertEqual(f(), Compiled(f)())

    def test_class_can_override_default_args(self):
        class B(Class):
            def f(self, x=10) -> int:
                return x

        class C(B):
            def f(self, x=20) -> int:
                return x + 100

        class D(C, Final):
            def f(self, x=30) -> int:
                return x + 200

        self.compileCheck(lambda: B().f())
        self.compileCheck(lambda: B().f(10))
        self.compileCheck(lambda: B().f(20))
        self.compileCheck(lambda: C().f())
        self.compileCheck(lambda: C().f(10))
        self.compileCheck(lambda: C().f(20))
        self.compileCheck(lambda: D().f())
        self.compileCheck(lambda: D().f(10))
        self.compileCheck(lambda: D().f(20))

    def test_class_methods_and_named_arguments(self):
        class B(Class):
            def f(self, **kwargs: int) -> int:
                return 0

        class C(B):
            def f(self, x, y) -> int:
                return x + y

            def f(self, x) -> int:  # noqa
                return x + 100

            def f(self) -> int:  # noqa
                return 1000

        self.assertEqual(B().f(x=10, y=20), 0)
        self.assertEqual(C().f(x=10, y=20), 30)

        self.compileCheck(lambda: B().f())
        self.compileCheck(lambda: C().f())

        self.compileCheck(lambda: B().f(x=10, y=20))
        self.compileCheck(lambda: C().f(x=10, y=20))

    def test_class_method_star_arg_type_dispatch(self):
        class B(Class):
            def f(self, *args: int, **kwargs: int) -> int:
                return 0

        self.assertEqual(B().f(1.5, x=2.5), 0)
        self.compileCheck(lambda: B().f(1.5, x=2.5))

    def test_class_method_star_arg_type_dispatch_unknown(self):
        class B(Class):
            def f(self, *args: int, **kwargs: int) -> str:
                return "int"

            def f(self, *args: float, **kwargs: float) -> str:  # noqa
                return "float"

        @Entrypoint
        def callBWith(o: object):
            return B().f(o)

        assert callBWith(1.0) == "float"
        assert callBWith(1) == "int"

    def test_class_methods_obey_star_arg_type_assignments(self):
        class B(Class):
            def f(self, *args, **kwargs) -> int:
                return 0

        class C(B):
            def f(self, *args: int, **kwargs: int) -> int:
                return 1

            def f(self, *args: str, **kwargs: str) -> int:  # noqa
                return 2

            def f(self, *args: int, **kwargs: str) -> int:  # noqa
                return 3

            def f(self, *args: str, **kwargs: int) -> int:  # noqa
                return 4

        self.assertEqual(C().f(), 1)
        self.assertEqual(C().f(1), 1)
        self.assertEqual(C().f("1"), 2)
        self.assertEqual(C().f(1, x="2"), 3)
        self.assertEqual(C().f("1", x=2), 4)

        for T in [B, C]:
            self.compileCheck(lambda: T().f())
            self.compileCheck(lambda: T().f(1, 2))
            self.compileCheck(lambda: T().f(x=1, y=2))
            self.compileCheck(lambda: T().f(1, y=2))
            self.compileCheck(lambda: T().f(1, y=2))

            self.compileCheck(lambda: T().f("1", "2"))
            self.compileCheck(lambda: T().f(x="1", y="2"))
            self.compileCheck(lambda: T().f("1", y="2"))
            self.compileCheck(lambda: T().f("1", y="2"))

            self.compileCheck(lambda: T().f("1", 2))
            self.compileCheck(lambda: T().f(x="1", y=2))
            self.compileCheck(lambda: T().f("1", y=2))
            self.compileCheck(lambda: T().f("1", y=2))

            self.compileCheck(lambda: T().f(1, "2"))
            self.compileCheck(lambda: T().f(x=1, y="2"))
            self.compileCheck(lambda: T().f(1, y="2"))
            self.compileCheck(lambda: T().f(1, y="2"))

    def test_class_with_global_closure_variables(self):
        def makeInt():
            return 10

        class BaseClass(Class):
            y = Member(int)

            def __init__(self):
                self.y = makeInt()

        class C(BaseClass, Final):
            x = Member(int)

            def __init__(self):
                BaseClass.__init__(self)
                self.x = makeInt()

            def g(self):
                return makeInt()

            def __len__(self):
                return makeInt()

            @Entrypoint
            @staticmethod
            def f():
                return makeInt()

        newC = C()
        self.assertEqual(newC.y, 10)

        self.assertEqual(C.f(), 10)

        @Entrypoint
        def makeAC():
            return C()

        self.assertEqual(makeAC().x, 10)

        @Entrypoint
        def callG():
            return C().g()

        self.assertEqual(callG(), 10)

        @Entrypoint
        def takeLen():
            return len(C())

        self.assertEqual(takeLen(), 10)

    def test_pass_child_to_function_expecting_base_and_kwargs(self):
        class X(Class):
            def f(self, **kwargs) -> str:
                return "kwargs"

        class Child(X, Final):
            def f(self, y) -> str:
                return "y"

        @Entrypoint
        def f(x: X):
            return x.f(y=10)

        @Entrypoint
        def doit():
            res = f(Child())
            return res

        @Entrypoint
        def done():
            doit()

        done()

    def test_compile_class_properties(self):
        class X(Class, Final):
            @property
            def value(self):
                return 10

        self.assertEqual(X().value, 10)

        @Entrypoint
        def callIt():
            return X().value

        self.assertEqual(callIt(), 10)

    def test_compile_class_properties_with_subclasses(self):
        class X(Class):
            @property
            def value(self):
                return 10

        class Child(X):
            @property
            def value(self):
                return 20

        self.assertEqual(X().value, 10)
        self.assertEqual(Child().value, 20)

        @Entrypoint
        def callIt(x: X):
            return x.value

        self.assertEqual(callIt(X()), 10)
        self.assertEqual(callIt(Child()), 20)

    def test_overloading_operators(self):
        class X(Class, Final):
            def __add__(self, other: int):
                return 1

            def __add__(self, other: float):  # noqa
                return 2

        self.assertEqual(X() + 1, 1)
        self.assertEqual(X() + 1.5, 2)

        @Entrypoint
        def add(x, y):
            return x + y

        self.assertEqual(add(X(), 1), 1)
        self.assertEqual(add(X(), 1.5), 2)

    def test_default_initialize_class_with_type_members(self):
        class C(Class, Final):
            t = Member(Value(float))

        assert C().t is float

        @Entrypoint
        def callIt():
            return C()

        assert callIt().t is float

    def test_can_compile_against_base_class_and_pass_child(self):
        class C(Class):
            t = Member(float)

            def f(self) -> float:
                return self.t

        class B(C):
            def f(self) -> float:
                return self.t + 1.0

        @Entrypoint
        def f(x: ListOf(C)):
            res = 0.0
            for i in x:
                res += i.f()
            return res

        assert f(ListOf(C)([C(t=10)])) == 10

        with CountCompilationsVisitor() as c:
            # the 'B' present here should trigger compilation and linking
            assert f(ListOf(C)([C(t=10), B(t=11)])) == 22

        assert c.count == 1

        with CountCompilationsVisitor() as c:
            assert f(ListOf(C)([C(t=10), B(t=11)])) == 22

        assert c.count == 0

        @Entrypoint
        def clearList(x: ListOf(C)):
            x.clear()

        aList = ListOf(C)([C(t=10), B(t=11)])
        theB = aList[1]
        assert _types.refcount(theB) == 2
        clearList(aList)
        assert _types.refcount(theB) == 1

    def test_class_member_assign_is_implicit_containers(self):
        class C(Class, Final):
            x = Member(ListOf(ListOf(int)))

        C().x = [ListOf(int)([1, 2, 3])]

        @Entrypoint
        def assignX(c, x):
            c.x = x

        assignX(C(), [ListOf(int)([1, 2, 3])])

    def test_kwarg_initialize_class(self):
        @Entrypoint
        def makeAClass(x, y, z):
            return AClass(x=x, z=z, y=y)

        assert makeAClass(1, 2, (3, 4)).z == TupleOf(int)((3, 4))

    def test_class_assign_invalid_member(self):
        @Entrypoint
        def assignInvalid(c):
            c.invalidMember = 10

        with self.assertRaisesRegex(AttributeError, "no attribute 'invalidMember'"):
            assignInvalid(AClass())

    def test_kwarg_initialize_class_invalid_arg(self):
        def makeAClass(x, y, z):
            return AClass(x=x, invalidMember=z, y=y)

        with self.assertRaisesRegex(AttributeError, "has no attribute 'invalidMember'"):
            makeAClass(1, 2, (3, 4))

        with self.assertRaisesRegex(AttributeError, "has no attribute 'invalidMember'"):
            Entrypoint(makeAClass)(1, 2, (3, 4))

    def test_kwarg_initialize_method(self):
        def initializeAClass(**kwargs):
            return AClassWithInit(**kwargs)

        def checkEqual(l, r):
            assert l.x == r.x
            assert l.y == r.y

        checkEqual(Entrypoint(initializeAClass)(), initializeAClass())
        checkEqual(Entrypoint(initializeAClass)(x=1), initializeAClass(x=1))
        checkEqual(Entrypoint(initializeAClass)(x=1, y=2), initializeAClass(x=1, y=2))

    def test_pointer_to_member(self):
        class C(Class, Final):
            x = Member(int)

        @Entrypoint
        def f(c):
            return pointerTo(c).x

        c = C()

        assert f(c) == pointerTo(c).x

    def test_type_of_subclass(self):
        class Base(Class):
            pass

        class Child(Base, Final):
            pass

        @Entrypoint
        def f(c: Base):
            return type(c)

        @Entrypoint
        def fKnowsChild(c: Child):
            return type(c)

        assert f(Base()) is Base
        assert f(Child()) is Child
        assert fKnowsChild(Child()) is Child

    def test_isinstance_on_subclass(self):
        class Base(Class):
            pass

        class Child(Base, Final):
            pass

        @Entrypoint
        def isinstanceCompiled(c: Base):
            return issubclass(type(c), Child)

        assert not isinstanceCompiled(Base())
        assert isinstanceCompiled(Child())

    def test_isinstance_on_two_subclasses(self):
        class Base(Class):
            pass

        class Child(Base, Final):
            pass

        @Entrypoint
        def isinstanceCompiled(l: Base, r: Base):
            return issubclass(type(l), type(r))

        assert isinstanceCompiled(Base(), Base())
        assert isinstanceCompiled(Child(), Base())
        assert isinstanceCompiled(Child(), Child())
        assert not isinstanceCompiled(Base(), Child())

    def test_downcast_in_compiled_code(self):
        class Base(Class):
            pass

        class Child(Base, Final):
            pass

        @Entrypoint
        def f(x: Child):
            pass

        @Entrypoint
        def g(x: Base):
            return f(x)

        g(Child())

        with self.assertRaises(TypeError):
            g(Base())

    def test_downcast_in_compiled_code_adjacent(self):
        class Base(Class):
            def baseFun(self) -> int:
                return 0

        class Base1(Base):
            def baseFun(self) -> int:
                return 1

        class Base2(Base):
            def baseFun(self) -> int:
                return 2

        class Child(Base1, Base2, Final):
            def baseFun(self) -> int:
                return 3

        @Entrypoint
        def callAsChild(x: Child):
            return x.baseFun()

        @Entrypoint
        def callAsBase1(x: Base1):
            return x.baseFun()

        @Entrypoint
        def callAsBase2(x: Base2):
            return x.baseFun()

        @Entrypoint
        def callAsBase(x: Base):
            return x.baseFun()

        @Entrypoint
        def callFromBase(f, x: Base):
            return f(x)

        # if we insist on converting to 'Child', only Child should work
        assert callFromBase(callAsChild, Child()) == 3

        with self.assertRaises(TypeError):
            assert callFromBase(callAsChild, Base())

        with self.assertRaises(TypeError):
            assert callFromBase(callAsChild, Base1())

        with self.assertRaises(TypeError):
            assert callFromBase(callAsChild, Base2())

        # if we insist on converting to 'Base1', only Child and Base1 should work
        assert callFromBase(callAsBase1, Child()) == 3
        assert callFromBase(callAsBase1, Base1()) == 1

        with self.assertRaises(TypeError):
            assert callFromBase(callAsBase1, Base())

        with self.assertRaises(TypeError):
            assert callFromBase(callAsBase1, Base2())

        # if we insist on converting to 'Base2', only Child and Base2 should work
        assert callFromBase(callAsBase2, Child()) == 3
        assert callFromBase(callAsBase2, Base2()) == 2

        with self.assertRaises(TypeError):
            assert callFromBase(callAsBase2, Base())

        with self.assertRaises(TypeError):
            assert callFromBase(callAsBase2, Base1())

        # if we insist on converting to 'Base', everything works
        assert callFromBase(callAsBase, Child()) == 3
        assert callFromBase(callAsBase, Base1()) == 1
        assert callFromBase(callAsBase, Base2()) == 2
        assert callFromBase(callAsBase, Base()) == 0

        @Entrypoint
        def callFromBase1(f, x: Base1):
            return f(x)

        assert callFromBase1(callAsBase, Base1()) == 1
        assert callFromBase1(callAsBase1, Base1()) == 1

        # if we know it as Base1, but it's a Child, we should be able
        # to cross call it
        assert callFromBase1(callAsBase2, Child()) == 3

    def test_class_nonempty(self):
        class CNonempty(Class, Final):
            x = Member(str, nonempty=True)

            def __init__(self):
                pass

        class CEmpty(Class, Final):
            x = Member(str)

            def __init__(self):
                pass

        @Entrypoint
        def callIt(X):
            return X()

        @Entrypoint
        def deleteX(x):
            del x.x

        @Entrypoint
        def getX(x):
            return x.x

        @Entrypoint
        def setX(x, xVal):
            x.x = xVal

        cN = callIt(CNonempty)
        cE = callIt(CEmpty)

        assert getX(cN) == ""
        assert getX(cE) == ""

        with self.assertRaisesRegex(Exception, "Attribute 'x' cannot be deleted"):
            deleteX(cN)
        deleteX(cE)

        with self.assertRaisesRegex(Exception, "Attribute 'x' is not initialized"):
            deleteX(cE)

        with self.assertRaisesRegex(Exception, "Attribute 'x' is not initialized"):
            getX(cE)

        setX(cE, "hihi")
        assert getX(cE) == "hihi"

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

        @Entrypoint
        def callIt(X):
            return X()

        callIt(CNonempty).x

        with self.assertRaisesRegex(Exception, "Attribute 'x' is not initialized"):
            callIt(CEmpty).x

    def test_class_hierarchy(self):
        class Base(Class):
            def f(self) -> int:
                return 0

            def g(self) -> int:
                return self.f()

        class C1(Base):
            def f(self) -> int:
                return 1

            def g(self) -> int:
                return self.f()

        def callG(x: Base):
            return x.g()

        assert callG(C1()) == Entrypoint(callG)(C1())

    def test_convert_refTo(self):
        class C(Class, Final):
            x = Member(int)

            def f(self):
                return self.x

        def f(c):
            return c.f()

        @Entrypoint
        def doIt():
            c = C(x=10)
            return f(refTo(c))

        assert doIt() == 10

    def test_call_overridden_fun(self):
        class C(Class):
            def getIt(self) -> int:
                return 0

        class B(C, Final):
            def getIt(self) -> int:
                return 1

        @Entrypoint
        def f(c: C):
            return c.getIt()

        f(C())

    def test_isinstance_perf(self):
        class C(Class):
            pass

        class D(C, Final):
            pass

        class E(C, Final):
            pass

        @Entrypoint
        def countIt(times, c: C):
            isC = 0
            isD = 0
            isE = 0
            isInt = 0

            assert typeKnownToCompiler(c) is C

            for _ in range(times):
                if isinstance(c, C):
                    isC += 1
                if isinstance(c, D):
                    isD += 1
                if isinstance(c, E):
                    isE += 1
                if isinstance(c, int):
                    isInt += 1

            return (isC, isD, isE, isInt)

        assert countIt(1, D()) == (1, 1, 0, 0)
        assert countIt(1, C()) == (1, 0, 0, 0)
        assert countIt(1, E()) == (1, 0, 1, 0)

        t0 = time.time()
        assert countIt(1000000, D())[1] == 1000000
        elapsed = time.time() - t0
        print(elapsed, " to do 1mm")

        # I get 1e-5 on my machine. This code doesn't have to hit
        # any functions to determine this because final classes have
        # an exactly known type.
        assert elapsed < 0.01

    def test_unresolved_forwards_in_class_type_signatures(self):
        UnresolvedForward = Forward("UnresolvedForward")

        class AClass(Class, Final):
            def f(self, other: UnresolvedForward):
                return "BAD"

        @Entrypoint
        def tryToCallF():
            return AClass().f(10)

        with self.assertRaisesRegex(TypeError, "unresolved forwards"):
            tryToCallF()

    def test_unresolved_forwards_in_function_type_signatures(self):
        UnresolvedForward = Forward("UnresolvedForward")

        @Function
        def f(x: UnresolvedForward):
            return "OK"

        @Entrypoint
        def tryToCallF():
            f(10)

        with self.assertRaisesRegex(TypeError, "has unresolved forwards"):
            tryToCallF()

    def test_could_be_instance_of(self):
        class A(Class):
            pass

        class B(Class):
            pass

        class C(A, B):
            pass

        # instances of A can be B and vice versa
        assert classCouldBeInstanceOf(A, B) == "Maybe"
        assert classCouldBeInstanceOf(B, A) == "Maybe"

        # instances of A and B can both be 'C'
        assert classCouldBeInstanceOf(A, C) == "Maybe"
        assert classCouldBeInstanceOf(B, C) == "Maybe"

        # instances of C are definitely A and B
        assert classCouldBeInstanceOf(C, A) == True  # noqa
        assert classCouldBeInstanceOf(C, B) == True  # noqa

        class AFinal(A, Final):
            pass

        class BFinal(B, Final):
            pass

        # instances of B cannot be AFinal, because we know exactly what base classes could
        # back a 'B'
        assert classCouldBeInstanceOf(A, BFinal) == False  # noqa

        class AWithMembers(A):
            x = Member(int)

        class BWithMembers(B):
            x = Member(int)

        # an instance of A could be a B with members, because its possible for
        # a subclass of BWithMembers to descend from A
        assert classCouldBeInstanceOf(A, BWithMembers) == "Maybe"

        # but if its an AWithMembers, then its not possible to create that subclass
        assert classCouldBeInstanceOf(AWithMembers, BWithMembers) == False  # noqa

        class AWithType(A):
            X = int

        class BWithType(B):
            X = float

        # an instance of A could be a BWithType
        assert classCouldBeInstanceOf(A, BWithType) == "Maybe"

        # but if its an AWithType, then its not possible to create that subclass
        assert classCouldBeInstanceOf(AWithType, BWithType) == False  # noqa

        # its possible to descend from two separate type functions that don't declare
        # any classmembers
        @TypeFunction
        def TF1(X):
            class TF1_(TF1):
                __name__ = f"TF2({X})"

                def f(self) -> X:
                    return X()

            return TF1_

        @TypeFunction
        def TF2(X):
            class TF2_(TF2):
                __name__ = f"TF2({X})"

                def g(self) -> X:
                    return X()

            return TF2_

        assert classCouldBeInstanceOf(TF1(int), TF2(float))

        class ActualSubclass(TF1(int), TF2(float)):
            pass

    def test_super(self):
        class B(Class):
            def f(self, x):
                return x

        class C(B):
            def f(self, x):
                return super().f(x) + 1

        @Entrypoint
        def callIt(c, x):
            return c.f(x)

        assert callIt(C(), 10) == 11

    def test_super_with_diamond(self):
        class B(Class):
            def f(self, x: ListOf(str)):
                x.append('B')

        class C(B):
            def f(self, x):
                x.append('C')
                super().f(x)

        class D(B):
            def f(self, x):
                x.append('D')
                super().f(x)

        class E(C, D):
            def f(self, x):
                x.append('E')
                super().f(x)

        lst = ListOf(str)()
        E().f(lst)

        assert lst == ['E', 'C', 'D', 'B']

        @Entrypoint
        def callIt(c, x):
            return c.f(x)

        lst.resize(0)
        callIt(E(), lst)

        assert lst == ['E', 'C', 'D', 'B']

    def test_super_with_no_superclass(self):
        class B(Class):
            def f(self):
                return super().f()

        @Entrypoint
        def callIt(c):
            return c.f()

        with self.assertRaisesRegex(AttributeError, r"'super' object has no attribute 'f'"):
            B().f()

        with self.assertRaisesRegex(AttributeError, r"'super' object has no attribute 'f'"):
            callIt(B())

    def test_super_in_init(self):
        class B(Class):
            x = Member(int)

            def __init__(self, x):
                self.x = x

        class C(B):
            y = Member(int)

            def __init__(self, x, y):
                super().__init__(x)
                self.y = y

        def check():
            c = C(10, 11)
            assert c.x == 10
            assert c.y == 11

        check()
        Entrypoint(check)()

    def test_compile_classmethods(self):
        class Normal:
            @classmethod
            def f(cls):
                return cls

        assert Normal.f() is Normal
        assert Normal().f() is Normal

        class A(Class):
            @classmethod
            def f(cls):
                return cls

        def checkIt():
            assert A.f() is A
            assert A().f() is A

        checkIt()
        Entrypoint(checkIt)()

    def test_compile_classmethod_on_subclass_multi(self):
        class B(Class):
            @classmethod
            def f(cls, x: int) -> int:
                return 1

            @classmethod
            def f(cls, x: float) -> float:  # noqa
                return 1.5

        class C(B):
            @classmethod
            def f(cls, x: int) -> int:
                return 2

        @Entrypoint
        def callIt(b: B, x: OneOf(int, float)):
            return b.f(x)

        assert callIt(B(), 1) == 1
        assert callIt(B(), 1.0) == 1.5
        assert callIt(C(), 1) == 2
        assert callIt(C(), 1.0) == 1.5

    def test_compile_staticmethod_on_subclass_multi(self):
        class B(Class):
            @staticmethod
            def f(x: int) -> int:
                return 1

            @staticmethod
            def f(x: float) -> float:  # noqa
                return 1.5

        class C(B):
            @staticmethod
            def f(x: int) -> int:
                return 2

        @Entrypoint
        def callIt(b: B, x: OneOf(int, float)):
            return b.f(x)

        assert callIt(B(), 1) == 1
        assert callIt(B(), 1.0) == 1.5
        assert callIt(C(), 1) == 2
        assert callIt(C(), 1.0) == 1.5

    def test_compile_classmethod_on_subclass(self):
        class BPy(object):
            @classmethod
            def f(cls, x) -> int:
                return 1

        class CPy(BPy):
            @classmethod
            def f(cls, x) -> int:
                return 2

        class DPy(CPy):
            @classmethod
            def f(cls, x) -> int:
                return 3

        def callItPy(b, times):
            x = 0
            for i in range(times):
                x += b.f(i)
            return x

        t0 = time.time()
        callItPy(BPy(), 1000000)
        callItPy(CPy(), 1000000)
        pyTime = time.time() - t0

        class B(Class):
            @classmethod
            def f(cls, x) -> int:
                return 1

        class C(B):
            @classmethod
            def f(cls, x) -> int:
                return 2

        class D(C, Final):
            @classmethod
            def f(cls, x) -> int:
                return 3

        def callItAs(T):
            @Entrypoint
            def callIt(b: T, times: int):
                x = 0
                for i in range(times):
                    x += b.f(i)
                return x

            return callIt

        def callItAsWithType(T):
            @Entrypoint
            def callIt(b: SubclassOf(T), times: int):
                x = 0
                for i in range(times):
                    x += b.f(i)
                return x

            return callIt

        callItAs(B)(B(), 1)
        callItAs(B)(C(), 1)
        callItAsWithType(B)(B, 1)
        callItAsWithType(B)(C, 1)

        t0 = time.time()
        assert callItAs(B)(B(), 1000000) == 1000000
        assert callItAs(B)(C(), 1000000) == 2000000
        compiledTime = time.time() - t0

        t0 = time.time()
        assert callItAsWithType(B)(B, 1000000) == 1000000
        assert callItAsWithType(B)(C, 1000000) == 2000000
        compiledTypeTime = time.time() - t0

        speedup = pyTime / compiledTime
        speedupType = pyTime / compiledTypeTime

        print("compiled virtual dispatch on instances ", speedup, "times faster")
        print("compiled virtual dispatch on type ", speedupType, "times faster")

    def test_compile_staticmethod_on_subclass(self):
        class B(Class):
            @staticmethod
            def f(x) -> int:
                return 1

        class C(B):
            @staticmethod
            def f(x) -> int:
                return 2

        class D(C, Final):
            @staticmethod
            def f(x) -> int:
                return 3

        def callItAs(T):
            @Entrypoint
            def callIt(b: T, times: int):
                x = 0
                for i in range(times):
                    x += b.f(i)
                return x

            return callIt

        assert B().f(10) == 1

        assert callItAs(B)(B(), 1000000) == 1000000
        assert callItAs(B)(C(), 1000000) == 2000000

    def test_class_dispatch_on_type(self):
        class B(Class):
            @classmethod
            def f(cls) -> int:
                return 0

        @Entrypoint
        def call(T: SubclassOf(B)):
            return T.f()

        assert call.resultTypeFor(SubclassOf(B)).typeRepresentation is int
        assert call(B) == 0

    def test_interpreter_can_see_class_members(self):
        class B(Class):
            X = int

        @Entrypoint
        def callType(T: SubclassOf(B)):
            return T.X

        @Entrypoint
        def callInst(T: B):
            return T.X

        assert callType.resultTypeFor(SubclassOf(B)).typeRepresentation.Value is int
        assert callInst.resultTypeFor(B).typeRepresentation.Value is int

    def test_class_init_is_virtual(self):
        B = Forward("B")

        @B.define
        class B(Class):
            x = Member(int)

            def __init__(self, x):
                self.x = x

            def f(self) -> int:
                return self.x

            @classmethod
            def makeOne(cls, x) -> B:
                return cls(x)

        class C(B):
            def __init__(self, x):
                super().__init__(x + 10)

            def f(self):
                return self.x + 100

        @Entrypoint
        def constructOne(T: SubclassOf(B), x, times):
            res = 0
            for i in range(times):
                res += T.makeOne(x).f()
            return res

        assert constructOne(C, 0, 1) == C(0).f()
        assert constructOne.resultTypeFor(SubclassOf(B), int, int).typeRepresentation is int

        t0 = time.time()
        assert constructOne(C, 0, 1000000) == C(0).f() * 1000000
        print(time.time() - t0)
