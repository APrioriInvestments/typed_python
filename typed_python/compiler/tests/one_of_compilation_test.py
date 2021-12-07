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
import typed_python

from typed_python import (
    OneOf, TupleOf, Forward, ConstDict, Class, Final, Member,
    ListOf, Compiled, Entrypoint, NamedTuple, UInt16,
    typeKnownToCompiler
)
from typed_python import Value as ValueType
import typed_python._types as _types
import unittest


typeWrapper = lambda t: typed_python.compiler.python_object_representation.typedPythonTypeToTypeWrapper(t)


# a simple recursive 'value' type for testing
Value = Forward("Value")


Value = Value.define(
    OneOf(
        None,
        False,
        True,
        float,
        int,
        str,
        bytes,
        ConstDict(Value, Value),
        TupleOf(Value)
    )
)

someValues = [
    None,
    False,
    True,
    0.0, 1.0,
    0, 1,
    "hi",
    b"bye",
    Value({'hi': 'bye'}),
    Value((1, 2, 3))
]


class ClassA(Class, Final):
    x = Member(int)

    def f(self, y):
        return self.x + y


class ClassB(Class, Final):
    x = Member(float)

    def f(self, y):
        return self.x - y


class TestOneOfCompilation(unittest.TestCase):
    def test_oneof_wrapper_fast_is_check(self):
        assert typeWrapper(OneOf(int, float))._simpleNoneCheckIndex() == -1
        assert typeWrapper(OneOf(int, None))._simpleNoneCheckIndex() == 1
        assert typeWrapper(OneOf(int, type(None)))._simpleNoneCheckIndex() == 1
        assert typeWrapper(OneOf(None, int))._simpleNoneCheckIndex() == 0
        assert typeWrapper(OneOf(None, object))._simpleNoneCheckIndex() is None

    def test_one_of_basic(self):
        @Compiled
        def f(x: OneOf(int, float)) -> OneOf(int, float):
            return x

        self.assertEqual(f(10), 10)
        self.assertEqual(f(10.2), 10.2)

    def test_one_of_with_refcounts(self):
        @Compiled
        def f(x: OneOf(None, TupleOf(int))) -> OneOf(None, TupleOf(int)):
            y = x
            return y

        self.assertIs(f(None), None)

        aTup = TupleOf(int)((1, 2, 3))
        self.assertEqual(f(aTup), aTup)

        self.assertEqual(_types.refcount(aTup), 1)

    def test_one_of_binop_stays_dual(self):
        @Compiled
        def f(x: OneOf(int, float), y: int) -> OneOf(int, float):
            return x + y

        def check(x, y):
            self.assertIs(type(f(x, y)), type(x+y))
            self.assertEqual(f(x, y), x+y)

        things = [0, 1, 2, 0.0, 1.0, 2.0]
        for a in things:
            for b in [0, 1, 2]:
                check(a, b)

    def test_one_of_binop_converges(self):
        @Compiled
        def f(x: OneOf(int, float), y: float) -> float:
            return x + y

        def check(x, y):
            self.assertIs(type(f(x, y)), type(x+y))
            self.assertEqual(f(x, y), x+y)

        things = [0, 1, 2, 0.0, 1.0, 2.0]
        for a in things:
            for b in [0.0, 1.0, 2.0]:
                check(a, b)

    def test_one_of_binop_rhs(self):
        @Compiled
        def f(x: int, y: OneOf(int, float)) -> OneOf(int, float):
            return x - y

        def check(x, y):
            self.assertIs(type(f(x, y)), type(x + y))
            self.assertEqual(f(x, y), x - y)

        things = [0, 1, 2, 0.0, 1.0, 2.0]

        for a in [0, 1, 2]:
            for b in things:
                check(a, b)

    def test_one_of_dual_binop(self):
        @Compiled
        def f(x: OneOf(int, float), y: OneOf(int, float)) -> OneOf(int, float):
            return x + y

        def check(x, y):
            self.assertIs(type(f(x, y)), type(x+y))
            self.assertEqual(f(x, y), x+y)

        things = [0, 1, 2, 0.0, 1.0, 2.0]
        for a in things:
            for b in things:
                check(a, b)

    def SKIPtest_one_of_dual_binop_power(self):
        @Compiled
        def f(x: OneOf(int, float), y: OneOf(int, float)) -> OneOf(int, float):
            return x ** y

        def check(x, y):
            self.assertIs(type(f(x, y)), type(x**y))
            self.assertEqual(f(x, y), x**y)

        things = [0, 1, 2, 0.0, 1.0, 2.0]
        for a in things:
            for b in things:
                check(a, b)

    def test_one_of_downcast_to_primitive(self):
        @Compiled
        def f(x: OneOf(int, float)) -> int:
            return x

        self.assertEqual(f(10), 10)
        self.assertEqual(f(10.5), 10)

    def test_one_of_downcast_to_oneof(self):
        @Compiled
        def f(x: OneOf(int, float, None)) -> OneOf(int, None):
            return x

        self.assertEqual(f(10), 10)
        self.assertIs(f(None), None)
        self.assertEqual(f(10.5), 10)

    def test_one_of_upcast(self):
        @Compiled
        def f(x: OneOf(int, None)) -> OneOf(int, float, None):
            return x

        self.assertEqual(f(10), 10)
        self.assertIs(f(None), None)

    def test_one_of_returning(self):
        @Compiled
        def f(x: OneOf(None, int, float)) -> OneOf(None, int, float):
            y = x
            return y

        self.assertEqual(f(10), 10)
        self.assertEqual(f(10.5), 10.5)
        self.assertIs(f(None), None)

    def test_value_equal(self):
        @Compiled
        def f(x: Value, y: Value) -> bool:
            return x == y

        for val1 in someValues:
            for val2 in someValues:
                self.assertEqual(val1 == val2, f(val1, val2), (val1, val2))

    def test_value_types(self):
        @Compiled
        def f(x: ValueType(1), y: ValueType(2)):
            return x + y

        self.assertEqual(f(1, 2), 3)

    def test_convert_bool_to_value(self):
        @Compiled
        def f(x: bool) -> Value:
            return x

        self.assertEqual(f(False), False)
        self.assertEqual(f(True), True)

    def test_value_get_method(self):
        @Compiled
        def getHi(x: Value) -> Value:
            return x.get("hi")

        self.assertEqual(getHi({}), None)
        self.assertEqual(getHi({'hi': 20}), 20)

    def test_convert_ordering(self):
        # we should always pick the int if we can
        @Compiled
        def f(x: int) -> OneOf(int, float):
            return x

        self.assertEqual(f(1), 1)
        self.assertIs(type(f(1)), int)

        @Compiled
        def f2(x: int) -> OneOf(float, int):
            return x

        self.assertEqual(f2(1), 1)
        self.assertIs(type(f2(1)), int)

        # but if there's no exact conversion form,
        # we will prefer whichever one comes first
        @Compiled
        def f3(x: float) -> OneOf(int, str):
            return x

        self.assertIs(type(f3(1.5)), int)

        @Compiled
        def f4(x: float) -> OneOf(str, int):
            return x

        # it'll pick '1' because we won't just convert something
        # to a string just because of the return type. That would
        # require an explicit cast.
        self.assertEqual(f4(1.5), 1)

    def test_oneof_method_dispatch(self):
        @Compiled
        def f(c: OneOf(ClassA, ClassB), y: OneOf(int, float)):
            return c.f(y)

        anA = ClassA(x=10)
        aB = ClassB(x=10.5)

        self.assertEqual(f(anA, 0.5), anA.f(0.5))
        self.assertEqual(f(anA, 1), anA.f(1))
        self.assertEqual(f(aB, 0.5), aB.f(0.5))
        self.assertEqual(f(aB, 1), aB.f(1))

    def test_oneof_attribute_dispatch(self):
        @Compiled
        def f(c: OneOf(ClassA, ClassB)):
            return c.x

        anA = ClassA(x=10)
        aB = ClassB(x=10.5)

        self.assertEqual(f(anA), anA.x)
        self.assertEqual(f(aB), aB.x)

    def test_oneof_getitem(self):
        @Compiled
        def f(c: OneOf(TupleOf(int), ListOf(float))):
            return c[0]

        self.assertEqual(f(TupleOf(int)((1,))), 1)
        self.assertEqual(f(ListOf(float)((1.5,))), 1.5)

    def test_oneof_to_bool(self):
        @Compiled
        def f(c: OneOf(int, float)):
            if c:
                return "yes"
            return "no"

        self.assertEqual(f(1), "yes")
        self.assertEqual(f(1.5), "yes")
        self.assertEqual(f(0), "no")
        self.assertEqual(f(0.0), "no")

    def test_oneof_round(self):
        def f(c: OneOf(int, float)):
            return round(c)

        fComp = Compiled(f)

        for thing in [0, 0.0, 1, 1.5]:
            self.assertEqual(f(thing), fComp(thing))

    def test_len_of_none_or_listof(self):
        @Entrypoint
        def iterate(x: OneOf(None, ListOf(int))):
            res = ListOf(int)()
            for i in range(len(x)):
                res.append(x[i])
            return res

        self.assertEqual(iterate(ListOf(int)([1, 2, 3])), [1, 2, 3])

    def test_oneof_call(self):
        @Compiled
        def f(i: int):
            if i % 2:
                T = str
            else:
                T = float

            return T(i)

        self.assertEqual(f(1), "1")
        self.assertEqual(f(2), 2.0)

    def test_named_tuple_with_oneof(self):
        @Entrypoint
        def makeNT(x: NamedTuple(x=OneOf("A", "B"))):  # noqa
            return NamedTuple(x=OneOf("A", "B"))(x=x.x)

        self.assertEqual(makeNT(NamedTuple(x=OneOf("A", "B"))(x="A")).x, "A")

    def test_oneof_in_return_types(self):
        class A(Class, Final):
            @staticmethod
            @Entrypoint
            def g(s: float) -> OneOf(None, float):
                return 0.0

        @Entrypoint
        def do():
            abc = OneOf(None, float)(90.0)
            A.g(abc)

        do()

    def test_make_oneof(self):
        @Entrypoint
        def f(x):
            return OneOf("A", "B")(x)

        assert f("A") == "A"

        with self.assertRaises(TypeError):
            f("C")

        with self.assertRaises(TypeError):
            f(10)

    def test_oneof_promotion(self):
        @Entrypoint
        def f(x: OneOf("A", "B")) -> str: # noqa
            return x

        assert f("A") == "A"
        assert f("B") == "B"

    def test_oneof_promotion_heterogeneous(self):
        @Entrypoint
        def f(x: OneOf("A", 10)) -> OneOf(str, int): # noqa
            return x

        assert f("A") == "A"
        assert f(10) == 10

    def test_operations_on_oneof_values(self):
        @Entrypoint
        def oneof_concat(x: OneOf('A', 'B')): # noqa
            return x + x

        @Entrypoint
        def oneof_getitem(x: OneOf('AB', 'CD'), i: int): # noqa
            return x[i]

        @Entrypoint
        def oneof_ord(x: OneOf('A', 'B')): # noqa
            return ord(x[0])

        @Entrypoint
        def oneof_isalpha(x: OneOf('A', '1')): # noqa
            return x.isalpha()

        @Entrypoint
        def oneof_abs(x: OneOf(123.4, -234.5)): # noqa
            return abs(x)

        @Entrypoint
        def oneof_not(x: OneOf(0, 1)):
            return not x

        self.assertEqual(oneof_concat('A'), 'AA')
        self.assertEqual(oneof_concat('B'), 'BB')
        self.assertEqual(oneof_getitem('AB', 0), 'A')
        self.assertEqual(oneof_getitem('AB', 1), 'B')
        self.assertEqual(oneof_getitem('CD', 0), 'C')
        self.assertEqual(oneof_getitem('CD', 1), 'D')
        self.assertEqual(oneof_ord('A'), 65)
        self.assertEqual(oneof_ord('B'), 66)
        self.assertEqual(oneof_isalpha('A'), True)
        self.assertEqual(oneof_isalpha('1'), False)
        self.assertEqual(oneof_abs(123.4), 123.4)
        self.assertEqual(oneof_abs(-234.5), 234.5)
        self.assertEqual(oneof_not(0), True)
        self.assertEqual(oneof_not(1), False)

    def test_oneof_setitem(self):
        @Entrypoint
        def setItem(x: OneOf(None, ListOf(int)), i, y):
            x[i] = y

        @Entrypoint
        def getItem(x: OneOf(None, ListOf(int)), i):
            return x[i]

        @Entrypoint
        def getSlice(x: OneOf(None, ListOf(int)), i, j):
            return x[i:j]

        aList = ListOf(int)([0])

        setItem(aList, 0, 10)

        assert getItem(aList, 0) == 10

        assert getSlice(aList, 0, 0) == ListOf(int)()
        assert getSlice(aList, 0, 1) == aList

    def test_oneof_resize_works_with_fcall(self):
        from typed_python import UInt16

        def callResize(s: ListOf(UInt16), ct):
            s.resize(ct)

        @Entrypoint
        def resizeOneof(sz):
            x = ListOf(OneOf(None, ListOf(UInt16)))()
            x.resize(10)

            for ix in range(10):
                i = x[ix]

                if not i:
                    i = ListOf(UInt16)()
                    x[ix] = i

                    callResize(i, sz)

            return x[0]

        assert len(resizeOneof(10)) == 10

    def test_oneof_resize_works_with_inline_call(self):
        @Entrypoint
        def resizeOneof(sz):
            x = ListOf(OneOf(None, ListOf(UInt16)))()
            x.resize(10)

            for ix in range(10):
                i = x[ix]

                if not i:
                    i = ListOf(UInt16)()
                    x[ix] = i

                    i.resize(sz)

            return x[0]

        assert len(resizeOneof(10)) == 10

    def test_assign_to_oneof_preserves_ref(self):
        @Entrypoint
        def preservesReference():
            x = ListOf(OneOf(None, ListOf(UInt16)))()
            x.resize(10)

            y = ListOf(UInt16)()

            x[5] = y

            y.resize(10)

            assert len(x[5]) == 10

    def test_returning_oneof_preserves_reference(self):
        @Entrypoint
        def returnAsNotNone(x: OneOf(None, ListOf(UInt16))) -> ListOf(UInt16):
            if x is None:
                return ListOf(UInt16)()
            return x

        @Entrypoint
        def preservesReference():
            x = ListOf(OneOf(None, ListOf(UInt16)))()
            x.resize(10)

            y = ListOf(UInt16)()

            x[5] = y

            returnAsNotNone(x[5]).resize(10)

            assert len(x[5]) == 10
            assert len(y) == 10

    def test_explicitly_converting_to_oneof_works(self):
        @Entrypoint
        def check():
            intList = ListOf(int)()
            floatList = ListOf(float)()

            # OneOf never wants to duplicate its arguments if it can
            # avoid it. In this case, the type is correct, so it
            # simply returns the relevant list.
            oneOfList = OneOf(None, ListOf(int))(intList)
            oneOfList.append(10)
            assert len(intList) == 1

            # if the type is different, however, it has to do a coersion
            # which requires a new list object
            oneOfList2 = OneOf(None, ListOf(int))(floatList)
            oneOfList2.append(10)
            assert len(floatList) == 0

        check()

    def test_oneof_binary_ops_dont_duplicate(self):
        class AddMakesTuple:
            def __init__(self, x):
                self.x = x

            def __add__(self, other):
                return (self, other)

        @Entrypoint
        def check(a1, a2):
            aList = ListOf(OneOf(ListOf(int), AddMakesTuple))()
            aList.append(a1)
            aList.append(a2)

            return aList[0] + aList[1]

        amt = AddMakesTuple(1)
        lst = ListOf(int)()

        aL = check(amt, lst)

        assert isinstance(aL, tuple)
        assert aL[0] is amt

        lst.append(10)
        assert aL[1] == lst

    def test_compile_str_on_oneof(self):
        @Entrypoint
        def callStr(x: OneOf(ZeroDivisionError, str)):
            return str(x)

        assert callStr(ZeroDivisionError()) == str(ZeroDivisionError())

    def test_convert_oneof_or_none_to_index(self):
        @Entrypoint
        def index(l: ListOf(int), y: OneOf(None, int)):
            return l[y]

        assert index([1, 2, 3], 2) == 3

        with self.assertRaisesRegex(TypeError, "Can't take instance of type 'NoneType'"):
            assert index([1, 2, 3], None) == 3

    def test_check_if_oneof_is_none(self):
        @Entrypoint
        def strTranslate(x: str, table):
            accumulator = ListOf(str)()
            for c in x:
                t = c

                try:
                    t = table.__getitem__(ord(c))
                except LookupError:
                    pass

                if t is not None:
                    accumulator.append(t)

            return ''.join(accumulator)

        strTranslate("hi", {})

    def test_oneof_method_call(self):
        @Entrypoint
        def append(x: OneOf(ListOf(int), ListOf(float))):
            x.append(1)

        l = ListOf(int)([1, 2, 3])
        append(l)
        assert len(l) == 4

    def test_split_on_oneof_type(self):
        @Entrypoint
        def split(x: Value):
            return typeKnownToCompiler(x.split)

        assert issubclass(split("HI"), OneOf)

    def test_string_split_on_oneof_with_constant(self):
        @Entrypoint
        def split(x: Value):
            return x.split("HI")

        assert split("AHIB") == ["A", "B"]

        with self.assertRaisesRegex(TypeError, "Can.t call bytes.split"):
            split(b"AHIB")

    def test_string_split_on_oneof_retains_type(self):
        @Entrypoint
        def split(x: Value):
            res = x.split("HI")

            return typeKnownToCompiler(res)

        assert split("AHIB") == ListOf(str)

    def test_len_of_oneof(self):
        @Entrypoint
        def lenOf(x: Value):
            return len(x)

        assert lenOf("AHIB") == 4

    def test_string_split_on_oneof_retains_type_2(self):
        @Entrypoint
        def split(x: Value, y: Value):
            res = x.split(y)

            return typeKnownToCompiler(res)

        assert split("AHIB", "HI") == list
        assert split(b"AHIB", b"HI") == list

    def test_isinstance_convert_overlapping(self):
        @Entrypoint
        def checkIt(x: OneOf(1, 2, 3, int)):
            # this verifies that the compiler understands that just because something
            # is an 'int' doesn't mean it's the 4th slot of the OneOf.
            if isinstance(x, int):
                return x

            return 10

        assert checkIt(2) == 2

    def test_call_function_with_none_and_ifcheck(self):
        @Entrypoint
        def checkIt(x, y: object):
            if x is None:
                x = y

        checkIt(None, float)

    def test_call_type_on_oneof(self):
        @Entrypoint
        def checkIt(x: OneOf(int, float)):
            if type(x) is int:
                return "int"
            return "float"

        assert checkIt(1) == "int"
        assert checkIt(1.0) == "float"
