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

from typed_python import Function, OneOf, TupleOf, Forward, ConstDict, Class, Member
from typed_python import Value as ValueType
import typed_python._types as _types
from nativepython.runtime import Runtime
import unittest


def Compiled(f):
    f = Function(f)
    return Runtime.singleton().compile(f)


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


class ClassA(Class):
    x = Member(int)

    def f(self, y):
        return self.x + y


class ClassB(Class):
    x = Member(float)

    def f(self, y):
        return self.x - y


class TestOneOfOfCompilation(unittest.TestCase):
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

        self.assertEqual(f4(1.5), "1.5")

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
