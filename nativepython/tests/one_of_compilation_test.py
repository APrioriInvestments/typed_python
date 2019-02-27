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

from typed_python import *
import typed_python._types as _types
from nativepython.runtime import Runtime
import unittest


def Compiled(f):
    f = Function(f)
    return Runtime.singleton().compile(f)


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
            return x + y

        def check(x, y):
            self.assertIs(type(f(x, y)), type(x+y))
            self.assertEqual(f(x, y), x+y)

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

    def test_one_of_dual_binop_power(self):
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
        with self.assertRaises(Exception):
            f(10.5)

    def test_one_of_downcast_to_oneof(self):
        @Compiled
        def f(x: OneOf(int, float, None)) -> OneOf(int, None):
            return x

        self.assertEqual(f(10), 10)
        self.assertIs(f(None), None)
        with self.assertRaises(Exception):
            f(10.5)

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
