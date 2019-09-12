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

from typed_python import TypeFunction, Function, Alternative, Forward
import typed_python._types as _types
from nativepython.runtime import Runtime
import unittest
import time


def Compiled(f):
    f = Function(f)
    return Runtime.singleton().compile(f)


class TestAlternativeCompilation(unittest.TestCase):
    def test_simple_alternative_passing(self):
        Simple = Alternative("Simple", A={}, B={}, C={})

        @Compiled
        def f(s: Simple):
            y = s
            return y

        self.assertEqual(f(Simple.A()), Simple.A())
        self.assertEqual(f(Simple.B()), Simple.B())
        self.assertEqual(f(Simple.C()), Simple.C())

    def test_complex_alternative_passing(self):
        Complex = Forward("Complex")
        Complex = Complex.define(Alternative(
            "Complex",
            A={'a': str, 'b': int},
            B={'a': str, 'c': int},
            C={'a': str, 'd': Complex}
        ))

        c = Complex.A(a="hi", b=20)
        c2 = Complex.C(a="hi", d=c)

        @Compiled
        def f(c: Complex):
            y = c
            return y

        self.assertEqual(f(c), c)
        self.assertEqual(f(c2), c2)

        self.assertEqual(_types.refcount(c), 2)
        self.assertEqual(_types.refcount(c2), 1)

    def test_construct_alternative(self):
        A = Alternative("A", X={'x': int})

        @Compiled
        def f():
            return A.X(x=10)

        self.assertTrue(f().matches.X)
        self.assertEqual(f().x, 10)

    def test_alternative_matches(self):
        A = Alternative("A", X={'x': int}, Y={'x': int})

        @Compiled
        def f(x: A):
            return x.matches.X

        self.assertTrue(f(A.X()))
        self.assertFalse(f(A.Y()))

    def test_alternative_member_homogenous(self):
        A = Alternative("A", X={'x': int}, Y={'x': int})

        @Compiled
        def f(x: A):
            return x.x

        self.assertEqual(f(A.X(x=10)), 10)
        self.assertEqual(f(A.Y(x=10)), 10)

    def test_alternative_member_diverse(self):
        A = Alternative("A", X={'x': int}, Y={'x': float})

        @Compiled
        def f(x: A):
            return x.x

        self.assertEqual(f(A.X(x=10)), 10)
        self.assertEqual(f(A.Y(x=10.5)), 10.5)

    def test_alternative_member_distinct(self):
        A = Alternative("A", X={'x': int}, Y={'y': float})

        @Compiled
        def f(x: A):
            if x.matches.X:
                return x.x
            if x.matches.Y:
                return x.y

        self.assertEqual(f(A.X(x=10)), 10)
        self.assertEqual(f(A.Y(y=10.5)), 10.5)

    def test_matching_recursively(self):
        @TypeFunction
        def Tree(T):
            TreeType = Forward("TreeType")
            TreeType = TreeType.define(Alternative(
                "Tree",
                Leaf={'value': T},
                Node={'left': TreeType, 'right': TreeType}
            ))
            return TreeType

        def treeSum(x: Tree(int)):
            matches = x.matches.Leaf
            if matches:
                return x.value
            if x.matches.Node:
                return treeSum(x.left) + treeSum(x.right)
            return 0

        def buildTree(depth: int, offset: int) -> Tree(int):
            if depth > 0:
                return Tree(int).Node(
                    left=buildTree(depth-1, offset),
                    right=buildTree(depth-1, offset+1),
                )
            return Tree(int).Leaf(value=offset)

        aTree = Compiled(buildTree)(15, 0)
        treeSumCompiled = Compiled(treeSum)

        t0 = time.time()
        sum = treeSum(aTree)
        t1 = time.time()
        sumCompiled = treeSumCompiled(aTree)
        t2 = time.time()

        self.assertEqual(sum, sumCompiled)
        speedup = (t1-t0)/(t2-t1)
        self.assertGreater(speedup, 20)  # I get about 50

    def test_compile_alternative_magic_methods(self):
        A = Alternative("A", a={'a': int}, b={'b': str},
                        extramethod=lambda self: self.Name,
                        __bool__=lambda self: False,
                        __str__=lambda self: "my str",
                        __repr__=lambda self: "my repr",
                        __call__=lambda self, i: "my call",
                        __len__=lambda self: 42,
                        __contains__=lambda self, item: item == 1,
                        # __contains__=lambda self, item: not item,  # TODO: why doesn't this compile?

                        __add__=lambda self, other: A.b("add"),
                        __sub__=lambda self, other: A.b("sub"),
                        __mul__=lambda self, other: A.b("mul"),
                        __matmul__=lambda self, other: A.b("matmul"),
                        __truediv__=lambda self, other: A.b("truediv"),
                        __floordiv__=lambda self, other: A.b("floordiv"),
                        __mod__=lambda self, other: A.b("mod"),
                        __divmod__=lambda self, other: A.b("divmod"),
                        __pow__=lambda self, other: A.b("pow"),
                        __lshift__=lambda self, other: A.b("lshift"),
                        __rshift__=lambda self, other: A.b("rshift"),
                        __and__=lambda self, other: A.b("and"),
                        __or__=lambda self, other: A.b("or"),
                        __xor__=lambda self, other: A.b("xor"),

                        __neg__=lambda self: A.b("neg"),
                        __pos__=lambda self: A.b("pos"),
                        __invert__=lambda self: A.b("invert"),

                        __abs__=lambda self: A.b("abs"),
                        )

        def f_extramethod(x: A):
            return x.extramethod()

        def f_bool(x: A):
            return bool(x)

        def f_str(x: A):
            return str(x)

        def f_repr(x: A):
            return repr(x)

        def f_call(x: A):
            return x(1)

        def f_1in(x: A):
            return 1 in x

        def f_0in(x: A):
            return 0 in x

        def f_len(x: A):
            return len(x)

        def f_add(x: A):
            return x + A.a()

        def f_sub(x: A):
            return x - A.a()

        def f_mul(x: A):
            return x * A.a()

        def f_div(x: A):
            return x / A.a()

        def f_floordiv(x: A):
            return x // A.a()

        def f_matmul(x: A):
            return x @ A.a()

        def f_mod(x: A):
            return x % A.a()

        def f_and(x: A):
            return x & A.a()

        def f_or(x: A):
            return x | A.a()

        def f_xor(x: A):
            return x ^ A.a()

        def f_rshift(x: A):
            return x >> A.a()

        def f_lshift(x: A):
            return x << A.a()

        def f_pow(x: A):
            return x ** A.a()

        def f_neg(x: A):
            return -x

        def f_pos(x: A):
            return +x

        def f_invert(x: A):
            return ~x

        def f_abs(x: A):
            return abs(x)

        test_cases = [f_bool, f_str, f_repr, f_call, f_0in, f_1in, f_len, f_add, f_sub, f_mul, f_div,
                      f_floordiv, f_matmul, f_mod, f_and, f_or, f_xor, f_rshift, f_lshift, f_neg, f_pos, f_invert, f_abs]

        for f in test_cases:
            compiled_f = Compiled(f)
            r1 = f(A.a())
            r2 = compiled_f(A.a())
            self.assertEqual(r1, r2)
