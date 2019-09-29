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

from typed_python import TypeFunction, Function, Alternative, Forward, Dict
import typed_python._types as _types
from nativepython.runtime import Runtime
from nativepython import SpecializedEntrypoint
import unittest
import time
from math import trunc, floor, ceil


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

    def test_compile_alternative_format(self):
        A1 = Alternative("A1", a={'a': int}, b={'b': str})
        A2 = Alternative("A2", a={'a': int}, b={'b': str},
                         __str__=lambda self: "my str"
                         )
        A3 = Alternative("A3", a={'a': int}, b={'b': str},
                         __format__=lambda self, spec: "my format " + spec
                         )

        def a1_format(x: A1):
            return format(x)

        def a2_format(x: A2):
            return format(x)

        def a3_format(x: A3):
            return format(x)

        def a3_format_spec(x: A3):
            return format(x, "spec")

        r1 = a1_format(A1.a())
        c1_format = Compiled(a1_format)
        r2 = c1_format(A1.a())
        self.assertEqual(r1, r2)

        r1 = a2_format(A2.a())
        c2_format = Compiled(a2_format)
        r2 = c2_format(A2.a())
        self.assertEqual(r1, r2)

        r1 = a3_format(A3.a())
        c3_format = Compiled(a3_format)
        r2 = c3_format(A3.a())
        self.assertEqual(r1, r2)

        r1 = a3_format_spec(A3.a())
        c3_format_spec = Compiled(a3_format_spec)
        r2 = c3_format_spec(A3.a())
        self.assertEqual(r1, r2)

        # This failed when I forgot to support ConcreteAlternativeWrappers
        @SpecializedEntrypoint
        def specialized_format(x):
            return format(x)

        test_values = [A1.a(), A1.b(), A2.a(), A2.b(), A3.a(), A3.b()]
        for v in test_values:
            r1 = format(v)
            r2 = specialized_format(v)
            self.assertEqual(r1, r2)

    def test_compile_alternative_bytes(self):
        A = Alternative("A", a={'a': int}, b={'b': str},
                        __bytes__=lambda self: b'my bytes'
                        )

        def f_bytes(x: A):
            return bytes(x)

        v = A.a()
        r1 = f_bytes(v)
        c_f = Compiled(f_bytes)
        r2 = c_f(v)
        self.assertEqual(r1, r2)

    def test_compile_alternative_attr(self):

        def A_getattr(self, n):
            return self.d[n]

        def A_setattr(self, n, v):
            self.d[n] = v

        def A_delattr(self, n):
            del self.d[n]

        A = Alternative("A", a={'d': Dict(str, str), 'i': int},
                        __getattr__=A_getattr,
                        __setattr__=A_setattr,
                        __delattr__=A_delattr
                        )

        def f_getattr1(x: A):
            return x.q

        def f_getattr2(x: A):
            return x.z

        def f_setattr1(x: A, s: str):
            x.q = s

        def f_setattr2(x: A, s: str):
            x.z = s

        def f_delattr1(x: A):
            del x.q

        def f_delattr2(x: A):
            del x.z

        c_getattr1 = Compiled(f_getattr1)
        c_getattr2 = Compiled(f_getattr2)
        c_setattr1 = Compiled(f_setattr1)
        c_setattr2 = Compiled(f_setattr2)
        c_delattr1 = Compiled(f_delattr1)
        c_delattr2 = Compiled(f_delattr2)
        for v in [A.a()]:
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
            # exception types are different
            with self.assertRaises(KeyError):
                f_getattr1(v)
            with self.assertRaises(TypeError):
                c_getattr1(v)
            self.assertEqual(f_getattr2(v), "4")
            self.assertEqual(f_getattr2(v), c_getattr2(v))
            f_delattr2(v)
            with self.assertRaises(KeyError):
                f_getattr1(v)
            with self.assertRaises(TypeError):
                c_getattr1(v)
            with self.assertRaises(KeyError):
                f_getattr2(v)
            with self.assertRaises(TypeError):
                c_getattr2(v)
            f_setattr1(v, "5")
            f_setattr2(v, "6")
            c_delattr1(v)
            with self.assertRaises(KeyError):
                f_getattr1(v)
            with self.assertRaises(TypeError):
                c_getattr1(v)
            self.assertEqual(f_getattr2(v), "6")
            self.assertEqual(f_getattr2(v), c_getattr2(v))
            c_delattr2(v)
            with self.assertRaises(KeyError):
                f_getattr1(v)
            with self.assertRaises(TypeError):
                c_getattr1(v)
            with self.assertRaises(KeyError):
                f_getattr2(v)
            with self.assertRaises(TypeError):
                c_getattr2(v)

    def test_compile_alternative_float_methods(self):
        # if __float__ is defined, then floor() and ceil() are based off this conversion,
        # when __floor__ and __ceil__ are not defined
        A = Alternative("A", a={'a': int}, b={'b': str},
                        __float__=lambda self: 1234.5
                        )

        def f_floor(x: A):
            return floor(x)

        def f_ceil(x: A):
            return ceil(x)

        test_cases = [f_floor, f_ceil]
        for f in test_cases:
            r1 = f(A.a())
            compiled_f = Compiled(f)
            r2 = compiled_f(A.a())
            self.assertEqual(r1, r2)

        B = Alternative("B", a={'a': int}, b={'b': str},
                        __round__=lambda self, n: 1234 + n,
                        __trunc__=lambda self: 1,
                        __floor__=lambda self: 2,
                        __ceil__=lambda self: 3
                        )

        def f_round0(x: B):
            return round(x, 0)

        def f_round1(x: B):
            return round(x, 1)

        def f_round2(x: B):
            return round(x, 2)

        def f_round_1(x: B):
            return round(x, -1)

        def f_round_2(x: B):
            return round(x, -2)

        def f_trunc(x: B):
            return trunc(x)

        def f_floor(x: B):
            return floor(x)

        def f_ceil(x: B):
            return ceil(x)

        test_cases = [f_round0, f_round1, f_round2, f_round_1, f_round_2, f_trunc, f_floor, f_ceil]
        for f in test_cases:
            r1 = f(B.a())
            compiled_f = Compiled(f)
            r2 = compiled_f(B.a())
            self.assertEqual(r1, r2)

    def test_compile_alternative_magic_methods(self):
        A = Alternative("A", a={'a': int}, b={'b': str},
                        __bool__=lambda self: False,
                        __str__=lambda self: "my str",
                        __repr__=lambda self: "my repr",
                        __call__=lambda self, i: "my call",
                        __len__=lambda self: 42,
                        __contains__=lambda self, item: item == 1,
                        __bytes__=lambda self: b'my bytes',
                        __format__=lambda self, spec: "my format",

                        __add__=lambda self, other: A.b("add"),
                        __sub__=lambda self, other: A.b("sub"),
                        __mul__=lambda self, other: A.b("mul"),
                        __matmul__=lambda self, other: A.b("matmul"),
                        __truediv__=lambda self, other: A.b("truediv"),
                        __floordiv__=lambda self, other: A.b("floordiv"),
                        __divmod__=lambda self, other: A.b("divmod"),
                        __mod__=lambda self, other: A.b("mod"),
                        __pow__=lambda self, other: A.b("pow"),
                        __lshift__=lambda self, other: A.b("lshift"),
                        __rshift__=lambda self, other: A.b("rshift"),
                        __and__=lambda self, other: A.b("and"),
                        __or__=lambda self, other: A.b("or"),
                        __xor__=lambda self, other: A.b("xor"),

                        __iadd__=lambda self, other: A.b("iadd"),
                        __isub__=lambda self, other: A.b("isub"),
                        __imul__=lambda self, other: A.b("imul"),
                        __imatmul__=lambda self, other: A.b("imatmul"),
                        __itruediv__=lambda self, other: A.b("itruediv"),
                        __ifloordiv__=lambda self, other: A.b("ifloordiv"),
                        __imod__=lambda self, other: A.b("imod"),
                        __ipow__=lambda self, other: A.b("ipow"),
                        __ilshift__=lambda self, other: A.b("ilshift"),
                        __irshift__=lambda self, other: A.b("irshift"),
                        __iand__=lambda self, other: A.b("iand"),
                        __ior__=lambda self, other: A.b("ior"),
                        __ixor__=lambda self, other: A.b("ixor"),

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

        def f_bytes(x: A):
            return bytes(x)

        def f_format(x: A):
            return format(x)

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

        def f_iadd(x: A):
            x += A.a()
            return x

        def f_isub(x: A):
            x -= A.a()
            return x

        def f_imul(x: A):
            x *= A.a()
            return x

        def f_idiv(x: A):
            x /= A.a()
            return x

        def f_ifloordiv(x: A):
            x //= A.a()
            return x

        def f_imatmul(x: A):
            x @= A.a()
            return x

        def f_imod(x: A):
            x %= A.a()
            return x

        def f_iand(x: A):
            x &= A.a()
            return x

        def f_ior(x: A):
            x |= A.a()
            return x

        def f_ixor(x: A):
            x ^= A.a()
            return x

        def f_irshift(x: A):
            x >>= A.a()
            return x

        def f_ilshift(x: A):
            x <<= A.a()
            return x

        def f_ipow(x: A):
            x **= A.a()
            return x

        test_cases = [f_bool, f_str, f_repr, f_call, f_0in, f_1in, f_len,
                      f_add, f_sub, f_mul, f_div, f_floordiv, f_matmul, f_mod, f_and, f_or, f_xor, f_rshift, f_lshift, f_pow,
                      f_neg, f_pos, f_invert, f_abs]
        # These fail:
        #   failing_test_cases = [f_bytes, f_format
        # These fail when compiling, because python_ast and native_ast have no representation of in-place operators:
        #   failing_test_cases = [f_iadd, f_isub, f_imul, f_idiv, f_ifloordiv, f_imatmul,
        #                         f_imod, f_iand, f_ior, f_ixor, f_irshift, f_ilshift, f_ipow]

        for f in test_cases:
            compiled_f = Compiled(f)
            r1 = f(A.a())
            r2 = compiled_f(A.a())
            self.assertEqual(r1, r2)

    def test_compile_alternative_comparison_defaults(self):

        B = Alternative("B", a={'a': int}, b={'b': str})

        def f_eq(x: B, y: B):
            return x == y

        def f_ne(x: B, y: B):
            return x != y

        def f_lt(x: B, y: B):
            return x < y

        def f_gt(x: B, y: B):
            return x > y

        def f_le(x: B, y: B):
            return x <= y

        def f_ge(x: B, y: B):
            return x >= y

        def f_hash(x: B):
            return hash(x)

        values = [B.a(0), B.a(1), B.b("a"), B.b("b")]
        test_cases = [f_eq, f_ne, f_lt, f_gt, f_le, f_ge]
        for f in test_cases:
            for v1 in values:
                for v2 in values:
                    compiled_f = Compiled(f)
                    r1 = f(v1, v2)
                    r2 = compiled_f(v1, v2)
                    self.assertEqual(r1, r2)
        test_cases = [f_hash]
        for f in test_cases:
            for v in values:
                compiled_f = Compiled(f)
                r1 = f(v)
                r2 = compiled_f(v)
                self.assertEqual(r1, r2)

    def test_compile_alternative_comparison_methods(self):

        C = Alternative("C", a={'a': int}, b={'b': str},
                        __eq__=lambda self, other: True,
                        __ne__=lambda self, other: False,
                        __lt__=lambda self, other: True,
                        __gt__=lambda self, other: False,
                        __le__=lambda self, other: True,
                        __ge__=lambda self, other: False,
                        __hash__=lambda self: 123,
                        )

        def f_eq(x: C):
            return x == C.a()

        def f_ne(x: C):
            return x != C.a()

        def f_lt(x: C):
            return x < C.a()

        def f_gt(x: C):
            return x > C.a()

        def f_le(x: C):
            return x <= C.a()

        def f_ge(x: C):
            return x >= C.a()

        # Note: hash is not overridden by method __hash__ in compiled or interpreted case
        def f_hash(x: C):
            return hash(x)

        test_cases = [f_eq, f_ne, f_lt, f_gt, f_le, f_ge, f_hash]

        for f in test_cases:
            compiled_f = Compiled(f)
            r1 = f(C.a())
            r2 = compiled_f(C.a())
            self.assertEqual(r1, r2)
