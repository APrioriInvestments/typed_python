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
        Complex = Alternative(
            "Complex",
            A={'a': str, 'b': int},
            B={'a': str, 'c': int},
            C={'a': str, 'd': lambda: Complex}
        )

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
            return Alternative(
                "Tree",
                Leaf={'value': T},
                Node={'left': Tree(T), 'right': Tree(T)}
            )

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
