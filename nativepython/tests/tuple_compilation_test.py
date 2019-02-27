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


class TestTupleCompilation(unittest.TestCase):
    def test_tuple_passing(self):
        T = Tuple(float, int, str)

        @Compiled
        def f(x: T) -> T:
            y = x
            return y

        t = T((0.0, 1, "hi"))
        self.assertEqual(f(t), t)

    def test_named_tuple_passing(self):
        NT = NamedTuple(a=float, b=int, c=str)

        @Compiled
        def f(x: NT) -> NT:
            y = x
            return y

        nt = NT(a=0.0, b=1, c="hi")
        self.assertEqual(f(nt), nt)

    def test_named_tuple_getattr(self):
        NT = NamedTuple(a=float, b=int, c=str)

        @Compiled
        def f(x: NT) -> str:
            return x.c + x.c

        nt = NT(a=0.0, b=1, c="hi")
        self.assertEqual(f(nt), "hihi")

    def test_named_tuple_assignment_refcounting(self):
        class C(Class):
            x = Member(int)

        NT = NamedTuple(c=C)

        @Compiled
        def f(x: NT):
            y = x
            return y.c

        c = C(x=20)
        res = f(NT(c=c))

        self.assertEqual(res.x, 20)
        self.assertEqual(_types.refcount(res), 2)
