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
import psutil

def Compiled(f):
    f = TypedFunction(f)
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
        Complex = Alternative("Complex",
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
