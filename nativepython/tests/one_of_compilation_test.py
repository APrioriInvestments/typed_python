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
    f = TypedFunction(f)
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

        aTup = TupleOf(int)((1,2,3))
        self.assertEqual(f(aTup), aTup)

        self.assertEqual(_types.refcount(aTup), 1)
