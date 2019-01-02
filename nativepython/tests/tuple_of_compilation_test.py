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

class TestTupleOfCompilation(unittest.TestCase):
    def checkFunction(self, f, argsToCheck):
        r  = Runtime.singleton()

        f_fast = r.compile(f)

        t_py = 0.0
        t_fast = 0.0
        for a in argsToCheck:
            t0 = time.time()
            fastval = f_fast(*a)
            t1 = time.time()
            slowval = f(*a)
            t2 = time.time()

            t_py += t2-t1
            t_fast += t1-t0

            self.assertEqual(fastval, slowval)
        return t_py, t_fast

    def test_tuple_of_float(self):
        def f(x: TupleOf(float), y:TupleOf(float)) -> float:
            j = 0
            res = 0.0
            i = 0

            while j < len(y):
                i = 0
                while i < len(x):
                    res = res + x[i] * y[j]
                    i = i + 1
                j = j + 1

            return res

        aTupleOfFloat = TupleOf(float)(list(range(1000)))
        aTupleOfFloat2 = TupleOf(float)(list(range(1000)))

        self.assertEqual(_types.refcount(aTupleOfFloat),1)

        t_py, t_fast = self.checkFunction(f, [(aTupleOfFloat,aTupleOfFloat2)])

        self.assertEqual(_types.refcount(aTupleOfFloat),1)

        #I get around 150x
        self.assertTrue(t_py / t_fast > 50.0)

        print(t_py / t_fast, " speedup")

    def test_tuple_indexing(self):
        @TypedFunction
        def f(x: TupleOf(int), y:int) -> int:
            return x[y]

        Runtime.singleton().compile(f)

        self.assertEqual(f((1,2,3),1), 2)

        with self.assertRaises(Exception):
            f((1,2,3),1000000000)

    def test_tuple_refcounting(self):
        @TypedFunction
        def f(x: TupleOf(int), y: TupleOf(int)) -> TupleOf(int):
            return x

        for compileIt in [False, True]:
            if compileIt:
                Runtime.singleton().compile(f)

            intTup = TupleOf(int)(list(range(1000)))

            self.assertEqual(_types.refcount(intTup),1)

            res = f(intTup, intTup)

            self.assertEqual(_types.refcount(intTup),2)

            res = None

            self.assertEqual(_types.refcount(intTup),1)

    def test_bad_mod_generates_exception(self):
        @TypedFunction
        def f(x: int, y:int) -> int:
            return x % y

        Runtime.singleton().compile(f)

        with self.assertRaises(Exception):
            f(0,0)

