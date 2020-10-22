#   Copyright 2017-2019 typed_python Authors
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

from typed_python import Function, TupleOf, Compiled, Entrypoint, ListOf
import typed_python._types as _types
import unittest
import time
import psutil


class TestTupleOfCompilation(unittest.TestCase):
    def checkFunction(self, f, argsToCheck):
        f_fast = Compiled(f)

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
        def f(x: TupleOf(float), y: TupleOf(float)) -> float:
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

        self.assertEqual(_types.refcount(aTupleOfFloat), 1)

        t_py, t_fast = self.checkFunction(f, [(aTupleOfFloat, aTupleOfFloat2)])

        self.assertEqual(_types.refcount(aTupleOfFloat), 1)

        # I get around 150x
        self.assertTrue(t_py / t_fast > 50.0)

        print(t_py / t_fast, " speedup")

    def test_tuple_passing(self):
        @Compiled
        def f(x: TupleOf(int)) -> int:
            return 0

        self.assertEqual(f((1, 2, 3)), 0)

    def test_tuple_len(self):
        @Compiled
        def f(x: TupleOf(int)) -> int:
            return len(x)

        self.assertEqual(f((1, 2, 3)), 3)

    def test_tuple_assign(self):
        @Compiled
        def f(x: TupleOf(int)) -> TupleOf(int):
            y = x
            return y

        t = TupleOf(int)((1, 2, 3))

        self.assertEqual(f(t), t)

        self.assertEqual(_types.refcount(t), 1)

    def test_tuple_indexing(self):
        @Compiled
        def f(x: TupleOf(int), y: int) -> int:
            return x[y]

        self.assertEqual(f((1, 2, 3), 1), 2)

        with self.assertRaises(Exception):
            f((1, 2, 3), 1000000000)

    def test_tuple_refcounting(self):
        @Function
        def f(x: TupleOf(int), y: TupleOf(int)) -> TupleOf(int):
            return x

        for compileIt in [False, True]:
            if compileIt:
                f = Compiled(f)

            intTup = TupleOf(int)(list(range(1000)))

            self.assertEqual(_types.refcount(intTup), 1)

            res = f(intTup, intTup)

            self.assertEqual(_types.refcount(intTup), 2)

            res = None  # noqa: F841

            self.assertEqual(_types.refcount(intTup), 1)

    def test_bad_mod_generates_exception(self):
        @Compiled
        def f(x: int, y: int) -> int:
            return x % y

        with self.assertRaises(Exception):
            f(0, 0)

    def test_tuple_of_adding(self):
        T = TupleOf(int)

        @Compiled
        def f(x: T, y: T) -> T:
            return x + y

        t1 = T((1, 2, 3))
        t2 = T((3, 4))

        res = f(t1, t2)

        self.assertEqual(_types.refcount(res), 1)
        self.assertEqual(_types.refcount(t1), 1)
        self.assertEqual(_types.refcount(t2), 1)

        self.assertEqual(res, t1+t2)

    def test_tuple_of_tuple_refcounting(self):
        T = TupleOf(int)
        TT = TupleOf(T)

        @Compiled
        def f(x: TT) -> TT:
            return x + x + x

        t1 = T((1, 2, 3))
        t2 = T((4, 5, 5))

        aTT = TT((t1, t2))

        fRes = f(aTT)

        self.assertEqual(fRes, aTT+aTT+aTT)
        self.assertEqual(_types.refcount(aTT), 1)
        self.assertEqual(_types.refcount(fRes), 1)

        fRes = None
        aTT = None
        self.assertEqual(_types.refcount(t1), 1)

    def test_tuple_creation_doesnt_leak(self):
        T = TupleOf(int)

        @Compiled
        def f(x: T, y: T) -> T:
            return x + y

        t1 = T(tuple(range(10000)))

        initMem = psutil.Process().memory_info().rss / 1024 ** 2

        for i in range(10000):
            f(t1, t1)

        finalMem = psutil.Process().memory_info().rss / 1024 ** 2

        self.assertTrue(finalMem < initMem + 5)

    def test_create_tuple_of_directly_from_list(self):
        def makeT():
            return TupleOf(int)([1, 2, 3, 4])

        self.assertEqual(makeT(), Compiled(makeT)())
        self.assertEqual(type(makeT()), type(Compiled(makeT)()))

    def test_create_tuple_of_directly_from_tuple(self):
        def makeT():
            return TupleOf(int)((1, 2, 3, 4))

        self.assertEqual(makeT(), Compiled(makeT)())
        self.assertEqual(type(makeT()), type(Compiled(makeT)()))

    def test_create_tuple_of_from_untyped(self):
        def makeT(aList: object):
            return TupleOf(int)(aList)

        self.assertEqual(makeT([1, 2, 3, 4]), Compiled(makeT)([1, 2, 3, 4]))
        self.assertEqual(makeT({1: 2}), Compiled(makeT)({1: 2}))

    def test_tuple_of_from_list_of_empty(self):
        @Entrypoint
        def makeT(aList: ListOf(int)):
            return TupleOf(int)(aList)

        assert len(makeT([])) == 0
