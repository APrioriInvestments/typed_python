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

import time
import typed_python._types as _types
import unittest

from flaky import flaky
from typed_python import ConstDict, TupleOf, ListOf, Tuple, Compiled
from typed_python import Entrypoint


dictTypes = [
    ConstDict(str, str),
    ConstDict(int, str),
    ConstDict(int, int)
]


def makeSomeValues(dtype, count=10):
    res = dtype()

    for i in range(count):
        if res.KeyType is str:
            k = str(i)
        else:
            k = i

        if res.ValueType is str:
            v = str(i)
        else:
            v = i
        res = res + {k: v}

    return res


class TestConstDictCompilation(unittest.TestCase):
    def test_const_dict_copying(self):
        for dtype in dictTypes:
            @Compiled
            def copyInOut(x: dtype):
                _ = x
                return x

            aDict = makeSomeValues(dtype)
            self.assertEqual(copyInOut(aDict), aDict)
            self.assertEqual(_types.refcount(aDict), 1)

    def test_const_dict_len(self):
        for dtype in dictTypes:
            @Compiled
            def compiledLen(x: dtype):
                return len(x)

            for ct in range(10):
                d = makeSomeValues(dtype, ct)
                self.assertEqual(len(d), compiledLen(d))

    def test_const_dict_getitem(self):
        for dtype in dictTypes:
            @Compiled
            def compiledGetItem(x: dtype, y: dtype.KeyType):
                return x[y]

            def callOrExpr(f):
                try:
                    return ("Value", f())
                except Exception:
                    return ("Exception", )

            d = makeSomeValues(dtype, 10)
            bigger_d = makeSomeValues(dtype, 20)

            for key in bigger_d:
                self.assertEqual(callOrExpr(lambda: d[key]), callOrExpr(lambda: compiledGetItem(d, key)))

    def test_const_dict_getitem_coercion(self):
        d = ConstDict(int, int)({1: 2})

        with self.assertRaises(TypeError):
            d.get(1.5)

        with self.assertRaises(TypeError):
            d[1.5]

        @Entrypoint
        def get(d, k):
            return d.get(k)

        @Entrypoint
        def getitem(d, k):
            return d[k]

        @Entrypoint
        def get_default(d, k, default):
            return d.get(k, default)

        # looking up should work
        self.assertEqual(get(d, 1), 2)
        self.assertEqual(get_default(d, 1, None), 2)
        self.assertEqual(getitem(d, 1), 2)

        # by default, 'get' returns None
        self.assertEqual(get(d, 2), None)
        self.assertEqual(d.get(2), None)

        # and can be overridden with arbitrary types
        self.assertEqual(get_default(d, 2, 123.5), 123.5)
        self.assertEqual(d.get(2, 123.5), 123.5)

        # getitem will throw an exception
        with self.assertRaises(KeyError):
            getitem(d, 2)

        # keys don't coerce types like float to int.
        with self.assertRaises(TypeError):
            get(d, 1.5)

        with self.assertRaises(TypeError):
            get_default(d, 1.5, None)

        with self.assertRaises(TypeError):
            getitem(d, 1.5)

    def test_const_dict_contains(self):
        for dtype in dictTypes:
            @Compiled
            def compiledIn(x: dtype, y: dtype.KeyType):
                return y in x

            @Compiled
            def compiledNotIn(x: dtype, y: dtype.KeyType):
                return y not in x

            d = makeSomeValues(dtype, 10)
            bigger_d = makeSomeValues(dtype, 20)

            for key in bigger_d:
                self.assertEqual(key in d, compiledIn(d, key))
                self.assertEqual(key not in d, compiledNotIn(d, key))

        @Compiled
        def compiledContains(x: ConstDict(int, int), y: int):
            return y in x

        self.assertTrue(compiledContains({k*2: k*2 for k in range(10)}, 2))
        self.assertFalse(compiledContains({k*2: k*2 for k in range(10)}, 3))
        self.assertTrue(compiledContains({k*2: k*2 for k in range(10)}, 4))
        self.assertFalse(compiledContains({k*2: k*2 for k in range(10)}, 5))

    @flaky(max_runs=3, min_passes=1)
    def test_const_dict_loops_perf(self):
        def loop(x: ConstDict(int, int)):
            res = 0
            i = 0
            while i < len(x):
                j = 0
                while j < len(x):
                    res = res + x[j] + x[i]
                    j = j + 1
                i = i + 1
            return res

        compiledLoop = Compiled(loop)

        aBigDict = {i: i % 20 for i in range(1000)}
        compiledLoop(aBigDict)

        t0 = time.time()
        interpreterResult = loop(aBigDict)
        t1 = time.time()
        compiledResult = compiledLoop(aBigDict)
        t2 = time.time()

        speedup = (t1-t0)/(t2-t1)

        self.assertEqual(interpreterResult, compiledResult)

        # I get about 3x. This is not as big a speedup as some other things we do
        # because most of the time is spent in the dictionary lookup, and python's
        # dict lookup is quite fast.
        print("ConstDict lookup speedup is ", speedup)
        self.assertGreater(speedup, 1.75)

    def test_const_dict_key_error(self):
        @Compiled
        def lookup(x: ConstDict(int, int), y: int):
            return x[y]

        self.assertEqual(lookup({1: 2}, 1), 2)
        with self.assertRaises(Exception):
            lookup({1: 2}, 2)

    def test_const_dict_unsafe_operations(self):
        T = ConstDict(int, int)

        t = T({1: 2, 3: 4})

        @Entrypoint
        def getkey(d, i):
            return d.get_key_by_index_unsafe(i)

        @Entrypoint
        def getvalue(d, i):
            return d.get_value_by_index_unsafe(i)

        self.assertEqual(getkey(t, 0), 1)
        self.assertEqual(getkey(t, 1), 3)
        self.assertEqual(getvalue(t, 0), 2)
        self.assertEqual(getvalue(t, 1), 4)

        TOI = TupleOf(int)
        T2 = ConstDict(TOI, TOI)

        toi = TOI((1, 2, 3))
        self.assertEqual(_types.refcount(toi), 1)

        t2 = T2({toi: toi})
        self.assertEqual(_types.refcount(toi), 3)

        toi_copy = getkey(t2, 0)
        self.assertEqual(_types.refcount(toi), 4)

        toi_copy_2 = getvalue(t2, 0)
        self.assertEqual(_types.refcount(toi), 5)

        toi_copy = toi_copy_2 = None # noqa
        t2 = None

        self.assertEqual(_types.refcount(toi), 1)

    def test_const_dict_comparison(self):
        @Entrypoint
        def eq(x, y):
            return x == y

        @Entrypoint
        def neq(x, y):
            return x != y

        @Entrypoint
        def lt(x, y):
            return x < y

        T = ConstDict(str, str)
        t1 = T({'1': '2'})
        t2 = T({'1': '3'})

        self.assertTrue(eq(t1, t1))
        self.assertFalse(neq(t1, t1))

        self.assertFalse(eq(t1, t2))
        self.assertTrue(neq(t1, t2))

    def test_const_dict_iteration(self):
        def iterateConstDict(cd):
            res = ListOf(type(cd).KeyType)()

            for k in cd:
                res.append(k)

            return res

        def iterateKeysObject(cd):
            res = ListOf(type(cd).KeyType)()

            for k in cd.keys():
                res.append(k)

            return res

        def iterateValuesObject(cd):
            res = ListOf(type(cd).ValueType)()

            for k in cd.values():
                res.append(k)

            return res

        def iterateItemsObject(cd):
            res = ListOf(Tuple(type(cd).KeyType, type(cd).ValueType))()

            for k in cd.items():
                res.append(k)

            return res

        T = ConstDict(int, int)
        t0 = T()
        t1 = T({1: 2})
        t2 = T({k * 2: k * 2 for k in range(10)})
        t3 = T({k * 2: k * 5 for k in range(10)})

        for kf in [iterateConstDict, iterateKeysObject, iterateValuesObject, iterateItemsObject]:
            for toCheck in [t0, t1, t2, t3]:
                if len(toCheck):
                    self.assertEqual(_types.refcount(toCheck), 1)

                self.assertEqual(kf(toCheck), Entrypoint(kf)(toCheck), (kf, toCheck))

                if len(toCheck):
                    self.assertEqual(_types.refcount(toCheck), 1)

        T2 = ConstDict(TupleOf(int), TupleOf(str))
        atup = TupleOf(int)((1, 2, 3))
        atup2 = TupleOf(str)(('1', '2', '3', '4'))

        t0 = T2({atup: atup2})

        self.assertEqual(_types.refcount(atup), 2)
        self.assertEqual(_types.refcount(atup2), 2)

        for kf in [iterateConstDict, iterateKeysObject, iterateValuesObject, iterateItemsObject]:
            Entrypoint(kf)(t0)

            self.assertEqual(_types.refcount(atup), 2)
            self.assertEqual(_types.refcount(atup2), 2)

    def test_const_dict_iteration_perf(self):
        def loop(x: ConstDict(int, int)):
            res = 0
            for k1 in x.values():
                for k2 in x.values():
                    res = res + k1 + k2
            return res

        compiledLoop = Compiled(loop)

        aBigDict = {i: i % 20 for i in range(3000)}
        compiledLoop(aBigDict)

        t0 = time.time()
        interpreterResult = loop(aBigDict)
        t1 = time.time()
        compiledResult = compiledLoop(aBigDict)
        t2 = time.time()

        speedup = (t1-t0)/(t2-t1)

        self.assertEqual(interpreterResult, compiledResult)

        # I get about 70x
        print("ConstDict iteration speedup is ", speedup)
        self.assertGreater(speedup, 2)
