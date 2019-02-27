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


dictTypes = [
    ConstDict(str, str),
    ConstDict(int, str),
    ConstDict(int, int)
]


def makeSomeValues(dtype, count=10):
    res = dtype()

    for i in range(count):
        if res.KeyType is String:
            k = str(i)
        else:
            k = i

        if res.ValueType is String:
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

    def test_const_dict_loops(self):
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

        t0 = time.time()
        interpreterResult = loop(aBigDict)
        t1 = time.time()
        compiledResult = compiledLoop(aBigDict)
        t2 = time.time()

        speedup = (t1-t0)/(t2-t1)

        self.assertEqual(interpreterResult, compiledResult)

        # I get about 3x. This is not as big a speedup as some other thigns we do
        # because most of the time is spent in the dictionary lookup, and python's
        # dict lookup is quite fast.
        self.assertGreater(speedup, 2)
