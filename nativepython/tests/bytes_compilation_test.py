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
from nativepython.runtime import Runtime
import unittest
import time


def Compiled(f):
    f = Function(f)
    return Runtime.singleton().compile(f)


someBytes = [
    b"",
    b"a",
    b"as\x00df",
    b"\x00\x01",
    b"\x00\x01\x02\x00\x01",
]


class TestBytesCompilation(unittest.TestCase):
    def test_bytes_passing_and_refcounting(self):
        @Compiled
        def takeFirst(x: bytes, y: bytes):
            return x

        @Compiled
        def takeSecond(x: bytes, y: bytes):
            return y

        for s in someBytes:
            for s2 in someBytes:
                self.assertEqual(s, takeFirst(s, s2))
                self.assertEqual(s2, takeSecond(s, s2))

    def test_bytes_len(self):
        @Compiled
        def compiledLen(x: bytes):
            return len(x)

        for s in someBytes:
            self.assertEqual(len(s), compiledLen(s))

    def test_bytes_concatenation(self):
        @Compiled
        def concat(x: bytes, y: bytes):
            return x + y

        @Compiled
        def concatLen(x: bytes, y: bytes):
            return len(x + y)

        for s in someBytes:
            for s2 in someBytes:
                self.assertEqual(s+s2, concat(s, s2))
                self.assertEqual(len(s+s2), concatLen(s, s2))

    def test_bytes_constants(self):
        def makeConstantConcatenator(s):
            def returner():
                return s
            return returner

        for s in someBytes:
            f = Compiled(makeConstantConcatenator(s))
            s_from_code = f()

            self.assertEqual(s, s_from_code, (repr(s), repr(s_from_code)))

    def test_bytes_getitem(self):
        @Compiled
        def getitem(x: bytes, y: int):
            return x[y]

        def callOrExcept(f, *args):
            try:
                return ("Normal", f(*args))
            except Exception as e:
                return ("Exception", str(e))

        for s in someBytes:
            for i in range(-20, 20):
                self.assertEqual(callOrExcept(getitem, s, i), callOrExcept(lambda s, i: s[i], s, i), (s, i))

    def test_bytes_perf(self):
        def bytesAdd(x: bytes):
            i = 0
            res = 0
            while i < len(x):
                j = 0
                while j < len(x):
                    res = res + x[i] + x[j]
                    j = j + 1
                i = i + 1
            return res

        compiled = Compiled(bytesAdd)

        t0 = time.time()
        interpreterRes = bytesAdd(b" " * 2000)
        t1 = time.time()
        compiledRes = compiled(b" " * 2000)
        t2 = time.time()

        self.assertEqual(interpreterRes, compiledRes)
        speedup = (t1 - t0) / (t2 - t1)

        # I get about 200
        self.assertGreater(speedup, 100)
