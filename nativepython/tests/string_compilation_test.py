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


someStrings = [
    "",
    "a",
    "as\x00df",
    "\u00F1",
    "\u0F01",
    "\u0F01",
    "\u1002"
    ]

for s1 in list(someStrings):
    for s2 in list(someStrings):
        someStrings.append(s1+s2)
someStrings = sorted(set(someStrings))

class TestStringCompilation(unittest.TestCase):
    def test_string_passing_and_refcounting(self):
        @Compiled
        def takeFirst(x: str, y: str):
            return x

        @Compiled
        def takeSecond(x: str, y: str):
            return y

        for s in someStrings:
            for s2 in someStrings:
                self.assertEqual(s, takeFirst(s,s2))
                self.assertEqual(s2, takeSecond(s,s2))

    def test_string_len(self):
        @Compiled
        def compiledLen(x: str):
            return len(x)

        for s in someStrings:
            self.assertEqual(len(s), compiledLen(s))

    def test_string_concatenation(self):
        @Compiled
        def concat(x: str, y: str):
            return x + y

        @Compiled
        def concatLen(x: str, y: str):
            return len(x + y)

        for s in someStrings:
            for s2 in someStrings:
                self.assertEqual(s+s2, concat(s,s2))
                self.assertEqual(len(s+s2), concatLen(s,s2))

    def test_string_constants(self):
        def makeConstantConcatenator(s):
            def returner():
                return s
            return returner

        for s in someStrings:
            f = Compiled(makeConstantConcatenator(s))
            s_from_code = f()

            self.assertEqual(s, s_from_code, (repr(s),repr(s_from_code)))

    def test_string_getitem(self):
        @Compiled
        def getitem(x: str, y: int):
            return x[y]

        def callOrExcept(f, *args):
            try:
                return ("Normal", f(*args))
            except Exception as e:
                return ("Exception", str(e))

        for s in someStrings:
            for i in range(-20, 20):
                self.assertEqual(callOrExcept(getitem, s,i), callOrExcept(lambda s,i: s[i], s, i), (s,i))

