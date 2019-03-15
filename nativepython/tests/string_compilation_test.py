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

from typed_python import Function
from nativepython.runtime import Runtime
import unittest


def Compiled(f):
    f = Function(f)
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
                self.assertEqual(s, takeFirst(s, s2))
                self.assertEqual(s2, takeSecond(s, s2))

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
                self.assertEqual(s+s2, concat(s, s2))
                self.assertEqual(len(s+s2), concatLen(s, s2))

    def test_string_comparison(self):
        @Compiled
        def lt(x: str, y: str):
            return x < y

        @Compiled
        def lte(x: str, y: str):
            return x <= y

        @Compiled
        def gte(x: str, y: str):
            return x >= y

        @Compiled
        def gt(x: str, y: str):
            return x > y

        @Compiled
        def eq(x: str, y: str):
            return x == y

        @Compiled
        def neq(x: str, y: str):
            return x != y

        for s in someStrings:
            for s2 in someStrings:
                self.assertEqual(eq(s, s2), s == s2)
                self.assertEqual(neq(s, s2), s != s2)
                self.assertEqual(gte(s, s2), s >= s2)
                self.assertEqual(lte(s, s2), s <= s2)
                self.assertEqual(gt(s, s2), s > s2)
                self.assertEqual(lt(s, s2), s < s2)

    def test_string_constants(self):
        def makeConstantConcatenator(s):
            def returner():
                return s
            return returner

        for s in someStrings:
            f = Compiled(makeConstantConcatenator(s))
            s_from_code = f()

            self.assertEqual(s, s_from_code, (repr(s), repr(s_from_code)))

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
                self.assertEqual(callOrExcept(getitem, s, i), callOrExcept(lambda s, i: s[i], s, i), (s, i))


    def test_string_lower(self):

        @Compiled
        def c_lower(s: str):
            return s.lower()

        someupper_strings = ["Abc","aBc","abC", "ABC","\u00CA\u00D1\u011A\u1E66\u1EEA","XyZ\U0001D471"]
        for s in someupper_strings:
            self.assertEqual(s.lower(), c_lower(s))


    def test_string_find(self):

        @Compiled
        def c_find(s: str, sub: str, start: int, end: int):
            return s.find(sub, start, end)

        @Compiled
        def c_find_3(s: str, sub: str, start: int):
            return s.find(sub, start)

        @Compiled
        def c_find_2(s: str, sub: str):
            return s.find(sub)

        def test_find(teststring):
            substrings = ["", "x", "xyz", teststring[0:-2] + teststring[-1] ]
            for start in range(0,len(teststring)):
                for end in range(start+1,len(teststring)+1):
                    substrings.append(teststring[start:end])
            for sub in substrings:
                for start in range(-len(teststring)-1, len(teststring)+1):
                    for end in range(-len(teststring)-1, len(teststring)+1):
                        i = teststring.find(sub, start, end)
                        c_i = c_find(teststring, sub, start, end)
                        self.assertEqual(i, c_i)
            for sub in substrings:
                for start in range(-len(teststring)-1, len(teststring)+1):
                    i = teststring.find(sub, start)
                    c_i = c_find_3(teststring, sub, start)
                    self.assertEqual(i, c_i)
            for sub in substrings:
                i = teststring.find(sub)
                c_i = c_find_2(teststring, sub)
                self.assertEqual(i, c_i)

        test_find("abcdef")
        test_find("\u00CA\u00D1\u011A\u1E66\u1EEA")
        test_find("\u00DD\U00012EEE\U0001D471\u00AA\U00011234")


