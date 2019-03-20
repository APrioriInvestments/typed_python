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
    "\u007F\u0080\u0081\u07FF",
    "\u0800\u0801\uFFFF",
    "\U00010000\U00010001\U0010FFFF"
]

for s1 in list(someStrings):
    for s2 in list(someStrings):
        someStrings.append(s1+s2)
someStrings = sorted(set(someStrings))


def callOrExceptType(f, *args):
    try:
        return ("Normal", f(*args))
    except Exception as e:
        return ("Exception", str(type(e)))


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

        @Compiled
        def c_lower2(s: str, t: str):
            return s.lower(t)

        someupper_strings = [
            "abc"
            "Abc",
            "aBc",
            "abC",
            "ABC",
            "aBcDeFgHiJkLm" * 10000,
            "\u00CA\u00D1\u011A\u1E66\u1EEA",
            "\u00CA\u00D1\u011A\u1E66\u1EEA" * 10000,
            "XyZ\U0001D471",
            "XyZ\U0001D471" * 10000,
            "\u007F\u0080\u0081\u07FF\u0800\u0801\uFFFF\U00010000\U00010001\U0010FFFF"
        ]
        for s in someupper_strings:
            self.assertEqual(c_lower(s), s.lower())

        for s in someupper_strings:
            self.assertEqual(callOrExceptType(c_lower2, s, s), callOrExceptType(s.lower, s))

    def test_string__find(self):

        @Compiled
        def c_find(s: str, sub: str, start: int, end: int):
            return s.find(sub, start, end)

        @Compiled
        def c_find_3(s: str, sub: str, start: int):
            return s.find(sub, start)

        @Compiled
        def c_find_2(s: str, sub: str):
            return s.find(sub)

        def test_find(t):
            substrings = ["", "x", "xyz", "a"*100, t[0:-2] + t[-1] if len(t) > 2 else ""]
            for start in range(0, min(len(t), 8)):
                for end in range(start+1, min(len(t), 8)+1):
                    substrings.append(t[start:end])

            indexrange = [
                -(len(t)+1)*256,
                -len(t)-1, -len(t), -len(t)+1,
                -len(t)//2-1, -len(t)//2, -len(t)//2+1,
                -2, -1, 0, 1, 2,
                len(t)//2-1, len(t)//2, len(t)//2+1,
                len(t)-3, len(t)-2, len(t)-1, len(t), len(t)+1,
                (len(t)+1)*256
            ]
            indexrange = sorted(set(indexrange))
            for sub in substrings:
                for start in indexrange:
                    for end in indexrange:
                        i = t.find(sub, start, end)
                        c_i = c_find(t, sub, start, end)
                        if i != c_i:
                            self.assertEqual(i, c_i)
                        self.assertEqual(i, c_i)
            for sub in substrings:
                for start in indexrange:
                    i = t.find(sub, start)
                    c_i = c_find_3(t, sub, start)
                    self.assertEqual(i, c_i)
            for sub in substrings:
                i = t.find(sub)
                c_i = c_find_2(t, sub)
                self.assertEqual(i, c_i)

        test_find("")
        test_find("a")
        test_find("abcdef")
        test_find("baaaab")
        test_find(("a"*99 + "b")*100)
        test_find("\u00CA\u00D1\u011A\u1E66\u1EEA")
        test_find("\u00DD\U00012EEE\U0001D471\u00AA\U00011234")

        @Compiled
        def c_find_1(s: str):
            return s.find()

        @Compiled
        def c_find_5(s: str, sub: str, start: int, end: int, extra: int):
            return s.find(str, start, end, extra)

        for s in ["a", ""]:
            self.assertEqual(callOrExceptType(c_find_5, s, s, 0, 1, 2), callOrExceptType(s.find, s, 0, 1, 2))
            self.assertEqual(callOrExceptType(c_find_1, s), callOrExceptType(s.find))

        """
        def rep_find(s, subs):
            result = 0
            for i in range(1000):
                for sub in subs:
                    result += s.find(sub)
            return result

        @Compiled
        def c_rep_find(s: str, subs: ListOf(str)) -> int:
            result = 0
            for i in range(1000):
                for sub in subs:
                    result += s.find(sub)
            return result

        def test_find_perf(f, s, subs):
            t0 = time.time()
            f(s,subs)
            return time.time() - t0

        s = "a" * 100000 + "b" + "a" * 100000
        subs = ["aaa", "aba","aab","baa", "bbb"]
        c_elapsed = test_find_perf(c_rep_find, s, subs)
        elapsed = test_find_perf(rep_find, s, subs)
        self.assertTrue(c_elapsed < elapsed,
                "Slow Performance: compiled took {0} sec versus baseline {1}"
                .format(c_elapsed, elapsed)
        )
        """
