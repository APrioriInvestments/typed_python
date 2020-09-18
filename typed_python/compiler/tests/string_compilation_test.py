#   Copyright 2017-2020 typed_python Authors
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

from typed_python import _types, ListOf, TupleOf, Dict, ConstDict, Compiled, Entrypoint, OneOf
from typed_python.test_util import currentMemUsageMb, compilerPerformanceComparison
from typed_python.compiler.runtime import PrintNewFunctionVisitor
import pytest
import unittest
import time
import flaky


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


def callOrExcept(f, *args):
    try:
        return ("Normal", f(*args))
    except Exception as e:
        return ("Exception", str(e))


def callOrExceptType(f, *args):
    try:
        return ("Normal", f(*args))
    except Exception as e:
        return ("Exception", str(type(e)))


def callOrExceptNoType(f, *args):
    try:
        return ("Normal", f(*args))
    except Exception:
        return ("Exception", )


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

        for s in someStrings:
            for i in range(-20, 20):
                self.assertEqual(callOrExcept(getitem, s, i), callOrExcept(lambda s, i: s[i], s, i), (s, i))

    def test_string_ord(self):
        @Compiled
        def callOrd(x: str):
            return ord(x)

        self.assertEqual(callOrd("a"), ord("a"))
        self.assertEqual(callOrd("\x00"), ord("\x00"))
        self.assertEqual(callOrd("\xFF"), ord("\xFF"))
        self.assertEqual(callOrd("\u1234"), ord("\u1234"))

        with self.assertRaisesRegex(TypeError, "of length 4 found"):
            callOrd("asdf")

    def test_string_chr(self):
        @Compiled
        def callChr(x: int):
            return chr(x)

        for i in range(0, 0x10ffff + 1):
            self.assertEqual(ord(callChr(i)), i)

    def test_string_startswith_endswith(self):
        def startswith(x: str, y: str):
            return x.startswith(y)

        def endswith(x: str, y: str):
            return x.endswith(y)

        def startswith_range(x: str, y: str, start: int, end: int):
            return x.startswith(y, start, end)

        def endswith_range(x: str, y: str, start: int, end: int):
            return x.endswith(y, start, end)

        c_startswith = Compiled(startswith)
        c_endswith = Compiled(endswith)
        c_startswith_range = Compiled(startswith_range)
        c_endswith_range = Compiled(endswith_range)

        strings = ["", "a", "ab", "b", "abc", "bc", "ac", "ab", "bc", "bca"]

        for s1 in strings:
            for s2 in strings:
                self.assertEqual(startswith(s1, s2), c_startswith(s1, s2))
                self.assertEqual(endswith(s1, s2), c_endswith(s1, s2))
                for start in range(-5, 5):
                    for end in range(-5, 5):
                        self.assertEqual(
                            startswith_range(s1, s2, start, end),
                            c_startswith_range(s1, s2, start, end),
                            (s1, s2, start, end)
                        )
                        self.assertEqual(
                            endswith_range(s1, s2, start, end),
                            c_endswith_range(s1, s2, start, end),
                            (s1, s2, start, end)
                        )

    def test_string_tuple_startswith_endswith(self):
        def startswith(x, y):
            return x.startswith(y)

        def endswith(x, y):
            return x.endswith(y)

        def startswith_range(x, y, start, end):
            return x.startswith(y, start, end)

        def endswith_range(x, y, start, end):
            return x.endswith(y, start, end)

        c_startswith = Entrypoint(startswith)
        c_endswith = Entrypoint(endswith)
        c_startswith_range = Entrypoint(startswith_range)
        c_endswith_range = Entrypoint(endswith_range)

        with self.assertRaises(TypeError):
            c_startswith('abc', ['a', 'b'])

        with self.assertRaises(TypeError):
            c_startswith('abc', (1, 3))

        strings = ["", "a", "ab", "b", "abc", "bc", "ac", "ab", "bc", "bca"]
        tuples = [(a, b) for a in strings for b in strings]
        for s in strings:
            for t in tuples:
                self.assertEqual(startswith(s, t), c_startswith(s, t))
                self.assertEqual(startswith(s, t), c_startswith(s, TupleOf(str)(t)))
                self.assertEqual(endswith(s, t), c_endswith(s, t))
                self.assertEqual(endswith(s, t), c_endswith(s, TupleOf(str)(t)))
                for start in range(-5, 5):
                    for end in range(-5, 5):
                        self.assertEqual(
                            startswith_range(s, t, start, end),
                            c_startswith_range(s, t, start, end),
                            (s, t, start, end)
                        )
                        self.assertEqual(
                            endswith_range(s, t, start, end),
                            c_endswith_range(s, t, start, end),
                            (s, t, start, end)
                        )

    def test_string_replace(self):
        def replace(x: str, y: str, z: str):
            return x.replace(y, z)

        def replace2(x: str, y: str, z: str, i: int):
            return x.replace(y, z, i)

        replaceCompiled = Compiled(replace)
        replace2Compiled = Compiled(replace2)

        strings = [""]
        for _ in range(2):
            for s in ["ab", "c", "ba"*100]:
                strings += [x + s for x in strings]

        for s1 in strings:
            for s2 in strings:
                for s3 in strings:
                    self.assertEqual(replace(s1, s2, s3), replaceCompiled(s1, s2, s3), (s1, s2, s3))

                    for i in [-1, 0, 1, 2]:
                        self.assertEqual(replace2(s1, s2, s3, i), replace2Compiled(s1, s2, s3, i), (s1, s2, s3, i))

    def test_string_getitem_slice(self):
        def getitem1(x: str, y: int):
            return x[:y]

        def getitem2(x: str, y: int):
            return x[y:]

        def getitem3(x: str, y: int, y2: int):
            return x[y:y2]

        getitem1Compiled = Compiled(getitem1)
        getitem2Compiled = Compiled(getitem2)
        getitem3Compiled = Compiled(getitem3)

        for s in ["", "asdf", "a", "asdfasdf"]:
            for i in range(-5, 10):
                self.assertEqual(getitem1(s, i), getitem1Compiled(s, i), (s, i))
                self.assertEqual(getitem2(s, i), getitem2Compiled(s, i), (s, i))

                for j in range(-5, 10):
                    self.assertEqual(getitem3(s, i, j), getitem3Compiled(s, i, j), (s, i, j))

    def test_string_lower_upper(self):

        @Compiled
        def c_lower(s: str):
            return s.lower()

        @Compiled
        def c_lower2(s: str, t: str):
            return s.lower(t)

        @Compiled
        def c_upper(s: str):
            return s.upper()

        @Compiled
        def c_upper2(s: str, t: str):
            return s.upper(t)

        some_lu_strings = [
            "abc"
            "Abc",
            "aBc",
            "abC",
            "ABC",
            "aBcDefGHiJkLm" * 10000,
            "\u00CA\u00F1\u011A\u1E66\u3444\u1E67\u1EEA\1F04",
            "\u00CA\u00F1\u011A\u1E66\u3444\u1E67\u1EEA\1F04" * 10000,
            "XyZ\U0001D471",
            "XyZ\U0001D471" * 10000,
            "\u007F\u0080\u0081\u07FF\u0800\u0801\uFFFF\U00010000\U00010001\U0010FFFF",
            "ß",
            "ﬁß",
            "ß\u1EEa",
            "ﬁß\u1EEa",
            "ß\U0001D471",
            "ﬁß\U0001D471",
            "ßİŉǰΐΰևẖẗẘẙẚẞὐὒὔὖᾀᾁᾂᾃᾄᾅᾆᾇᾈᾉᾊᾋᾌᾍᾎᾏᾐᾑᾒᾓᾔᾕᾖᾗᾘᾙᾚᾛᾜᾝᾞᾟᾠᾡᾢᾣᾤᾥᾦᾧᾨᾩᾪᾫᾬᾭᾮᾯᾲᾳᾴᾶᾷᾼῂῃῄῆῇῌῒΐῖῗῢΰῤῦῧῲῳῴῶῷῼﬀﬁﬂﬃﬄﬅﬆﬓﬔﬕﬖﬗ"
        ]
        for s in some_lu_strings:
            self.assertEqual(c_lower(s), s.lower(), s)
            self.assertEqual(c_upper(s), s.upper(), s)

        for s in some_lu_strings:
            self.assertEqual(callOrExceptType(c_lower2, s, s), callOrExceptType(s.lower, s), s)
            self.assertEqual(callOrExceptType(c_upper2, s, s), callOrExceptType(s.upper, s), s)

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

    def test_string_find2(self):
        def f_find(x, sub):
            return x.find(sub)

        def f_find2(x, sub, start):
            return x.find(sub, start)

        def f_find3(x, sub, start, end):
            return x.find(sub, start, end)

        def f_rfind(x, sub):
            return x.rfind(sub)

        def f_rfind2(x, sub, start):
            return x.rfind(sub, start)

        def f_rfind3(x, sub, start, end):
            return x.rfind(sub, start, end)

        def f_index(x, sub):
            return x.index(sub)

        def f_index2(x, sub, start):
            return x.index(sub, start)

        def f_index3(x, sub, start, end):
            return x.index(sub, start, end)

        def f_rindex(x, sub):
            return x.rindex(sub)

        def f_rindex2(x, sub, start):
            return x.rindex(sub, start)

        def f_rindex3(x, sub, start, end):
            return x.rindex(sub, start, end)

        cases = [b'ababcab', b'ababcabcab', b'aabbabbbaaabbaaba']
        subs = [97, 98, b'a', b'c', b'X', b'ab', b'ba', b'abc', b'']
        for v in cases:
            for sub in subs:
                for f in [f_find]:  # [f_find, f_rfind]:
                    r1 = f(v, sub)
                    r2 = Entrypoint(f)(v, sub)
                    self.assertEqual(r1, r2, (f, v, sub))
                #     g = f_index if f == f_find else f_rindex
                #     if r1 != -1:
                #         r3 = Entrypoint(g)(v, sub)
                #         self.assertEqual(r1, r3, (g, v, sub))
                #     else:
                #         with self.assertRaises(ValueError):
                #             Entrypoint(g)(v, sub)
                #
                for start in range(-10, 11, 2):
                    for f in [f_find2]:  # [f_find2, f_rfind2]:
                        r1 = f(v, sub, start)
                        r2 = Entrypoint(f)(v, sub, start)
                        self.assertEqual(r1, r2, (f, v, sub, start))
                #         g = f_index2 if f == f_find2 else f_rindex2
                #         if r1 != -1:
                #             r3 = Entrypoint(g)(v, sub, start)
                #             self.assertEqual(r1, r3, (g, v, sub, start))
                #         else:
                #             with self.assertRaises(ValueError):
                #                 Entrypoint(g)(v, sub, start)
                    for end in range(-10, 11, 2):
                        for f in [f_find3]:  # [f_find3, f_rfind3]:
                            r1 = f(v, sub, start, end)
                            r2 = Entrypoint(f)(v, sub, start, end)
                            self.assertEqual(r1, r2, (f, v, sub, start, end))
                #
                #             g = f_index3 if f == f_find3 else f_rindex3
                #             if r1 != -1:
                #                 r3 = Entrypoint(g)(v, sub, start, end)
                #                 self.assertEqual(r1, r3, (g, v, sub, start, end))
                #             else:
                #                 with self.assertRaises(ValueError):
                #                     Entrypoint(g)(v, sub, start, end)

    def test_string_from_float(self):
        @Compiled
        def toString(f: float):
            return str(f)

        self.assertEqual(toString(1.2), "1.2")

        self.assertEqual(toString(1), "1.0")

    def test_string_is_something(self):
        @Compiled
        def c_isalpha(s: str):
            return s.isalpha()

        @Compiled
        def c_isalnum(s: str):
            return s.isalnum()

        @Compiled
        def c_isdecimal(s: str):
            return s.isdecimal()

        @Compiled
        def c_isdigit(s: str):
            return s.isdigit()

        @Compiled
        def c_islower(s: str):
            return s.islower()

        @Compiled
        def c_isnumeric(s: str):
            return s.isnumeric()

        @Compiled
        def c_isprintable(s: str):
            return s.isprintable()

        @Compiled
        def c_isspace(s: str):
            return s.isspace()

        @Compiled
        def c_istitle(s: str):
            return s.istitle()

        @Compiled
        def c_isupper(s: str):
            return s.isupper()

        @Compiled
        def c_isidentifier(s: str):
            return s.isidentifier()

        def perform_comparison(s: str):
            self.assertEqual(c_isalpha(s), s.isalpha(), [hex(ord(c)) for c in s])
            self.assertEqual(c_isalnum(s), s.isalnum(), [hex(ord(c)) for c in s])
            self.assertEqual(c_isdecimal(s), s.isdecimal(), [hex(ord(c)) for c in s])
            self.assertEqual(c_isdigit(s), s.isdigit(), [hex(ord(c)) for c in s])
            self.assertEqual(c_islower(s), s.islower(), [hex(ord(c)) for c in s])
            self.assertEqual(c_isnumeric(s), s.isnumeric(), [hex(ord(c)) for c in s])
            self.assertEqual(c_isprintable(s), s.isprintable(), [hex(ord(c)) for c in s])
            self.assertEqual(c_isspace(s), s.isspace(), [hex(ord(c)) for c in s])
            self.assertEqual(c_istitle(s), s.istitle(), [hex(ord(c)) for c in s])
            self.assertEqual(c_isupper(s), s.isupper(), [hex(ord(c)) for c in s])
            self.assertEqual(c_isidentifier(s), s.isidentifier(), [hex(ord(c)) for c in s])

        perform_comparison("")
        for i in range(0, 0x1000):
            perform_comparison(chr(i))
        for i in range(0x1000, 0x110000, 47):
            perform_comparison(chr(i))
        for i in range(0, 0x1000):
            perform_comparison(chr(i) + "a")
            perform_comparison(chr(i) + "1")
            perform_comparison(chr(i) + "A")
            perform_comparison(chr(i) + "/")
            perform_comparison(chr(i) + " ")
            perform_comparison(chr(i) + "\u01C5")
            perform_comparison(chr(i) + "\u0FFF")
            perform_comparison(chr(i) + "\x00")
            perform_comparison(chr(i) + "\U00010401")
            perform_comparison(chr(i) + "\U00010428")

        titlestrings = [
            "Title Case", "TitleCa)se", "2Title/Case", "2title/case",
            "\u01C5a", "\u01C5A", "\u01C5\u1F88",
            "\u01C5 \u1F88", "\u01C5 \u1F88a", "\u01C5 \u1F88A",
            "\U00010401 \u1F88", "\U00010401 \u1F88a", "\U00010401 \u1F88A",
            "\U00010428 \u1F88", "\U00010428 \u1F88a", "\U00010428 \u1F88A"
        ]
        for s in titlestrings:
            self.assertEqual(c_istitle(s), s.istitle(), s)

    def test_string_case(self):
        def f_lower(x):
            return x.lower()

        def f_upper(x):
            return x.upper()

        def f_capitalize(x):
            return x.capitalize()

        def f_swapcase(x):
            return x.swapcase()

        def f_title(x):
            return x.title()

        def f_casefold(x):
            return x.casefold()

        cases = [
            'ﬁ',
            'abc'*10,
            '\xE1\u1F11c'*10,
            'ABC'*10,
            '\xC1\u1F19c'*10,
            'aBc\u2D1E\U0001D73D\U0001D792'*10,
            '1@=.,z中',
            'straße',
            'ﬁ',
            '',
            "ßİŉǰΐΰևẖẗẘẙẚẞὐὒὔὖᾀᾁᾂᾃᾄᾅᾆᾇᾈᾉᾊᾋᾌᾍᾎᾏᾐᾑᾒᾓᾔᾕᾖᾗᾘᾙᾚᾛᾜᾝᾞᾟᾠᾡᾢᾣᾤᾥᾦᾧᾨᾩᾪᾫᾬᾭᾮᾯᾲᾳᾴᾶᾷᾼῂῃῄῆῇῌῒΐῖῗῢΰῤῦῧῲῳῴῶῷῼﬀﬁﬂﬃﬄﬅﬆﬓﬔﬕﬖﬗ",
        ]
        cases += [x + ' ' + y for x in cases for y in cases]
        for f in [f_capitalize, f_lower, f_upper, f_capitalize, f_swapcase, f_title, f_casefold]:
            for v in cases:
                r1 = f(v)
                r2 = Entrypoint(f)(v)
                self.assertEqual(r1, r2, (f, v))

    def test_string_strip(self):
        @Compiled
        def strip(s: str):
            return s.strip()

        @Compiled
        def rstrip(s: str):
            return s.rstrip()

        @Compiled
        def lstrip(s: str):
            return s.lstrip()

        for s in ["", "asdf", "       ", "   asdf", "asdf   ", "\nasdf", "\tasdf"]:
            self.assertEqual(s.strip(), strip(s), s)
            self.assertEqual(s.rstrip(), rstrip(s), s)
            self.assertEqual(s.lstrip(), lstrip(s), s)

    @pytest.mark.skip(reason='just for comparing performance when changing implementation')
    def test_string_find_perf(self):
        @Compiled
        def c_find(c: str, s: str) -> int:
            return c.find(s)

        cases = ["ab" * 1000 + "c", 'x' * 2000 + 'yxx', "abc"*1000]
        subs = ["ab", "abc", "xy", "ca"]
        total = 0
        t0 = time.time()
        for _ in range(10000):
            for c in cases:
                for s in subs:
                    total += c_find(c, s)
        t1 = time.time()

        print(total)
        print(f"total time {t1-t0}")
        self.assertTrue(False)

    @flaky.flaky(max_runs=3, min_passes=1)
    def test_string_strip_perf(self):
        bigS = " " * 1000000
        littleS = " "

        def stripMany(s: str, times: int):
            res = 0
            for i in range(times):
                res += len(s.strip())
            return res

        compiledStripMany = Compiled(stripMany)

        for s, expectedRatio, passCount in [(bigS, 2.0, 100), (littleS, .25, 10000)]:
            t0 = time.time()
            stripMany(s, passCount)
            t1 = time.time()
            compiledStripMany(s, passCount)
            t2 = time.time()

            pyTime = t1 - t0
            compiledTime = t2 - t1
            ratio = compiledTime / pyTime

            print(f"Ratio of compiled to python for string of len {len(s)} is {ratio}")

            self.assertLess(ratio, expectedRatio)

    def test_string_split(self):
        @Compiled
        def c_split(s: str, sep: str, max: int) -> ListOf(str):
            return s.split(sep, max)

        @Compiled
        def c_split_2(s: str) -> ListOf(str):
            return s.split()

        @Compiled
        def c_split_3(s: str, sep: str) -> ListOf(str):
            return s.split(sep)

        @Compiled
        def c_split_3max(s: str, max: int) -> ListOf(str):
            return s.split(max)

        # unexpected standard behavior:
        #   "   abc   ".split(maxsplit=0) = "abc   "  *** not "abc" nor "   abc   " ***
        split_strings = [
            "  abc  ",
            "ahjdfashjkdfsj ksdjkhsfhjdkf",
            "ahjdfashjkdfsj ksdjkhsfhjdkf" * 100,
            "",
            "a",
            " one two  three   \tfour    \n\nfive\r\rsix\n",
            " one two  three   \tfour    \n\nfive\r\rsix\n" * 100
        ]
        for s in split_strings:
            result = callOrExceptNoType(c_split_2, s)
            if result[0] == "Normal":
                self.assertEqual(_types.refcount(result[1]), 1)
            baseline = callOrExceptNoType(lambda: s.split())
            self.assertEqual(result, baseline, f"{s} -> {result}")
            self.assertEqual(result, baseline, "{} -> {}".format(s, result))
            for m in range(-2, 10):
                result = callOrExceptNoType(c_split_3max, s, m)
                if result[0] == "Normal":
                    self.assertEqual(_types.refcount(result[1]), 1)
                baseline = callOrExceptNoType(lambda: s.split(maxsplit=m))
                self.assertEqual(result, baseline, f"{s},{m}-> {result}")

            for sep in ["", "j", "s", "d", "t", " ", "as", "jks"]:
                result = callOrExceptNoType(c_split_3, s, sep)
                if result[0] == "Normal":
                    self.assertEqual(_types.refcount(result[1]), 1)
                baseline = callOrExceptNoType(lambda: s.split(sep))
                self.assertEqual(result, baseline, f"{s},{sep}-> {result}")
                for m in range(-2, 10):
                    result = callOrExceptNoType(c_split, s, sep, m)
                    if result[0] == "Normal":
                        self.assertEqual(_types.refcount(result[1]), 1)
                    baseline = callOrExceptNoType(lambda: s.split(sep, m))
                    self.assertEqual(result, baseline, f"{s},{sep},{m}-> {result}")

        startusage = currentMemUsageMb()
        for i in range(100000):
            for s in split_strings:
                result = c_split_2(s)
                result = c_split(s, " ", 9)
        endusage = currentMemUsageMb()
        self.assertLess(endusage, startusage + 1)

    @flaky.flaky(max_runs=3, min_passes=1)
    def test_string_split_perf(self):
        def splitAndCount(s: str, sep: str, times: int):
            res = 0

            for i in range(times):
                res += len(s.split(sep))

            return res

        compiled, uncompiled = compilerPerformanceComparison(splitAndCount, ("a" + ",") * 100, ",", 100000)

        # our string split function is about 6 times slower than python. Mostly due to memory management
        # issues.
        print(uncompiled / compiled, " times faster in compiler")

        self.assertTrue(
            compiled < uncompiled * 10,
            f"Expected compiled time {compiled} to be not much slower than uncompiled time {uncompiled}. "
            f"Compiler was {compiled / uncompiled} times slower."
        )

    def test_type_of_string_split(self):
        def splitType():
            return type("asdf".split("s"))

        self.assertEqual(splitType(), Compiled(splitType)())

    def validate_joining_strings(self, function, make_obj):
        # Test data, the fields are: description, separator, items, expected output
        test_data = [
            ["simple data",
             ",", ["1", "2", "3"], "1,2,3"],

            ["longer separator",
             "---", ["1", "2", "3"], "1---2---3"],

            ["longer items",
             "---", ["aaa", "bb", "c"], "aaa---bb---c"],

            ["empty separator",
             "", ["1", "2", "3"], "123"],

            ["everything empty",
             "", [], ""],

            ["empty list",
             "a", [], ""],

            ["empty string in the items",
             "--", ["", "1", "3"], "--1--3"],

            ["blank string in the items",
             "--", [" ", "1", "3"], " --1--3"],

            ["separator with 3 codepoints",
             "☺", ["a", "bb", "ccc"], "a☺bb☺ccc"],

            #   ® - 2B
            #   ☺ - 3B
            #   𝍭 - 4B
            #   a - 1B
            #   🚀 - 4B
            #   Ͽ - 2B
            #   హ - 3B
            #   ీ - 3B
            #   c - 1B
            ["items with 1, 2, and 3 bytes for code point",
             "--", ["123", "®®", "హీaa"], "123--®®--హీaa"],

            ["separator with 4 bytes for code point, items with less",
             "𝍭", ["123", "®®", "హీ"], "123𝍭®®𝍭హీ"],
        ]

        for description, separator, items, expected in test_data:
            res = function(separator, make_obj(items))
            self.assertEqual(expected, res, description)

    def test_string_join_for_tuple_of_str(self):
        # test passing tuple of strings
        @Compiled
        def f(sep: str, items: TupleOf(str)) -> str:
            return sep.join(items)

        self.validate_joining_strings(f, lambda items: TupleOf(str)(items))

    def test_string_join_for_list_of_str(self):
        # test passing list of strings
        @Compiled
        def f(sep: str, items: ListOf(str)) -> str:
            return sep.join(items)

        self.validate_joining_strings(f, lambda items: ListOf(str)(items))

    def test_string_join_for_dict_of_str(self):
        # test passing list of strings
        @Compiled
        def f(sep: str, items: Dict(str, str)) -> str:
            return sep.join(items)

        self.validate_joining_strings(f, lambda items: Dict(str, str)({i: "a" for i in items}))

    def test_string_join_for_const_dict_of_str(self):
        # test passing list of strings
        @Compiled
        def f(sep: str, items: ConstDict(str, str)) -> str:
            return sep.join(items)

        self.validate_joining_strings(f, lambda items: ConstDict(str, str)({i: "a" for i in items}))

    def test_string_join_for_bad_types(self):
        """str.join supports only joining ListOf(str) or TupleOf(str)."""

        # test passing tuple of ints
        @Compiled
        def f_tup_int(sep: str, items: TupleOf(int)) -> str:
            return sep.join(items)

        with self.assertRaisesRegex(TypeError, ""):
            f_tup_int(",", ListOf(int)([1, 2, 3]))

        # test passing list of other types than strings
        @Compiled
        def f_int(sep: str, items: ListOf(int)) -> str:
            return sep.join(items)

        with self.assertRaisesRegex(TypeError, ""):
            f_int(",", ListOf(int)([1, 2, 3]))

    def test_fstring(self):
        @Compiled
        def f():
            a = 1
            b = "bb"
            f = 1.234567
            return f"<< {a} !! {b} ?? {f} -- {a + a + a} || {len(b)} >>"

        res = f()
        expected = "<< 1 !! bb ?? 1.234567 -- 3 || 2 >>"
        self.assertEqual(expected, res)
        # Note: this test fails with 1.23456 instead of 1.234567
        # due to inexact representation of that value as 1.2345600000000001

    def test_fstring_exception(self):
        @Compiled
        def f():
            return f"{not_valid_variable}"  # noqa

        with self.assertRaisesRegex(Exception, "not_valid"):
            f()

    def test_string_contains_string(self):
        @Entrypoint
        def f(x, y):
            return x in y

        @Entrypoint
        def fNot(x, y):
            return x not in y

        self.assertTrue(f("a", "asfd"))
        self.assertFalse(f("b", "asfd"))
        self.assertFalse(f("b", ListOf(str)(["asfd"])))
        self.assertTrue(f("asdf", ListOf(str)(["asdf"])))

        self.assertFalse(fNot("a", "asfd"))
        self.assertTrue(fNot("b", "asfd"))
        self.assertTrue(fNot("b", ListOf(str)(["asfd"])))
        self.assertFalse(fNot("asdf", ListOf(str)(["asdf"])))

    def test_string_of_global_function(self):
        def f():
            return str(callOrExcept)

        @Entrypoint
        def callit(f):
            return f()

        self.assertEqual(callit(f), str(callOrExcept))

    def test_compare_strings_to_constant(self):
        @Entrypoint
        def countEqualTo(z):
            res = 0
            for s in z:
                if s == "this is a string":
                    res += 1
            return res

        someStrings = ListOf(str)(["this is a string", "boo"] * 1000000)

        # burn in the compiler
        countEqualTo(someStrings)

        t0 = time.time()
        countEqualTo(someStrings)
        elapsed = time.time() - t0

        # I get about .03
        self.assertLess(elapsed, .1)

    def test_add_constants(self):
        @Entrypoint
        def addConstants(count):
            res = 0
            for i in range(count):
                if "a" + "b" == "ab":
                    res += 1
            return res

        self.assertEqual(addConstants(1000), 1000)

        t0 = time.time()
        addConstants(100000000)
        # llvm should recognize that this is just 'N' and so it should take no time.
        self.assertLess(time.time() - t0, 1e-4)

    def test_bad_string_index(self):
        @Entrypoint
        def doIt(x: OneOf(str, ConstDict(str, str))):
            return x["bd"]

        doIt({'bd': 'yes'})

    def test_iterate_list_of_strings(self):
        @Entrypoint
        def sumSplit(x: str):
            res = 0
            for i in x.split(","):
                res += int(i)
            return res

        assert sumSplit("1") == 1
        assert sumSplit("1,2") == 3

    def test_can_split_result_of_split(self):
        @Entrypoint
        def sumSplit(x: str):
            res = 0
            for segment in x.split(","):
                for i in segment.split("."):
                    res += int(i)
            return res

        assert sumSplit("1") == 1
        assert sumSplit("1.2,2") == 5

    def test_can_index_into_result_of_split(self):
        @Entrypoint
        def sumSplit(x: str):
            res = 0
            for segment in x.split(","):
                fields = segment.split(".")
                field0 = fields[0]
                res += int(field0)
            return res

        with PrintNewFunctionVisitor():
            assert sumSplit("1") == 1

        assert sumSplit("1.2,2") == 3

    def test_call_int_on_object(self):
        @Entrypoint
        def f(x: object):
            return int(x)

        assert f("1") == 1

    def test_string_iteration(self):
        def iter(x: str):
            r = ListOf(str)()
            for a in x:
                r.append(a)
            return r

        def iter_constant():
            r = ListOf(str)()
            for a in "constant":
                r.append(a)
            return r

        def contains_space(x: str):
            for c in x:
                if c == ' ':
                    return True
            return False

        r1 = iter_constant()
        r2 = Compiled(iter_constant)()
        self.assertEqual(type(r1), type(r2))
        self.assertEqual(r1, r2)

        for v in ['whatever', 'o', '']:
            r1 = iter(v)
            r2 = Compiled(iter)(v)
            self.assertEqual(type(r1), type(r2))
            self.assertEqual(r1, r2)

        for v in ['', 'a', ' ', 'abc ', 'x'*1000+' '+'x'*1000, 'y'*1000]:
            r1 = contains_space(v)
            r2 = Compiled(contains_space)(v)
            self.assertEqual(r1, r2)

    def test_string_mult(self):
        def f_mult(x, n):
            return x * n

        v = "XyZ"
        for n in [1, 5, 100, 0, -1]:
            r1 = f_mult(v, n)
            r2 = Entrypoint(f_mult)(v, n)
            self.assertEqual(r1, r2)

    def test_string_decode(self):
        # various permutation of parameters (positional, keyword, missing)
        def f_1(x, enc, err):
            return str(x, enc, err)

        def f_2(x, enc, err):
            return str(x, encoding=enc, errors=err)

        def f_3(x, enc, _err):
            return (str(x, enc), str(x))

        def f_4(x, enc, err):
            return (str(x, encoding=enc), str(x, errors=err))

        for v in [b'quarter'*1000, b'25\xC2\xA2', b'\xE2\x82\xAC100', b'\xF0\x9F\x98\x80', b'']:
            for f in [f_1, f_2, f_3, f_4]:
                for enc in ["utf-8", "utf-16", "ascii", "cp863", "hz"]:
                    for err in ["strict", "ignore", "replace"]:
                        r1 = callOrExceptType(f, v, enc, err)
                        r2 = callOrExceptType(Entrypoint(f), v, enc, err)
                        self.assertEqual(r1, r2)

    def test_string_encode(self):
        # various permutation of parameters (positional, keyword, missing)
        def f_1(x, enc, err):
            return x.encode(enc, err)

        def f_2(x, enc, err):
            return x.encode(encoding=enc, errors=err)

        def f_3(x, enc, _err):
            return (x.encode(enc), x.encode())

        def f_4(x, enc, err):
            return (x.encode(encoding=enc), (x.encode(errors=err)))

        for v in ['quarter'*1000, '25¢', '€100', '汉字', '']:
            for f in [f_1, f_2, f_3, f_4]:
                for enc in ["utf-8", "utf-16", "ascii", "cp863", "hz"]:
                    for err in ["strict", "ignore", "replace"]:
                        r1 = callOrExceptType(f, v, enc, err)
                        r2 = callOrExceptType(Entrypoint(f), v, enc, err)
                        self.assertEqual(r1, r2, (f, v, enc, err))

    @pytest.mark.skip(reason='not performant')
    def test_string_codec(self):
        s1 = ''.join([chr(i) for i in range(0, 0x10ffff, 13) if i < 0xD800 or i > 0xDFFF])
        s2 = ''.join([chr(i) for i in range(1, 0x10ffff, 17) if i < 0xD800 or i > 0xDFFF])
        s3 = ''.join([chr(i) for i in range(2, 0x10ffff, 11) if i < 0xD800 or i > 0xDFFF])
        cases = [s1, s2, s3]
        for i in [2 ** n for n in range(16)]:
            cases += [s1[1:i], s2[1:i], s3[1:i]]

        def f_encode(s: str) -> bytes:
            return s.encode('utf-8', 'strict')

        def f_decode(s: bytes) -> str:
            return s.decode('utf-8', 'strict')

        def f_endecode(s: str) -> bool:
            s2 = s.encode('utf-8', 'strict').decode('utf-8', 'strict')
            return s == s2

        # c_encode = Compiled(f_encode)
        # c_decode = Compiled(f_decode)
        c_endecode = Compiled(f_endecode)

        for v in cases:
            self.assertTrue(f_endecode(v))
            self.assertTrue(c_endecode(v))

    @pytest.mark.skip(reason='not performant')
    def test_string_codec_perf(self):
        repeat = 500
        s1 = ''.join([chr(i) for i in range(0, 0x10ffff, 13) if i < 0xD800 or i > 0xDFFF])
        s2 = ''.join([chr(i) for i in range(1, 0x10ffff, 17) if i < 0xD800 or i > 0xDFFF])
        s3 = ''.join([chr(i) for i in range(2, 0x10ffff, 11) if i < 0xD800 or i > 0xDFFF])
        cases = [s1, s2, s3]
        for i in [2**n for n in range(16)]:
            cases += [s1[1:i], s2[1:i], s3[1:i]]
        Cases = ListOf(str)(cases)

        def f_endecode(s: str) -> bool:
            # s2 = s.encode('utf-8', 'strict').decode('utf-8', 'strict')
            s2 = s.encode('utf-8', 'strict')
            return s == s2

        def f_endecode2(cases: ListOf(str)) -> bool:
            # s2 = s.encode('utf-8', 'strict').decode('utf-8', 'strict')
            ret = True
            for _ in range(1000):
                for s in cases:
                    s2 = s.encode('utf-8', 'strict')
                    ret &= (s == s2)
            return ret

        verify = True
        t0 = time.time()
        for _ in range(repeat):
            for v in cases:
                verify &= f_endecode(v)
        t1 = time.time()
        print("baseline ", t1 - t0)
        # self.assertTrue(verify)

        c_endecode = Compiled(f_endecode)
        verify = True
        t2 = time.time()
        for _ in range(repeat):
            for v in Cases:
                verify &= c_endecode(v)
        t3 = time.time()
        print("compiled ", t3 - t2)
        # self.assertTrue(verify)

        t0 = time.time()
        f_endecode2(Cases)
        t1 = time.time()
        print("baseline2 ", t1 - t0)

        c_endecode2 = Compiled(f_endecode2)
        t2 = time.time()
        c_endecode2(Cases)
        t3 = time.time()
        print("compiled2 ", t3 - t2)
