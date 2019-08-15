#   Coyright 2017-2019 Nativepython Authors
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

from typed_python import _types, Function, ListOf, TupleOf, NamedTuple, Dict, ConstDict, OneOf, \
    Int64, Int32, Int16, Int8, UInt64, UInt32, UInt16, UInt8, Bool, Float64, Float32, \
    String, Bytes, Alternative, Set
from typed_python.test_util import currentMemUsageMb
from nativepython.runtime import Runtime
import unittest
import pytest


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
            "\u007F\u0080\u0081\u07FF\u0800\u0801\uFFFF\U00010000\U00010001\U0010FFFF"
        ]
        for s in some_lu_strings:
            self.assertEqual(c_lower(s), s.lower())
            self.assertEqual(c_upper(s), s.upper())

        for s in some_lu_strings:
            self.assertEqual(callOrExceptType(c_lower2, s, s), callOrExceptType(s.lower, s))
            self.assertEqual(callOrExceptType(c_upper2, s, s), callOrExceptType(s.upper, s))

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

    def test_string_from_float(self):
        @Compiled
        def toString(f: float):
            return str(f)

        self.assertEqual(toString(1.2), "1.2")

        # this is not actually correct, but it's our current behavior
        self.assertEqual(toString(1), "1")

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

    def test_string_split(self):
        @Compiled
        def c_split(s: str, sep: str, max: int) -> ListOf(str):
            r = ListOf(str)()
            s.split(r, s, sep, max)
            return r

        @Compiled
        def c_split_2(s: str) -> ListOf(str):
            r = ListOf(str)()
            s.split(r, s)
            return r

        @Compiled
        def c_split_3(s: str, sep: str) -> ListOf(str):
            r = ListOf(str)()
            s.split(r, s, sep)
            return r

        @Compiled
        def c_split_3max(s: str, max: int) -> ListOf(str):
            r = ListOf(str)()
            s.split(r, s, max)
            return r

        @Compiled
        def c_split_initialized(s: str, lst: ListOf(str)) -> ListOf(str):
            r = ListOf(str)(lst)
            s.split(r, s)
            return r

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
            result = callOrExceptNoType(c_split_initialized, s, ["a", "b", "c"])
            if result[0] == "Normal":
                self.assertEqual(_types.refcount(result[1]), 1)
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
            f = 1.23456
            return f"<< {a} !! {b} ?? {f} -- {a + a + a} || {len(b)} >>"
        res = f()
        expected = "<< 1 !! bb ?? 1.23456 -- 3 || 2 >>"
        self.assertEqual(expected, res)

    def test_fstring_exception(self):
        @Compiled
        def f():
            return f"{not_valid_variable}"  # noqa

        with self.assertRaisesRegex(Exception, "not_valid"):
            f()

    @pytest.mark.skip(reason="not implemented yet")
    @unittest.skip
    def test_string_from_various_types(self):
        NT1 = NamedTuple(a=int, b=float, c=str, d=str)
        NT2 = NamedTuple(s=String, t=TupleOf(int))
        Alt1 = Alternative("Alt1", X={'a': int}, Y={'b': str})
        cases = [
            (Bool, True),
            (Float64, 1.0/7.0),  # verify number of digits in string representation
            (Float64, 8.0/7.0),  # verify number of digits in string representation
            (Float64, 71.0/7.0),  # verify number of digits in string representation
            (Float64, 0.123456789),
            (Float64, 1.23456789),
            (Float64, 12.3456789),
            (Float64, 1.0),  # verify trailing zeros in string representation of float
            (Float64, 2**32),  # verify trailing zeros in string representation of float
            (Float64, 2**64),  # verify trailing zeros in string representation of float
            (Float64, 1.8e19),  # verify trailing zeros in string representation of float
            (Float64, 1e16),  # verify exp transition in string representation of float
            (Float64, 1e16-2),  # verify exp transition in string representation of float
            (Float64, 1e16+2),  # verify exp transition in string representation of float
            (Float64, -1.0/7.0),  # verify number of digits in string representation
            (Float64, -8.0/7.0),  # verify number of digits in string representation
            (Float64, -71.0/7.0),  # verify number of digits in string representation
            (Float64, -0.123456789),
            (Float64, -1.23456789),
            (Float64, -12.3456789),
            (Float64, -1.0),  # verify trailing zeros in string representation of float
            (Float64, -2**32),  # verify trailing zeros in string representation of float
            (Float64, -2**64),  # verify trailing zeros in string representation of float
            (Float64, -1.8e19),  # verify trailing zeros in string representation of float
            (Float64, -1e16),  # verify exp transition in string representation of float
            (Float64, -1e16-2),  # verify exp transition in string representation of float
            (Float64, -1e16+2),  # verify exp transition in string representation of float
            (Alt1, Alt1.X(a=-1)),
            (Alt1, Alt1.Y(b='yes')),
            (Float32, 1.234),
            (int, 3),
            (int, -2**63),
            (Bool, True),
            (Int8, -128),
            (Int16, -32768),
            (Int32, -2**31),
            (UInt8, 127),
            (UInt16, 65535),
            (UInt32, 2**32-1),
            (UInt64, 2**64-1),
            (Float32, 1.234567),
            (Float32, 1.234),
            (String, "abcd"),
            (Bytes, b"\01\00\02\03"),
            (Set(int), [1, 3, 5, 7]),
            (TupleOf(Int64), (7, 6, 5, 4, 3, 2, -1)),
            (TupleOf(Int32), (7, 6, 5, 4, 3, 2, -2)),
            (TupleOf(Int16), (7, 6, 5, 4, 3, 2, -3)),
            (TupleOf(Int8), (7, 6, 5, 4, 3, 2, -4)),
            (TupleOf(UInt64), (7, 6, 5, 4, 3, 2, 1)),
            (TupleOf(UInt32), (7, 6, 5, 4, 3, 2, 2)),
            (TupleOf(UInt16), (7, 6, 5, 4, 3, 2, 3)),
            (TupleOf(UInt8), (7, 6, 5, 4, 3, 2, 4)),
            (TupleOf(str), ("a", "b", "c")),
            (ListOf(str), ["a", "b", "d"]),
            (Dict(str, int), {'y': 7, 'n': 6}),
            (ConstDict(str, int), {'y': 2, 'n': 4}),
            (TupleOf(int), tuple(range(10000))),
            (OneOf(String, Int64), "ab"),
            (OneOf(String, Int64), 34),
            (NT1, NT1(a=1, b=2.3, c="c", d="d")),
            (NT2, NT2(s="xyz", t=tuple(range(10000))))
        ]

        for T, v in cases:
            @Compiled
            def toString(f: T):
                return str(f)

            a1 = toString(v)
            a2 = str(T(v))
            if a1 != a2:
                print("mismatch")
            # self.assertEqual(toString(v), str(T(v)))
