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

from typed_python import Compiled, Entrypoint, ListOf, TupleOf, Dict, ConstDict
from typed_python.compiler.type_wrappers.bytes_wrapper import bytes_isalnum, bytes_isalpha, \
    bytes_isdigit, bytes_islower, bytes_isspace, bytes_istitle, bytes_isupper
from typed_python.test_util import compilerPerformanceComparison
import flaky

import unittest
import time


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

    def test_bytes_literals(self):

        def f(i: int):
            x = b"abcdefghijklmnopqrstuvwxyz"
            return x[i]

        def g(i: int):
            y = bytes(b'01234567890123456789012345')
            return y[i]

        cf = Compiled(f)
        cg = Compiled(g)

        for i in range(26):
            self.assertEqual(f(i), cf(i))
            self.assertEqual(g(i), cg(i))

    def test_bytes_conversions(self):

        def f(x: bytes):
            return bytes(x)

        cf = Compiled(f)

        for v in [b'123', b'abcdefgh', b'\x00\x01\x02\x00']:
            self.assertEqual(f(v), cf(v))

        # TODO: support this
        # def g(x: str):
        #     return bytes(x, "utf8")
        #
        # cg = Compiled(g)
        #
        # for v in ['123', 'abcdefgh', 'a\u00CAb', 'XyZ\U0001D471']:
        #     self.assertEqual(g(v), cg(v))

    def test_bytes_slice(self):
        def f(x: bytes, l: int, r: int):
            return x[l:r]

        fComp = Compiled(f)

        for l in range(-10, 10):
            for r in range(-10, 10):
                for b in [b"", b"a", b"as", b"asdf"]:
                    self.assertEqual(fComp(b, l, r), f(b, l, r))

    def test_compare_bytes_to_constant(self):
        @Entrypoint
        def countEqualTo(z):
            res = 0
            for s in z:
                if s == b"this are some bytes":
                    res += 1
            return res

        someBytes = ListOf(bytes)([b"this are some bytes", b"boo"] * 1000000)

        # burn in the compiler
        countEqualTo(someBytes)

        t0 = time.time()
        countEqualTo(someBytes)
        elapsed = time.time() - t0

        # I get about .044
        self.assertLess(elapsed, .15)
        print("elapsed = ", elapsed)

    def test_add_constants(self):
        @Entrypoint
        def addConstants(count):
            res = 0
            for i in range(count):
                if b"a" + b"b" == b"ab":
                    res += 1
            return res

        self.assertEqual(addConstants(1000), 1000)

        t0 = time.time()
        addConstants(1000000)
        # llvm should recognize that this is just 'N' and so it should take no time.
        self.assertLess(time.time() - t0, 1e-4)

    def test_bytes_split(self):
        def split(someBytes, *args):
            return someBytes.split(*args)

        compiledSplit = Entrypoint(split)

        for args in [
            (b'asdf',),
            (b'asdf', b'blahblah'),
            (b'asdf', b'a'),
            (b'asdf', b's'),
            (b'asdf', b'K'),
            (b'asdf', b's', 0),
            (b'a aaa bababa a',),
            (b'a aaa bababa a', b'b'),
            (b'a aaa bababa a', b'b', 1),
            (b'a aaa bababa a', b'b', 2),
            (b'a aaa bababa a', b'b', 3),
            (b'a aaa bababa a', b'b', 4),
            (b'a aaa bababa a', b'b', 5),
            (b'a aaa bababa a', b'a', 5),
            (b'a aaa bababa a', b'K', 5),
        ]:
            assert split(*args) == compiledSplit(*args)

    @flaky.flaky(max_runs=3, min_passes=1)
    def test_bytes_split_perf(self):
        def splitAndCount(s: bytes, sep: bytes, times: int):
            res = 0

            for i in range(times):
                res += len(s.split(sep))

            return res

        compiled, uncompiled = compilerPerformanceComparison(splitAndCount, (b"a,") * 100, b",", 100000)

        # our string split function is about 6 times slower than python. Mostly due to memory management
        # issues.
        print(uncompiled / compiled, " times faster in compiler")

        Entrypoint(splitAndCount)((b"a,") * 100, b",", 1000000)

        self.assertTrue(
            compiled < uncompiled * 10,
            f"Expected compiled time {compiled} to be not much slower than uncompiled time {uncompiled}. "
            f"Compiler was {compiled / uncompiled} times slower."
        )

    def test_bytes_iteration(self):
        def iter(x: bytes):
            r = ListOf(int)()
            for a in x:
                r.append(a)
            return r

        def iter_constant():
            r = ListOf(int)()
            for a in b'constant':
                r.append(a)
            return r

        def contains_space(x: bytes):
            for i in x:
                if i == 32:
                    return True
            return False

        def f_index(x: bytes, i: int):
            return x[i]

        r1 = iter_constant()
        r2 = Compiled(iter_constant)()
        self.assertEqual(type(r1), type(r2))
        self.assertEqual(r1, r2)

        for v in [b'whatever', b'o', b'']:
            r1 = iter(v)
            r2 = Compiled(iter)(v)
            self.assertEqual(type(r1), type(r2))
            self.assertEqual(r1, r2)

        for v in [b'', b'a', b' ', b'abc ', b'x' * 1000 + b' ' + b'x' * 1000, b'y' * 1000]:
            r1 = contains_space(v)
            r2 = Compiled(contains_space)(v)
            self.assertEqual(r1, r2)

    def test_bytes_bool_fns(self):
        def f_isalnum(x):
            return x.isalnum()

        def f_isalpha(x):
            return x.isalpha()

        def f_isdigit(x):
            return x.isdigit()

        def f_islower(x):
            return x.islower()

        def f_isspace(x):
            return x.isspace()

        def f_istitle(x):
            return x.istitle()

        def f_isupper(x):
            return x.isupper()

        cases = [b'abc' + bytes([i]) + b'abc' for i in range(256)]
        cases += [b'ABC' + bytes([i]) + b'ABC' for i in range(256)]
        cases += [b'123' + bytes([i]) for i in range(256)]
        cases += [b'  \t\r\n\f' + bytes([i]) for i in range(256)]
        cases += [bytes([i]) + b'123' for i in range(256)]
        cases += [bytes([i]) + b'   'for i in range(256)]
        cases += [b'', b'A' * 1000, b'9' * 1000]
        cases += [b'Title Case', b'NotTitleCase']
        for f in [f_isalnum, f_isalpha, f_isdigit, f_islower, f_isspace, f_istitle, f_isupper]:
            for v in cases:
                r1 = f(v)
                r2 = Entrypoint(f)(v)
                self.assertEqual(r1, r2, (f, v))

    def test_bytes_startswith_endswith(self):
        def f_startswith(x, y):
            return x.startswith(y)

        def f_endswith(x, y):
            return x.endswith(y)

        xcases = [b'abaabaaabaaaaabaaaabaaabaabab', b'']
        ycases = [b'a', b'ab', b'ba', b'abaabaaabaaaaabaaaabaaabaabab', b'abaabaaabaaaaabaaaabaaabaababa', b'']
        for f in [f_startswith, f_endswith]:
            for x in xcases:
                for y in ycases:
                    r1 = f(x, y)
                    r2 = Entrypoint(f)(x, y)
                    self.assertEqual(r1, r2, (f, x, y))

    def test_bytes_lower_upper(self):
        def f_lower(x):
            return x.lower()

        def f_upper(x):
            return x.upper()

        cases = [b'abc'*10, b'ABC'*10, b'aBc'*10, b'1@=.,']
        for f in [f_lower, f_upper]:
            for v in cases:
                r1 = f(v)
                r2 = Entrypoint(f)(v)
                self.assertEqual(r1, r2, (f, v))

    def test_bytes_strip(self):
        def f_strip(x):
            return x.strip()

        def f_lstrip(x):
            return x.lstrip()

        def f_rstrip(x):
            return x.rstrip()

        pieces = [b'', b' ', b'\t\n', b'abc']
        cases = [a + b + c for a in pieces for b in pieces for c in pieces]

        for f in [f_strip, f_lstrip, f_rstrip]:
            for v in cases:
                r1 = f(v)
                r2 = Entrypoint(f)(v)
                self.assertEqual(r1, r2, (f, v))

        def f_strip2(x, y):
            return x.strip(y)

        def f_lstrip2(x, y):
            return x.lstrip(y)

        def f_rstrip2(x, y):
            return x.rstrip(y)

        for f in [f_strip2, f_lstrip2, f_rstrip2]:
            for y in [b'', b'a', b'ab', b'Z']:
                for v in cases:
                    r1 = f(v, y)
                    r2 = Entrypoint(f)(v, y)
                    self.assertEqual(r1, r2, (f, v, y))

    def test_bytes_count(self):
        def f_count(x, sub):
            return x.count(sub)

        def f_count2(x, sub, start):
            return x.count(sub, start)

        def f_count3(x, sub, start, end):
            return x.count(sub, start, end)

        cases = [b'ababcab', b'ababcabcab', b'aabbabbbaaabbaaba']
        subs = [b'a', b'ab', b'ba', b'abc', b'']
        for v in cases:
            for sub in subs:
                f = f_count
                r1 = f(v, sub)
                r2 = Entrypoint(f)(v, sub)
                self.assertEqual(r1, r2, (v, sub))

                for start in range(-10, 11, 2):
                    f = f_count2
                    r1 = f(v, sub, start)
                    r2 = Entrypoint(f)(v, sub, start)
                    self.assertEqual(r1, r2, (v, sub, start))
                    for end in range(-10, 11, 2):
                        f = f_count3
                        r1 = f(v, sub, start, end)
                        r2 = Entrypoint(f)(v, sub, start, end)
                        self.assertEqual(r1, r2, (v, sub, start, end))

    def test_bytes_find(self):
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
                for f in [f_find, f_rfind]:
                    r1 = f(v, sub)
                    r2 = Entrypoint(f)(v, sub)
                    self.assertEqual(r1, r2, (f, v, sub))
                    g = f_index if f == f_find else f_rindex
                    if r1 != -1:
                        r3 = Entrypoint(g)(v, sub)
                        self.assertEqual(r1, r3, (g, v, sub))
                    else:
                        with self.assertRaises(ValueError):
                            Entrypoint(g)(v, sub)

                for start in range(-10, 11, 2):
                    for f in [f_find2, f_rfind2]:
                        r1 = f(v, sub, start)
                        r2 = Entrypoint(f)(v, sub, start)
                        self.assertEqual(r1, r2, (f, v, sub, start))
                        g = f_index2 if f == f_find2 else f_rindex2
                        if r1 != -1:
                            r3 = Entrypoint(g)(v, sub, start)
                            self.assertEqual(r1, r3, (g, v, sub, start))
                        else:
                            with self.assertRaises(ValueError):
                                Entrypoint(g)(v, sub, start)
                    for end in range(-10, 11, 2):
                        for f in [f_find3, f_rfind3]:
                            r1 = f(v, sub, start, end)
                            r2 = Entrypoint(f)(v, sub, start, end)
                            self.assertEqual(r1, r2, (f, v, sub, start, end))

                            g = f_index3 if f == f_find3 else f_rindex3
                            if r1 != -1:
                                r3 = Entrypoint(g)(v, sub, start, end)
                                self.assertEqual(r1, r3, (g, v, sub, start, end))
                            else:
                                with self.assertRaises(ValueError):
                                    Entrypoint(g)(v, sub, start, end)

    def test_bytes_mult(self):
        def f_mult(x, n):
            return x * n

        v = b'XyZ'
        for n in [1, 5, 100, 0, -1]:
            r1 = f_mult(v, n)
            r2 = Entrypoint(f_mult)(v, n)
            self.assertEqual(r1, r2, n)

    def test_bytes_contains_bytes(self):
        @Entrypoint
        def f_contains(x, y):
            return x in y

        @Entrypoint
        def f_not_contains(x, y):
            return x not in y

        self.assertTrue(f_contains(b'a', b'asfd'))
        self.assertFalse(f_contains(b'b', b'asfd'))
        self.assertFalse(f_contains(b'b', ListOf(bytes)([b'asfd'])))
        self.assertTrue(f_contains(b'asdf', ListOf(bytes)([b'asdf'])))

        self.assertFalse(f_not_contains(b'a', b'asfd'))
        self.assertTrue(f_not_contains(b'b', b'asfd'))
        self.assertTrue(f_not_contains(b'b', ListOf(bytes)([b'asfd'])))
        self.assertFalse(f_not_contains(b'asdf', ListOf(bytes)([b'asdf'])))

    def test_bytes_replace(self):
        def replace(x: bytes, y: bytes, z: bytes):
            return x.replace(y, z)

        def replace2(x: bytes, y: bytes, z: bytes, i: int):
            return x.replace(y, z, i)

        replaceCompiled = Compiled(replace)
        replace2Compiled = Compiled(replace2)

        values = [b'']
        for _ in range(2):
            for y in [b'ab', b'c', b'ba' * 100]:
                values += [x + y for x in values]

        for s1 in values:
            for s2 in values:
                for s3 in values:
                    r1 = replace(s1, s2, s3)
                    r2 = replaceCompiled(s1, s2, s3)
                    self.assertEqual(r1, r2)

                    for i in [-1, 0, 1, 2]:
                        r1 = replace2(s1, s2, s3, i)
                        r2 = replace2Compiled(s1, s2, s3, i)
                        self.assertEqual(replace2(s1, s2, s3, i), replace2Compiled(s1, s2, s3, i))

    def validate_joining_bytes(self, function, make_obj):
        # Test data, the fields are: description, separator, items, expected output
        test_data = [
            ["simple data",
             b",", [b"1", b"2", b"3"], b"1,2,3"],

            ["longer separator",
             b"---", [b"1", b"2", b"3"], b"1---2---3"],

            ["longer items",
             b"---", [b"aaa", b"bb", b"c"], b"aaa---bb---c"],

            ["empty separator",
             b"", [b"1", b"2", b"3"], b"123"],

            ["everything empty",
             b"", [], b""],

            ["empty list",
             b"a", [], b""],

            ["empty bytes in the items",
             b"--", [b"", b"1", b"3"], b"--1--3"],

            ["blank bytes in the items",
             b"--", [b" ", b"1", b"3"], b" --1--3"],
        ]

        for description, separator, items, expected in test_data:
            res = function(separator, make_obj(items))
            self.assertEqual(expected, res, description)

    def test_bytes_join_for_tuple_of_bytes(self):
        # test passing tuple of bytes
        @Compiled
        def f(sep: bytes, items: TupleOf(bytes)) -> bytes:
            return sep.join(items)

        self.validate_joining_bytes(f, lambda items: TupleOf(bytes)(items))

    def test_bytes_join_for_list_of_bytes(self):
        # test passing list of bytes
        @Compiled
        def f(sep: bytes, items: ListOf(bytes)) -> bytes:
            return sep.join(items)

        self.validate_joining_bytes(f, lambda items: ListOf(bytes)(items))

    def test_bytes_join_for_dict_of_bytes(self):
        # test passing list of bytes
        @Compiled
        def f(sep: bytes, items: Dict(bytes, bytes)) -> bytes:
            return sep.join(items)

        self.validate_joining_bytes(f, lambda items: Dict(bytes, bytes)({i: b"a" for i in items}))

    def test_bytes_join_for_const_dict_of_bytes(self):
        # test passing list of bytes
        @Compiled
        def f(sep: bytes, items: ConstDict(bytes, bytes)) -> bytes:
            return sep.join(items)

        self.validate_joining_bytes(f, lambda items: ConstDict(bytes, bytes)({i: b"a" for i in items}))

    def test_bytes_join_for_bad_types(self):
        """bytes.join supports only joining iterables of bytes."""

        # test passing tuple of ints
        @Compiled
        def f_tup_int(sep: bytes, items: TupleOf(int)) -> bytes:
            return sep.join(items)

        with self.assertRaisesRegex(TypeError, ""):
            f_tup_int(b",", ListOf(int)([1, 2, 3]))

        # test passing list of other types than bytes
        @Compiled
        def f_int(sep: bytes, items: ListOf(str)) -> bytes:
            return sep.join(items)

        with self.assertRaisesRegex(TypeError, ""):
            f_int(b",", ListOf(int)(["1", "2", "3"]))

    def test_bytes_internal_fns(self):
        """
        These are functions that are normally not called directly.
        They are called here in order to improve codecov coverage.
        """
        v = b'a1A'
        self.assertEqual(bytes_isalnum(v), v.isalnum())
        self.assertEqual(bytes_isalpha(v), v.isalpha())
        self.assertEqual(bytes_isdigit(v), v.isdigit())
        self.assertEqual(bytes_islower(v), v.islower())
        self.assertEqual(bytes_isspace(v), v.isspace())
        self.assertEqual(bytes_istitle(v), v.istitle())
        self.assertEqual(bytes_isupper(v), v.isupper())
