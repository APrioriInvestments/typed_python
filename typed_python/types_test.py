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

import math
import numpy
import os
import psutil
import sys
import time
import unittest
import pytest

import typed_python._types as _types

from flaky import flaky
from typed_python import (
    Value,
    Int8, Int16, Int32,
    UInt8, UInt16, UInt32, UInt64,
    Float32, SubclassOf,
    TupleOf, ListOf, OneOf, Tuple, NamedTuple, Dict,
    ConstDict, Alternative, serialize, deserialize, Class,
    TypeFilter, Function, Forward, Set, PointerTo, Entrypoint, Final
)
from typed_python.type_promotion import (
    computeArithmeticBinaryResultType, floatness, bitness, isSignedInt
)
from typed_python.test_util import currentMemUsageMb


AnAlternative = Alternative("AnAlternative", X={'x': int})

AForwardAlternative = Forward("AForwardAlternative")
AForwardAlternative.define(Alternative("AForwardAlternative", Y={}, X={'x': AForwardAlternative}))


def typeFor(t):
    assert not isinstance(t, list), t
    return type(t)


def typeForSeveral(t):
    ts = set(typeFor(a) for a in t)
    if len(ts) == 1:
        return list(ts)[0]
    return OneOf(*ts)


def makeTupleOf(*args):
    if not args:
        return TupleOf(int)()
    return TupleOf(typeForSeveral(args))(args)


def makeNamedTuple(**kwargs):
    if not kwargs:
        return NamedTuple()()
    return NamedTuple(**{k: typeFor(v) for k, v in kwargs.items()})(**kwargs)


def makeTuple(*args):
    if not args:
        return Tuple()()
    return Tuple(*[typeFor(v) for v in args])(args)


def makeDict(d):
    if not d:
        return ConstDict(int, int)()

    return ConstDict(typeForSeveral(d.keys()), typeForSeveral(d.values()))(d)


def makeAlternative(severalDicts):
    types = list(
        set(
            tuple(
                (k, typeFor(v)) for k, v in ntDict.items()
            )
            for ntDict in severalDicts
        )
    )

    alt = Alternative("Alt", **{
        "a_%s" % i: dict(types[i]) for i in range(len(types))
    })

    res = []
    for thing in severalDicts:
        did = False
        for i in range(len(types)):
            try:
                res.append(getattr(alt, "a_%s" % i)(**thing))
                did = True
            except Exception:
                pass

            if did:
                break
    assert len(res) == len(severalDicts)

    return res


def choice(x):
    # numpy.random.choice([1,(1,2)]) blows up because it looks 'multidimensional'
    # so we have to pick from a list of indices
    if not isinstance(x, list):
        x = list(x)
    return x[numpy.random.choice(list(range(len(x))))]


class RandomValueProducer:
    def __init__(self):
        self.levels = {0: [b'1', b'', '2', '', 0, 1, 0.0, 1.0, None, False, True, "a ", "a string", "b string", "b str"]}

    def addEvenly(self, levels, count):
        for level in range(1, levels+1):
            self.addValues(level, count)

    def all(self):
        res = []
        for valueList in self.levels.values():
            res.extend(valueList)
        return res

    def addValues(self, level, count, sublevels=None):
        assert level > 0

        if sublevels is None:
            sublevels = list(range(level))
        sublevels = [x for x in sublevels if x in self.levels]

        assert sublevels

        def picker():
            whichLevel = choice(sublevels)
            try:
                return choice(self.levels[whichLevel])
            except Exception:
                print(self.levels[whichLevel])
                raise

        for _ in range(count):
            val = self.randomValue(picker)
            if not isinstance(val, list):
                val = [val]
            self.levels.setdefault(level, []).extend(val)

    def randomValue(self, picker):
        def randomTuple():
            return makeTuple(*[picker() for i in range(choice([0, 1, 2, 3, 4]))])

        def randomNamedTupleDict():
            return {"x_%s" % i: picker() for i in range(choice([0, 1, 2, 3, 4]))}

        def randomNamedTuple():
            return makeNamedTuple(**randomNamedTupleDict())

        def randomDict():
            return makeDict({picker(): picker() for i in range(choice([0, 1, 2, 3, 4]))})

        def randomTupleOf():
            return makeTupleOf(*[picker() for i in range(choice([0, 1, 2, 3, 4]))])

        def randomAlternative():
            return makeAlternative([randomNamedTupleDict() for i in range(choice([1, 2, 3, 4]))])

        return choice([randomTuple, randomNamedTuple, randomDict, randomTupleOf, randomAlternative, picker])()

    def pickRandomly(self):
        return choice(self.levels[choice(list(self.levels))])


class TypesTests(unittest.TestCase):
    def test_alternative_module(self):
        assert AnAlternative.__module__ == 'typed_python.types_test'
        assert AForwardAlternative.__module__ == 'typed_python.types_test'

    def test_refcount_bug_with_simple_string(self):
        with self.assertRaisesRegex(TypeError, "first argument to refcount '111' not a permitted Type"):
            _types.refcount(111)

        with self.assertRaisesRegex(TypeError, "first argument to refcount 'aa' not a permitted Type"):
            _types.refcount('aa')

    def check_expected_performance(self, elapsed, expected=1.0):
        if os.environ.get('TRAVIS_CI', None) is not None:
            expected = 2 * expected

        self.assertTrue(
            elapsed < expected,
            "Slow Performance: expected to take {expected} sec, but took {elapsed}"
            .format(expected=expected, elapsed=elapsed)
        )

    def test_object_binary_compatibility(self):
        ibc = _types.isBinaryCompatible

        self.assertTrue(ibc(type(None), type(None)))
        self.assertTrue(ibc(Int8, Int8))

        NT = NamedTuple(a=int, b=int)

        class X(NamedTuple(a=int, b=int)):
            pass

        class Y(NamedTuple(a=int, b=int)):
            pass

        self.assertTrue(ibc(X, X))
        self.assertTrue(ibc(X, Y))
        self.assertTrue(ibc(X, NT))
        self.assertTrue(ibc(Y, NT))
        self.assertTrue(ibc(NT, Y))

        self.assertFalse(ibc(OneOf(int, float), OneOf(float, int)))
        self.assertTrue(ibc(OneOf(int, X), OneOf(int, Y)))

    def test_binary_compatibility_incompatible_alternatives(self):
        ibc = _types.isBinaryCompatible

        A1 = Alternative("A1", X={'a': int}, Y={'b': float})
        A2 = Alternative("A2", X={'a': int}, Y={'b': str})

        self.assertTrue(ibc(A1, A1.X))
        self.assertTrue(ibc(A1, A1.Y))
        self.assertTrue(ibc(A1.Y, A1.Y))
        self.assertTrue(ibc(A1.Y, A1))
        self.assertTrue(ibc(A1.X, A1))
        self.assertFalse(ibc(A1.X, A1.Y))

        self.assertFalse(ibc(A1, A2))
        self.assertFalse(ibc(A1.X, A2.X))
        self.assertFalse(ibc(A1.Y, A2.Y))

    def test_binary_compatibility_compatible_alternatives(self):
        ibc = _types.isBinaryCompatible

        A1 = Alternative("A1", X={'a': int}, Y={'b': float})
        A2 = Alternative("A2", X={'a': int}, Y={'b': float})

        self.assertTrue(ibc(A1.X, A2.X))
        self.assertTrue(ibc(A1.Y, A2.Y))

        self.assertFalse(ibc(A1.X, A2.Y))
        self.assertFalse(ibc(A1.Y, A2.X))

    def test_callable_alternatives(self):
        def myCall(self):
            if self.matches.One:
                return 1
            elif self.matches.Two:
                return 2
            else:
                raise TypeError("Unexpected alternative kind")

        alt = Alternative("alts", One={}, Two={}, __call__=myCall)

        one = alt.One()
        self.assertEqual(one(), 1)

        two = alt.Two()
        self.assertEqual(two(), 2)

        alt = Alternative("alts", One={}, Two={}, myCall=myCall)
        with self.assertRaises(TypeError):
            one = alt.One()
            one()

        with self.assertRaises(TypeError):
            two = alt.Two()
            two()

    def test_object_bytecounts(self):
        self.assertEqual(_types.bytecount(type(None)), 0)
        self.assertEqual(_types.bytecount(Int8), 1)
        self.assertEqual(_types.bytecount(int), 8)

    def test_type_stringification(self):
        self.assertEqual(str(_types.Int8), "<class 'typed_python._types.Int8'>")
        self.assertEqual(str(Tuple(int)), "<class 'Tuple(int)'>")

    def test_tuple_of(self):
        tupleOfInt = TupleOf(int)
        i = tupleOfInt(())
        i = tupleOfInt((1, 2, 3))

        self.assertEqual(len(i), 3)
        self.assertEqual(tuple(i), (1, 2, 3))

        for x in range(10):
            self.assertEqual(
                tuple(tupleOfInt(tuple(range(x)))),
                tuple(range(x))
            )

        with self.assertRaisesRegex(AttributeError, "has no attribute 'x'"):
            tupleOfInt((1, 2, 3)).x = 2

    def test_one_of_and_types(self):
        # when we use types in OneOf, we need to wrap them in Value. Otherwise we can't
        # tell the difference between OneOf(int) and OneOf(Value(int))
        with self.assertRaisesRegex(Exception, "arguments must be types or simple values"):
            OneOf(lambda: 10)

        X = OneOf(Value(int), Value(Alternative))

        self.assertEqual(OneOf(Value(int)).__qualname__, "OneOf(<class 'int'>)")

        # these sould work
        X(int)
        X(Alternative)

        # these should not
        with self.assertRaises(Exception):
            X(float)

        with self.assertRaises(Exception):
            X(10)

        self.assertNotEqual(Value(int), Value(float))

        X2 = OneOf(Value(int), Value(float))
        X2(int)
        X2(float)

    def test_const_dict_add_mappable(self):
        T = ConstDict(int, int)

        self.assertEqual(
            T({1: 2}) + {2: 3},
            {1: 2, 2: 3}
        )

        self.assertEqual(
            T({1: 2}) + ConstDict(int, OneOf(int, None))({2: 3}),
            {1: 2, 2: 3}
        )

        self.assertEqual(
            T({1: 2}) + ConstDict(int, OneOf(int, None))({2: 3}),
            {1: 2, 2: 3}
        )

    def test_const_dict_add_mappable_with_exceptions(self):
        T = ConstDict(int, TupleOf(int))

        to = TupleOf(int)((1,))

        self.assertEqual(_types.refcount(to), 1)

        t = T({1: to})

        self.assertEqual(_types.refcount(to), 2)

        t = t + {2: (1, 2, 3)}

        self.assertEqual(t, {1: (1,), 2: (1, 2, 3)})

        try:
            t = t + {3: to, 4: "hi"}
        except TypeError:
            pass

        # the refcount of 'to' should not have increased
        self.assertEqual(_types.refcount(to), 2)

    def test_one_of_alternative(self):
        X = Alternative("X", V={'a': int})
        Ox = OneOf(None, X)

        self.assertEqual(Ox(X.V(a=10)), X.V(a=10))

    def test_one_of_py_subclass(self):
        class X(NamedTuple(x=int)):
            def f(self):
                return self.x

        Ox = OneOf(None, X)

        self.assertEqual(NamedTuple(x=int)(x=10).x, 10)
        self.assertEqual(X(x=10).f(), 10)
        self.assertEqual(Ox(X(x=10)).f(), 10)

    def test_one_of_distinguishes_py_subclasses(self):
        class X(NamedTuple(x=int)):
            def f(self):
                return self.x

        class X2(NamedTuple(x=int)):
            def f(self):
                return self.x + 2

        XorX2 = OneOf(X, X2)

        self.assertTrue(isinstance(XorX2(X()), X))
        self.assertTrue(isinstance(XorX2(X2()), X2))

    def test_function_as_type_arg(self):
        @Function
        def f(x: int):
            return x

        self.assertEqual(type(f).__typed_python_category__, "Function")

        aTup = Tuple(type(f))
        self.assertEqual(aTup.ElementTypes[0], type(f))
        self.assertEqual(aTup.ElementTypes[0].__typed_python_category__, "Function")

    @flaky(max_runs=3, min_passes=1)
    @pytest.mark.skipif('sys.platform=="darwin"')
    def test_tuple_of_tuple_of_perf(self):
        tupleOfInt = TupleOf(int)
        tupleOfTupleOfInt = TupleOf(tupleOfInt)

        pyVersion = (1, 2, 3), (1, 2, 3, 4)
        nativeVersion = tupleOfTupleOfInt(pyVersion)

        self.assertEqual(len(nativeVersion), 2)
        self.assertEqual(len(nativeVersion[0]), 3)
        self.assertEqual(tuple(tuple(x) for x in nativeVersion), pyVersion)

        bigTup = tupleOfInt(list(range(1000)))

        t0 = time.time()
        t = (bigTup, bigTup, bigTup, bigTup, bigTup)
        for i in range(1000000):
            tupleOfTupleOfInt(t)

        elapsed = time.time() - t0
        print("Took ", elapsed, " to do 1mm")
        self.check_expected_performance(elapsed)

    @flaky(max_runs=3, min_passes=1)
    @pytest.mark.skipif('sys.platform=="darwin"')
    def test_tuple_of_string_perf(self):
        t = NamedTuple(a=str, b=str)

        t0 = time.time()
        for i in range(1000000):
            t(a="a", b="b").a

        elapsed = time.time() - t0
        print("Took ", elapsed, " to do 1mm")
        self.check_expected_performance(elapsed, expected=1.5)

    def test_default_initializer_oneof(self):
        x = OneOf(None, int)
        self.assertTrue(x() is None, repr(x()))

    def test_tuple_of_various_things(self):
        for thing, typeOfThing in [("hi", str), (b"somebytes", bytes),
                                   (1.0, float), (2, int),
                                   (None, type(None))
                                   ]:
            tupleType = TupleOf(typeOfThing)
            t = tupleType((thing,))
            self.assertTrue(type(t[0]) is typeOfThing)
            self.assertEqual(t[0], thing)

    def test_tuple_assign_fails(self):
        with self.assertRaisesRegex(TypeError, "does not support item assignment"):
            (1, 2, 3)[10] = 20
        with self.assertRaisesRegex(TypeError, "does not support item assignment"):
            TupleOf(int)((1, 2, 3))[10] = 20

    def test_list_of(self):
        L = ListOf(int)
        self.assertEqual(L.__qualname__, "ListOf(int)")

        l1 = L([1, 2, 3, 4])

        self.assertEqual(l1[0], 1)
        self.assertEqual(l1[-1], 4)

        l1[0] = 10
        self.assertEqual(l1[0], 10)

        l1[-1] = 11
        self.assertEqual(l1[3], 11)

        with self.assertRaisesRegex(IndexError, "index out of range"):
            l1[100] = 20

        l2 = L((10, 2, 3, 11))

        self.assertEqual(l1, l2)
        self.assertNotEqual(l1, (10, 2, 3, 11))
        self.assertEqual(l1, [10, 2, 3, 11])

        self.assertEqual(str(l1), str([10, 2, 3, 11]))

        l3 = l1 + l2
        self.assertEqual(l3, [10, 2, 3, 11, 10, 2, 3, 11])

        l3.append(23)
        self.assertEqual(l3, [10, 2, 3, 11, 10, 2, 3, 11, 23])

    def test_list_resize(self):
        l1 = ListOf(TupleOf(int))()

        l1.resize(10)
        self.assertEqual(l1.reserved(), 10)
        self.assertEqual(len(l1), 10)

        emptyTup = TupleOf(int)()
        aTup = TupleOf(int)((1, 2, 3))

        self.assertEqual(list(l1), [emptyTup] * 10)
        l1.resize(20, aTup)
        self.assertEqual(list(l1), [emptyTup] * 10 + [aTup] * 10)

        self.assertEqual(_types.refcount(aTup), 11)

        self.assertEqual(l1.pop(15), aTup)
        self.assertEqual(l1.pop(5), emptyTup)

        self.assertEqual(_types.refcount(aTup), 10)

        l1.resize(15)

        with self.assertRaises(IndexError):
            l1.pop(100)

        self.assertEqual(_types.refcount(aTup), 7)  # 6 in the list because we popped at '5'

        l1.pop()

        self.assertEqual(_types.refcount(aTup), 6)

        # this pops one of the empty tuples
        l1.pop(-10)

        self.assertEqual(_types.refcount(aTup), 6)

        l1.clear()
        self.assertEqual(len(l1), 0)

    def test_one_of(self):
        o = OneOf(None, str)

        self.assertEqual(o("hi"), "hi")
        self.assertTrue(o(None) is None)

        # TODO: investigate and correct: with the ordering 1, True, the assertion o(True) is True fails
        # o = OneOf(None, "hi", 1.5, 1, True, b"hi2")
        o = OneOf(None, "hi", 1.5, True, 1, b"hi2")

        self.assertTrue(o(None) is None)
        self.assertTrue(o("hi") == "hi")
        self.assertTrue(o(b"hi2") == b"hi2")
        self.assertTrue(o(1.5) == 1.5)
        self.assertTrue(o(1) == 1)
        self.assertIs(o(True), True)

        with self.assertRaises(TypeError):
            o("hi2")
        with self.assertRaises(TypeError):
            o(b"hi")
        with self.assertRaises(TypeError):
            o(3)
        with self.assertRaises(TypeError):
            o(False)

    def test_use_of_None(self):
        o1 = OneOf(None, str)
        o2 = OneOf(type(None), str)
        assert o1.Types[0] == o2.Types[0] == type(None)  # noqa

        p1 = PointerTo(None)
        p2 = PointerTo(type(None))
        assert p1.ElementType == p2.ElementType == type(None)  # noqa

        s1 = Set(None)
        s2 = Set(type(None))
        assert s1.ElementType == s2.ElementType == type(None)  # noqa

    def test_dict_equality_with_python(self):
        assert Dict(int, int)({1: 2}) == {1: 2}

    def test_ordering(self):
        # TODO: investigate and correct: with the ordering 1, True, the assertion o(True) is True fails
        o = OneOf(None, "hi", 1.5, True, 1, b"hi2")

        self.assertIs(o(True), True)

    def test_one_of_flattening(self):
        self.assertEqual(OneOf(OneOf(None, 1.0), OneOf(2.0, 3.0)), OneOf(None, 1.0, 2.0, 3.0))

    def test_one_of_order_matters(self):
        self.assertNotEqual(OneOf(1.0, 2.0), OneOf(2.0, 1.0))

    def test_type_filter(self):
        EvenInt = TypeFilter(int, lambda i: i % 2 == 0)

        self.assertTrue(isinstance(2, EvenInt))
        self.assertFalse(isinstance(1, EvenInt))
        self.assertFalse(isinstance(2.0, EvenInt))

        EvenIntegers = TupleOf(EvenInt)

        e = EvenIntegers(())
        e2 = e + (2, 4, 0)

        with self.assertRaises(TypeError):
            EvenIntegers((1,))

        with self.assertRaises(TypeError):
            e2 + (1,)

    def test_tuple_of_one_of_fixed_size(self):
        t = TupleOf(OneOf(0, 1, 2, 3, 4))

        ints = tuple([x % 5 for x in range(1000000)])

        typedInts = t(ints)

        self.assertEqual(len(serialize(t, typedInts)), len(ints) * 2 + 6)  # 3 bytes for extra flags
        self.assertEqual(tuple(typedInts), ints)

    def test_tuple_of_one_of_multi(self):
        t = TupleOf(OneOf(int, bool))

        someThings = tuple([100 + x % 5 if x % 17 != 0 else bool(x%19) for x in range(1000000)])

        typedThings = t(someThings)

        self.assertEqual(
            len(serialize(t, typedThings)),
            sum(3 if isinstance(t, bool) else 4 for t in someThings) +
            2 +  # two bytes for begin / end flags
            2 +  # two bytes for the id
            2    # two bytes for the size
        )

        self.assertEqual(tuple(typedThings), someThings)

    def test_compound_oneof(self):
        producer = RandomValueProducer()
        producer.addEvenly(1000, 2)

        for _ in range(1000):
            vals = (producer.pickRandomly(), producer.pickRandomly(), producer.pickRandomly())

            a = OneOf(vals[0], vals[1], type(vals[2]))

            for v in vals:
                self.assertEqual(a(v), v, (a(v), v))

            tup = TupleOf(a)
            tupInst = tup(vals)

            for i in range(len(vals)):
                self.assertEqual(tupInst[i], vals[i], vals)

    def test_one_of_conversion_failure(self):
        o = OneOf(None, str)

        with self.assertRaises(TypeError):
            o(b"bytes")

    def test_one_of_in_tuple(self):
        t = Tuple(OneOf(None, str), str)

        self.assertEqual(t(("hi", "hi2"))[0], "hi")
        self.assertEqual(t(("hi", "hi2"))[1], "hi2")
        self.assertEqual(t((None, "hi2"))[1], "hi2")
        self.assertEqual(t((None, "hi2"))[0], None)
        with self.assertRaises(TypeError):
            t((None, None))
        with self.assertRaises(IndexError):
            t((None, "hi2"))[2]

    def test_one_of_composite(self):
        t = OneOf(TupleOf(str), TupleOf(float))

        self.assertIsInstance(t((1.0, 2.0)), TupleOf(float))
        self.assertIsInstance(t(("1.0", "2.0")), TupleOf(str))

        with self.assertRaises(TypeError):
            t((1.0, "2.0"))

    def test_comparisons_in_one_of(self):
        t = OneOf(None, float)

        def map(x):
            if x is None:
                return -1000000.0
            else:
                return x

        lt = lambda a, b: map(a) < map(b)
        le = lambda a, b: map(a) <= map(b)
        eq = lambda a, b: map(a) == map(b)
        ne = lambda a, b: map(a) != map(b)
        gt = lambda a, b: map(a) > map(b)
        ge = lambda a, b: map(a) >= map(b)

        funcs = [lt, le, eq, ne, gt, ge]
        ts = [None, 1.0, 2.0, 3.0]

        for f in funcs:
            for t1 in ts:
                for t2 in ts:
                    self.assertTrue(f(t1, t2) is f(t(t1), t(t2)))

    def test_comparisons_equivalence(self):
        t = TupleOf(OneOf(None, str, bytes, float, int, TupleOf(int), bool),)

        self.assertEqual(t((3,))[0], 3)
        self.assertEqual(t(((3,),))[0], TupleOf(int)((3,)))

        def lt(a, b):
            return a < b

        def le(a, b):
            return a <= b

        def eq(a, b):
            return a == b

        def ne(a, b):
            return a != b

        def gt(a, b):
            return a > b

        def ge(a, b):
            return a >= b

        funcs = [lt, le, eq, ne, gt, ge]

        tgroups = [
            [1.0, 2.0, 3.0],
            [1, 2, 3],
            [True, False],
            ["a", "b", "ab", "bb", "ba", "aaaaaaa", "", "asdf"],
            ["1", "2", "3", "12", "13", "23", "24", "123123", "0", ""],
            [b"a", b"b", b"ab", b"bb", b"ba", b"aaaaaaa", b"", b"asdf"],
            [(1, 2), (1, 2, 3), (), (1, 1), (1,)]
        ]

        for ts in tgroups:
            for f in funcs:
                for t1 in ts:
                    for t2 in ts:
                        self.assertTrue(
                            f(t1, t2) is f(t((t1,)), t((t2,))),
                            (f, t1, t2, t((t1,)), t((t2,)), f(t1, t2), f(t((t1,)), t((t2,))))
                        )

    def test_const_dict(self):
        t = ConstDict(str, str)

        self.assertEqual(len(t()), 0)
        self.assertEqual(len(t({})), 0)
        self.assertEqual(len(t({'a': 'b'})), 1)
        self.assertEqual(t({'a': 'b'})['a'], 'b')
        self.assertEqual(t({'a': 'b', 'b': 'c'})['b'], 'c')

        self.assertTrue("a" in deserialize(t, serialize(t, t({'a': 'b'}))))

        self.assertTrue("a" in deserialize(t, serialize(t, t({'a': 'b', 'b': 'c'}))))
        self.assertTrue("a" in deserialize(t, serialize(t, t({'a': 'b', 'b': 'c', 'c': 'd'}))))
        self.assertTrue("a" in deserialize(t, serialize(t, t({'a': 'b', 'b': 'c', 'c': 'd', 'd': 'e'}))))
        self.assertTrue("c" in deserialize(t, serialize(t, t({'a': 'b', 'b': 'c', 'c': 'd', 'def': 'e'}))))
        self.assertTrue("def" in deserialize(t, serialize(t, t({'a': 'b', 'b': 'c', 'c': 'd', 'def': 'e'}))))

    def test_const_dict_get(self):
        a = ConstDict(str, str)({'a': 'b', 'c': 'd'})

        self.assertEqual(a.get('a'), 'b')
        self.assertEqual(a.get('asdf'), None)
        self.assertEqual(a.get('asdf', 20), 20)

    def test_const_dict_items_keys_and_values(self):
        a = ConstDict(str, str)({'a': 'b', 'c': 'd'})

        self.assertEqual(sorted(a.items()), [('a', 'b'), ('c', 'd')])
        self.assertEqual(sorted(a.keys()), ['a', 'c'])
        self.assertEqual(sorted(a.values()), ['b', 'd'])

    def test_empty_string(self):
        a = ConstDict(str, str)({'a': ''})

        print(a['a'])

    def test_dict_to_oneof(self):
        t = ConstDict(str, OneOf("A", "B", "ABCDEF"))
        a = t({'a': 'A', 'b': 'ABCDEF'})

        self.assertEqual(a['a'], "A")
        self.assertEqual(a['b'], "ABCDEF")

        self.assertEqual(a, deserialize(t, serialize(t, a)))

    def test_dict_assign_coercion(self):
        T = Dict(str, int)

        t = T()
        t["hi"] = 1.5

        self.assertEqual(t, {"hi": 1})

    def test_dict_update(self):
        T = Dict(str, int)
        t = T()

        t.update({"hi": 0})
        self.assertEqual(t, {"hi": 0})

        t.update({"hi": 1})
        self.assertEqual(t, {"hi": 1})

        t.update({"hi": 1.4})
        self.assertEqual(t, {"hi": 1})

        t.update(T({"hi": 2}))
        self.assertEqual(t, {"hi": 2})

    def test_dict_clear(self):
        T = Dict(str, TupleOf(int))

        a = T()
        aTup = TupleOf(int)([1, 2, 3])

        self.assertEqual(_types.refcount(aTup), 1)
        a["hi"] = aTup
        self.assertEqual(_types.refcount(aTup), 2)
        a.clear()
        self.assertEqual(_types.refcount(aTup), 1)

        self.assertNotIn("hi", a)

        a["bye"] = aTup
        a["good"] = (1, 2, 3, 4)

        a.clear()

        self.assertEqual(len(a), 0)

    def test_dict_clear_large(self):
        T = Dict(str, str)

        for i in range(100):
            d = T()

            for passIx in range(3):
                for j in range(i + 1):
                    d[str(j)] = str(j)

                for j in range(i + 1):
                    assert str(j) in d

                    if j % 4 in (0, 1, 2):
                        del d[str(j)]

                d.clear()

            self.assertEqual(len(d), 0)

            self.assertTrue("0" not in d)

            d["1"] = "1"
            self.assertTrue("1" in d)

    def test_deserialize_primitive(self):
        x = deserialize(str, serialize(str, "a"))
        self.assertTrue(isinstance(x, str))

    def test_dict_containment(self):
        for _ in range(100):
            producer = RandomValueProducer()
            producer.addEvenly(20, 2)

            values = producer.all()

            for v in values:
                if str(type(v))[:17] == "<class 'ConstDict":
                    v = deserialize(type(v), serialize(type(v), v))
                    for k in v:
                        self.assertTrue(k in v)

    def test_dict_keys_values_and_items(self):
        # check that 'Dict().keys' behaves correctly.
        T = Dict(str, int)
        aDict = T({"hi": 10, "bye": 20})

        self.assertEqual(str(aDict.keys()), 'dict_keys(["hi", "bye"])')
        self.assertEqual(str(aDict.values()), 'dict_values([10, 20])')
        self.assertEqual(str(aDict.items()), 'dict_items([("hi", 10), ("bye", 20)])')

        self.assertEqual(aDict.keys(), T({"hi": 10, "bye": 30}).keys())
        self.assertNotEqual(aDict.keys(), aDict.values())
        self.assertEqual(aDict.items(), aDict.items())

        # you can't use dict methods on iterators, even though
        # they have the same underlying representation
        with self.assertRaises(TypeError):
            aDict.keys().keys()
        with self.assertRaises(TypeError):
            aDict.keys().values()
        with self.assertRaises(TypeError):
            aDict.keys().items()
        with self.assertRaises(TypeError):
            aDict.keys().setdefault("hi", 10)
        with self.assertRaises(TypeError):
            aDict.keys()['hi']

        self.assertIn('hi', aDict.keys())
        self.assertNotIn('boo', aDict.keys())

        with self.assertRaises(TypeError):
            'hi' in aDict.items()
        with self.assertRaises(TypeError):
            'hi' in aDict.values()

        self.assertIn(10, aDict.values())
        self.assertNotIn(11, aDict.values())

        with self.assertRaises(TypeError):
            10 in aDict.keys()
        with self.assertRaises(TypeError):
            10 in aDict.items()

        self.assertIn(('hi', 10), aDict.items())
        self.assertNotIn(('asdf', 10), aDict.items())

        with self.assertRaises(TypeError):
            ('hi', 10) in aDict.keys()

        with self.assertRaises(TypeError):
            ('hi', 10) in aDict.values()

        with self.assertRaises(Exception):
            aDict.keys().keys()

    def test_const_dict_keys_values_and_items(self):
        # check that 'ConstDict().keys' behaves correctly.
        T = ConstDict(str, int)
        aDict = T({"hi": 10, "bye": 20})

        self.assertEqual(str(aDict.keys()), 'const_dict_keys(["bye", "hi"])')
        self.assertEqual(str(aDict.values()), 'const_dict_values([20, 10])')
        self.assertEqual(str(aDict.items()), 'const_dict_items([("bye", 20), ("hi", 10)])')

        self.assertEqual(aDict.keys(), T({"hi": 10, "bye": 30}).keys())
        self.assertNotEqual(aDict.keys(), aDict.values())

        # this is odd, but follows python's builtin behavior
        self.assertNotEqual(aDict.values(), aDict.values())
        self.assertNotEqual({1: 2}.values(), {1: 2}.values())

        self.assertEqual(aDict.items(), aDict.items())

        # you can't use dict methods on iterators, even though
        # they have the same underlying representation
        with self.assertRaises(TypeError):
            aDict.keys().keys()
        with self.assertRaises(TypeError):
            aDict.keys().values()
        with self.assertRaises(TypeError):
            aDict.keys().items()
        with self.assertRaises(TypeError):
            aDict.keys()['hi']

        self.assertIn('hi', aDict.keys())
        self.assertNotIn('boo', aDict.keys())

        with self.assertRaises(TypeError):
            'hi' in aDict.items()
        with self.assertRaises(TypeError):
            'hi' in aDict.values()

        self.assertIn(10, aDict.values())
        self.assertNotIn(11, aDict.values())

        with self.assertRaises(TypeError):
            10 in aDict.keys()
        with self.assertRaises(TypeError):
            10 in aDict.items()

        self.assertIn(('hi', 10), aDict.items())
        self.assertNotIn(('asdf', 10), aDict.items())

        with self.assertRaises(TypeError):
            ('hi', 10) in aDict.keys()

        with self.assertRaises(TypeError):
            ('hi', 10) in aDict.values()

        with self.assertRaises(Exception):
            aDict.keys().keys()

    def test_const_dict_mixed(self):
        t = ConstDict(str, int)
        self.assertTrue(t({"a": 10})["a"] == 10)

        t = ConstDict(int, str)
        self.assertTrue(t({10: "a"})[10] == "a")

    def test_const_dict_comparison(self):
        t = ConstDict(str, str)

        self.assertEqual(t({'a': 'b'}), t({'a': 'b'}))
        self.assertLess(t({}), t({'a': 'b'}))

    def test_const_dict_lookup(self):
        for type_to_use, vals in [(int, list(range(20))),
                                  (bytes, [b'1', b'2', b'3', b'4', b'5'])]:
            t = ConstDict(type_to_use, type_to_use)

            for _ in range(10):
                ks = list(vals)
                vs = list(vals)

                numpy.random.shuffle(ks)
                numpy.random.shuffle(vs)

                py_d = {}
                for i in range(len(ks)):
                    py_d[ks[i]] = vs[i]

                typed_d = t(py_d)

                for k in py_d:
                    self.assertEqual(py_d[k], typed_d[k])

                last_k = None
                for k in typed_d:
                    assert last_k is None or k > last_k, (k, last_k)
                    last_k = k

    def test_const_dict_lookup_time(self):
        int_dict = ConstDict(int, int)

        d = int_dict({k: k for k in range(1000000)})

        for k in range(1000000):
            self.assertTrue(k in d)
            self.assertTrue(d[k] == k)

    def test_const_dict_of_dict(self):
        int_dict = ConstDict(int, int)
        int_dict_2 = ConstDict(int_dict, int_dict)

        d = int_dict({1: 2})
        d2 = int_dict({1: 2, 3: 4})

        big = int_dict_2({d: d2})

        self.assertTrue(d in big)
        self.assertTrue(d2 not in big)
        self.assertTrue(big[d] == d2)

    @flaky(max_runs=3, min_passes=1)
    def test_dict_hash_perf(self):
        str_dict = ConstDict(str, str)

        s = str_dict({'a' * 1000000: 'b' * 1000000})

        t0 = time.time()
        for k in range(1000000):
            hash(s)

        elapsed = time.time() - t0
        print(elapsed, " to do 1mm")
        self.check_expected_performance(elapsed)

    def test_mutable_dict_not_hashable(self):
        with self.assertRaisesRegex(Exception, "not hashable"):
            hash(Dict(int, int)())

    @flaky(max_runs=3, min_passes=1)
    def test_const_dict_str_perf(self):
        t = ConstDict(str, str)

        t0 = time.time()
        for i in range(50000):
            t({str(k): str(k+1) for k in range(10)})

        elapsed = time.time() - t0
        print("Took ", elapsed, " to do 1mm")
        self.check_expected_performance(elapsed)

    @flaky(max_runs=3, min_passes=1)
    def test_const_dict_int_perf(self):
        t = ConstDict(int, int)

        t0 = time.time()
        for i in range(100000):
            t({k: k+1 for k in range(10)})

        elapsed = time.time() - t0
        print("Took ", elapsed, " to do 1mm")
        self.check_expected_performance(elapsed)

    def test_const_dict_iter_int(self):
        t = ConstDict(int, int)

        aDict = t({k: k+1 for k in range(100)})
        for k in aDict:
            self.assertEqual(aDict[k], k+1)

    def test_const_dict_iter_str(self):
        t = ConstDict(str, str)

        aDict = t({str(k): str(k+1) for k in range(100)})
        for k in aDict:
            self.assertEqual(aDict[str(k)], str(int(k)+1))

    def test_alternative_matcher_type(self):
        A = Alternative("A", X=dict(x=int))

        assert type(A.X().matches).__name__ == "AlternativeMatcher(X)"
        assert type(A.X().matches).__typed_python_category__ == "AlternativeMatcher"
        assert type(A.X().matches).Alternative is A
        assert A.X().matches.X
        assert not A.X().matches.NotX

    def test_alternative_bytecounts(self):
        alt = Alternative(
            "Empty",
            X={},
            Y={}
        )

        self.assertEqual(_types.bytecount(alt), 1)
        self.assertEqual(_types.bytecount(alt.X), 1)
        self.assertEqual(_types.bytecount(alt.Y), 1)

    def test_alternatives_with_Bytes(self):
        alt = Alternative(
            "Alt",
            x_0={'a': bytes}
        )
        self.assertEqual(alt.x_0(a=b''), alt.x_0(a=b''))

    def test_alternatives_with_str_func(self):
        alt = Alternative(
            "Alt",
            x_0={'a': bytes},
            f=lambda self: 1,
            __str__=lambda self: "not_your_usual_str"
        )

        self.assertEqual(alt.x_0().f(), 1)
        self.assertEqual(str(alt.x_0()), "not_your_usual_str")

    def test_alternatives_with_str_and_repr(self):
        A = Alternative(
            "A",
            X={},
            __str__=lambda self: "alt_str",
            __repr__=lambda self: "alt_repr"
        )

        self.assertEqual(str(A.X()), "alt_str")
        self.assertEqual(repr(A.X()), "alt_repr")
        self.assertEqual(repr(ListOf(A)([A.X()])), "[alt_repr]")
        self.assertEqual(str(ListOf(A)([A.X()])), "[alt_repr]")

    def test_alternative_magic_methods(self):
        A_attrs = {"q": "value-q", "z": "value-z"}

        def A_getattr(s, n):
            if n not in A_attrs:
                raise AttributeError(f"no attribute {n}")
            return A_attrs[n]

        def A_setattr(s, n, v):
            A_attrs[n] = v

        def A_delattr(s, n):
            A_attrs.pop(n, None)

        A = Alternative("A", a={'a': int}, b={'b': str},
                        __bool__=lambda self: self.matches.b,
                        __str__=lambda self: "str",
                        __repr__=lambda self: "repr",
                        __call__=lambda self: "call",
                        __len__=lambda self: 42,
                        __contains__=lambda self, item: not not item,

                        __add__=lambda lhs, rhs: A.b("add"),
                        __sub__=lambda lhs, rhs: A.b("sub"),
                        __mul__=lambda lhs, rhs: A.b("mul"),
                        __matmul__=lambda lhs, rhs: A.b("matmul"),
                        __truediv__=lambda lhs, rhs: A.b("truediv"),
                        __floordiv__=lambda lhs, rhs: A.b("floordiv"),
                        __divmod=lambda lhs, rhs: A.b("divmod"),
                        __mod__=lambda lhs, rhs: A.b("mod"),
                        __pow__=lambda lhs, rhs: A.b("pow"),
                        __lshift__=lambda lhs, rhs: A.b("lshift"),
                        __rshift__=lambda lhs, rhs: A.b("rshift"),
                        __and__=lambda lhs, rhs: A.b("and"),
                        __or__=lambda lhs, rhs: A.b("or"),
                        __xor__=lambda lhs, rhs: A.b("xor"),

                        __neg__=lambda self: A.b("neg"),
                        __pos__=lambda self: A.b("pos"),
                        __invert__=lambda self: A.b("invert"),

                        __abs__=lambda self: A.b("abs"),
                        __int__=lambda self: 123,
                        __float__=lambda self: 234.5,
                        __index__=lambda self: 124,
                        __complex__=lambda self: complex(1, 2),
                        __round__=lambda self: 6,
                        __trunc__=lambda self: 7,
                        __floor__=lambda self: 8,
                        __ceil__=lambda self: 9,

                        __bytes__=lambda self: b'bytes',
                        __format__=lambda self, spec: "format",
                        __getattr__=A_getattr,
                        __setattr__=A_setattr,
                        __delattr__=A_delattr,
                        )

        self.assertEqual(A.a().__bool__(), False)
        self.assertEqual(bool(A.a()), False)
        self.assertEqual(A.b().__bool__(), True)
        self.assertEqual(bool(A.b()), True)
        self.assertEqual(A.a().__str__(), "str")
        self.assertEqual(str(A.a()), "str")
        self.assertEqual(A.a().__repr__(), "repr")
        self.assertEqual(repr(A.a()), "repr")
        self.assertEqual(A.a().__call__(), "call")
        self.assertEqual(A.a()(), "call")
        self.assertEqual(A.a().__contains__(0), False)
        self.assertEqual(A.a().__contains__(1), True)
        self.assertEqual(0 in A.a(), False)
        self.assertEqual(1 in A.a(), True)
        self.assertEqual(A.a().__len__(), 42)
        self.assertEqual(len(A.a()), 42)

        self.assertEqual(A.a().__add__(A.a()).Name, "b")
        self.assertEqual(A.a().__add__(A.a()).b, "add")
        self.assertEqual((A.a() + A.a()).Name, "b")
        self.assertEqual((A.a() + A.a()).b, "add")
        self.assertEqual(A.a().__sub__(A.a()).Name, "b")
        self.assertEqual(A.a().__sub__(A.a()).b, "sub")
        self.assertEqual((A.a() - A.a()).Name, "b")
        self.assertEqual((A.a() - A.a()).b, "sub")
        self.assertEqual((A.a() * A.a()).Name, "b")
        self.assertEqual((A.a() * A.a()).b, "mul")
        self.assertEqual((A.a() @ A.a()).Name, "b")
        self.assertEqual((A.a() @ A.a()).b, "matmul")
        self.assertEqual((A.a() / A.a()).Name, "b")
        self.assertEqual((A.a() / A.a()).b, "truediv")
        self.assertEqual((A.a() // A.a()).Name, "b")
        self.assertEqual((A.a() // A.a()).b, "floordiv")
        self.assertEqual((A.a() % A.a()).Name, "b")
        self.assertEqual((A.a() % A.a()).b, "mod")
        self.assertEqual((A.a() ** A.a()).Name, "b")
        self.assertEqual((A.a() ** A.a()).b, "pow")
        self.assertEqual((A.a() >> A.a()).Name, "b")
        self.assertEqual((A.a() >> A.a()).b, "rshift")
        self.assertEqual((A.a() << A.a()).Name, "b")
        self.assertEqual((A.a() << A.a()).b, "lshift")
        self.assertEqual((A.a() & A.a()).Name, "b")
        self.assertEqual((A.a() & A.a()).b, "and")
        self.assertEqual((A.a() | A.a()).Name, "b")
        self.assertEqual((A.a() | A.a()).b, "or")
        self.assertEqual((A.a() ^ A.a()).Name, "b")
        self.assertEqual((A.a() ^ A.a()).b, "xor")
        self.assertEqual((+A.a()).Name, "b")
        self.assertEqual((+A.a()).b, "pos")
        self.assertEqual((-A.a()).Name, "b")
        self.assertEqual((-A.a()).b, "neg")
        self.assertEqual((~A.a()).Name, "b")
        self.assertEqual((~A.a()).b, "invert")
        self.assertEqual(abs(A.a()).Name, "b")
        self.assertEqual(abs(A.a()).b, "abs")
        self.assertEqual(int(A.a()), 123)
        self.assertEqual(float(A.a()), 234.5)
        self.assertEqual(range(1000)[1:A.a():2], range(1, 124, 2))
        self.assertEqual(complex(A.a()), 1+2j)
        self.assertEqual(round(A.a()), 6)
        self.assertEqual(math.trunc(A.a()), 7)
        self.assertEqual(math.floor(A.a()), 8)
        self.assertEqual(math.ceil(A.a()), 9)

        self.assertEqual(bytes(A.a()), b"bytes")
        self.assertEqual(format(A.a()), "format")

        self.assertEqual(A.a().Name, "a")
        self.assertEqual(A.a().q, "value-q")
        self.assertEqual(A.b().z, "value-z")
        A.a().q = "changedvalue for q"
        self.assertEqual(A.b().q, "changedvalue for q")
        with self.assertRaises(AttributeError):
            print(A.a().invalid)
        del A.a().z
        with self.assertRaises(AttributeError):
            print(A.a().z)
        A.a().Name = "can't change Name"
        self.assertEqual(A.a().Name, "a")
        d = dir(A.a())
        self.assertGreater(len(d), 50)

        A2_items = dict()

        def A2_setitem(self, i, v):
            A2_items[i] = v
            return 0

        A2 = Alternative("A2", a={'a': int}, b={'b': str},
                         __getattribute__=A_getattr,
                         __setattr__=A_setattr,
                         __delattr__=A_delattr,
                         __dir__=lambda self: list(A_attrs.keys()),
                         __getitem__=lambda self, i: A2_items.get(i, i),
                         __setitem__=A2_setitem
                         )

        self.assertEqual(A2.b().q, "changedvalue for q")
        A2.a().Name = "can change Name"
        self.assertEqual(A2.b().Name, "can change Name")
        self.assertEqual(dir(A2.b()), ["Name", "q"])
        self.assertEqual(A2.b()[123], 123)
        A2.b()[123] = 7
        self.assertEqual(A2.b()[123], 7)

    def test_alternative_iter(self):

        class A_iter():
            def __init__(self):
                self._cur = 0

            def __iter__(self):
                return self

            def __next__(self):
                if self._cur >= 10:
                    raise StopIteration
                self._cur += 1
                return self._cur

        class A_reversed():
            def __init__(self):
                self._cur = 11

            def __iter__(self):
                return self

            def __next__(self):
                if self._cur <= 1:
                    raise StopIteration
                self._cur -= 1
                return self._cur

        A = Alternative("A", a={'a': int}, b={'b': str},
                        __iter__=lambda self: A_iter(),
                        __reversed__=lambda self: A_reversed()
                        )
        self.assertEqual([x for x in A.a()], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.assertEqual([x for x in reversed(A.a())], [10, 9, 8, 7, 6, 5, 4, 3, 2, 1])

    def test_alternative_as_iterator(self):

        class B_iter():
            def __init__(self):
                self._cur = 0

            def __iter__(self):
                return self

            def __next__(self):
                if self._cur >= 10:
                    raise StopIteration
                self._cur += 1
                return self._cur

        x = B_iter()
        Iterator = Alternative("B", a={'a': int},
                               __iter__=lambda self: self,
                               __next__=lambda self: x.__next__()
                               )
        A = Alternative("A", a={'a': int},
                        __iter__=lambda self: Iterator.a()
                        )
        # this is a one-time iterator
        self.assertEqual([x for x in A.a()], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.assertEqual([x for x in A.a()], [])

    def test_alternative_with(self):
        depth = 0

        def A_enter(s):
            nonlocal depth
            depth += 1
            return depth

        def A_exit(s, t, v, b):
            nonlocal depth
            depth -= 1
            return True

        A = Alternative("A", a={'a': int}, b={'b': str},
                        __enter__=A_enter,
                        __exit__=A_exit
                        )

        self.assertEqual(depth, 0)
        with A.a():
            self.assertEqual(depth, 1)
            with A.b():
                self.assertEqual(depth, 2)
        self.assertEqual(depth, 0)

    def test_empty_alternatives(self):
        a = Alternative(
            "Alt",
            A={},
            B={}
        )

        self.assertEqual(a.A(), a.A())
        self.assertIsInstance(deserialize(a, serialize(a, a.A())), a.A)
        self.assertEqual(a.A(), deserialize(a, serialize(a, a.A())))

        self.assertEqual(a.B(), a.B())
        self.assertNotEqual(a.A(), a.B())
        self.assertNotEqual(a.B(), a.A())

    def test_extracted_alternatives_have_correct_type(self):
        Alt = Alternative(
            "Alt",
            A={},
            B={}
        )
        tOfAlt = TupleOf(Alt)

        a = Alt.A()
        aTup = tOfAlt((a,))

        self.assertEqual(a, aTup[0])
        self.assertTrue(type(a) is type(aTup[0]))  # noqa

    def test_alternatives(self):
        alt = Alternative(
            "Alt",
            child_ints={'x': int, 'y': int},
            child_strings={'x': str, 'y': str}
        )

        self.assertTrue(issubclass(alt.child_ints, alt))
        self.assertTrue(issubclass(alt.child_strings, alt))

        a = alt.child_ints(x=10, y=20)
        a2 = alt.child_ints(x=10, y=20)

        self.assertEqual(a, a2)

        self.assertTrue(isinstance(a, alt))
        self.assertTrue(isinstance(a, alt.child_ints))

        self.assertEqual(a.x, 10)
        self.assertEqual(a.y, 20)
        self.assertTrue(a.matches.child_ints)
        self.assertFalse(a.matches.child_strings)

        with self.assertRaisesRegex(AttributeError, "immutable"):
            a.x = 20

    def test_alternatives_comparison(self):
        empty = Alternative("X", A={}, B={})

        self.assertEqual(empty.A(), empty.A())
        self.assertEqual(empty.B(), empty.B())
        self.assertNotEqual(empty.A(), empty.B())

        a = Alternative(
            "X",
            A={'a': int},
            B={'b': int},
            C={'c': str},
            D={'d': bytes},
        )

        self.assertEqual(a.A(a=10), a.A(a=10))
        self.assertNotEqual(a.A(a=10), a.A(a=11))

        self.assertNotEqual(a.C(c=""), a.C(c="hi"))
        self.assertFalse(a.C(c="") == a.C(c="hi"))
        self.assertNotEqual(a.D(d=b""), a.D(d=b"hi"))

    def test_alternatives_add_operator(self):
        alt = Alternative(
            "Alt",
            child_ints={'x': int, 'y': int},
            __add__=lambda lhs, rhs: (lhs, rhs)
        )

        a = alt.child_ints(x=0, y=2)

        self.assertEqual(a+a, (a, a))

    @flaky(max_runs=3, min_passes=1)
    def test_alternatives_radd_operator(self):
        alt = Alternative(
            "Alt",
            child_ints={'x': int, 'y': int},
            __radd__=lambda lhs, rhs: "radd"
        )

        a = alt.child_ints(x=0, y=2)

        values = [1, Int16(1), UInt64(1), 1.234, Float32(1.234), True, "abc",
                  ListOf(int)((1, 2)), ConstDict(str, str)({"a": "1"})]
        for v in values:
            self.assertEqual(v + a, "radd")
            with self.assertRaises(Exception):
                print(a + v)

    @flaky(max_runs=3, min_passes=1)
    def test_alternatives_perf(self):
        alt = Alternative(
            "Alt",
            child_ints={'x': int, 'y': int},
            child_strings={'x': str, 'y': str}
        )

        t0 = time.time()

        for i in range(1000000):
            a = alt.child_ints(x=10, y=20)
            a.matches.child_ints
            a.x

        elapsed = time.time() - t0
        print("Took ", elapsed, " to do 1mm")
        self.check_expected_performance(elapsed, expected=2.0)

    def test_object_hashing_and_equality(self):
        for _ in range(100):
            producer = RandomValueProducer()
            producer.addEvenly(20, 2)

            values = producer.all()

            for v1 in values:
                for v2 in values:
                    if hash(v1) != hash(v2) and v1 == v2:
                        print(v1, v2, type(v1), type(v2))

            for v1 in values:
                for v2 in values:
                    if type(v1) == type(v2) and v1 == v2:
                        self.assertEqual(hash(v1), hash(v2), (v1, v2))
                        if type(v1) is type(v2):
                            self.assertEqual(repr(v1), repr(v2), (v1, v2, type(v1), type(v2)))

    def test_bytes_repr(self):
        # macos has some weird behavior where it can't convert the numpy array
        # to bytes because of a unicode error.
        if sys.platform == "darwin":
            return

        for _ in range(100000):
            # always start with a '"' because otherwise python may choose to start the
            # string with either a ' or a " depending on what it sees inside the string, and
            # typed_python always picks " for now
            someBytes = b'"' + numpy.random.uniform(size=2).tobytes()
            self.assertEqual(repr(makeTuple(someBytes)), repr((someBytes,)))

    def test_equality_with_native_python_objects(self):
        tups = [(1, 2, 3), (), ("2",), (b"2",), (1, 2, 3, "b"), (2,), (None,)]

        for tup1 in tups:
            self.assertEqual( makeTuple(*tup1), tup1 )

            for tup2 in tups:
                if tup1 != tup2:
                    self.assertNotEqual( makeTuple(*tup1), tup2 )

        for tup1 in tups:
            self.assertEqual( makeTupleOf(*tup1), tup1 )

            for tup2 in tups:
                if tup1 != tup2:
                    self.assertNotEqual( makeTupleOf(*tup1), tup2 )

    def test_add_tuple_of(self):
        tupleOfInt = TupleOf(int)

        tups = [(), (1, 2), (1,), (1, 2, 3, 4)]

        for tup1 in tups:
            for tup2 in tups:
                self.assertEqual(tupleOfInt(tup1) + tupleOfInt(tup2), tupleOfInt(tup1+tup2))
                self.assertEqual(tupleOfInt(tup1) + tup2, tupleOfInt(tup1+tup2))

    def test_stringification_of_none(self):
        T = TupleOf(OneOf(None, str))

        self.assertEqual(str(T([None, 'hi'])), '(None, "hi")')

    def test_slice_tuple_of(self):
        tupleOfInt = TupleOf(int)

        ints = tuple(range(20))
        aTuple = tupleOfInt(ints)

        for i in range(-21, 21):
            for i2 in range(-21, 21):
                for step in range(-3, 3):
                    if step != 0:
                        self.assertEqual(aTuple[i:i2:step], ints[i:i2:step])

            try:
                ints[i]
                self.assertEqual(aTuple[i], ints[i])
            except IndexError:
                with self.assertRaises(IndexError):
                    aTuple[i]

    def test_dictionary_subtraction_basic(self):
        intDict = ConstDict(int, int)

        self.assertEqual(intDict({1: 2}) - (1,), intDict({}))
        self.assertEqual(intDict({1: 2, 3: 4}) - (1,), intDict({3: 4}))
        self.assertEqual(intDict({1: 2, 3: 4}) - (3,), intDict({1: 2}))

    def test_dictionary_addition_and_subtraction(self):
        someDicts = [{i: choice([1, 2, 3, 4, 5]) for i in range(choice([4, 6, 10, 20]))} for _ in range(20)]
        intDict = ConstDict(int, int)

        for d1 in someDicts:
            for d2 in someDicts:
                addResult = dict(d1)
                addResult.update(d2)

                self.assertEqual(intDict(d1) + intDict(d2), intDict(addResult))

                res = intDict(addResult)

                while len(res):
                    toRemove = []

                    for i in range(choice(list(range(len(res))))+1):
                        key = choice(list(addResult))
                        del addResult[key]
                        toRemove.append(key)

                    res = res - toRemove

                    self.assertEqual(res, intDict(addResult))

    def test_serialization_primitives(self):
        def checkCanSerialize(x):
            self.assertEqual(x, deserialize(type(x), serialize(type(x), x)), x)

        checkCanSerialize(0)
        checkCanSerialize(1)
        checkCanSerialize(2)
        checkCanSerialize(4)
        checkCanSerialize(8)
        checkCanSerialize(16)
        checkCanSerialize(32)
        checkCanSerialize(64)
        checkCanSerialize(128)
        checkCanSerialize(-1)
        checkCanSerialize(290)
        checkCanSerialize(1000)
        checkCanSerialize(99.5)
        checkCanSerialize("hi")
        checkCanSerialize(b"bye")
        checkCanSerialize(None)
        checkCanSerialize(True)
        checkCanSerialize(False)

    def test_serialization_bytecounts(self):
        ints = TupleOf(int)((1, 2, 3, 4))

        def varintBytecount(value):
            """the length (in bytes) of a varint"""
            res = 1
            while value >= 128:
                res += 1
                value /= 128
            return res

        while len(ints) < 1000000:
            ints = ints + ints
            t0 = time.time()

            expectedBytecount = (
                sum(varintBytecount(0) + varintBytecount(i) for i in ints) +
                (varintBytecount(0) * 3) + varintBytecount(len(ints))
            )

            self.assertEqual(len(serialize(TupleOf(int), ints)), expectedBytecount)

            print(time.time() - t0, " for ", len(ints))

    def test_serialization_roundtrip(self):
        for _ in range(100):
            producer = RandomValueProducer()
            producer.addEvenly(30, 3)

            values = producer.all()
            for v in values:
                ser = serialize(type(v), v)

                v2 = deserialize(type(v), ser)

                ser2 = serialize(type(v), v2)

                self.assertTrue(type(v2) is type(v))
                self.assertEqual(ser, ser2)
                self.assertEqual(str(v), str(v2))
                self.assertEqual(v, v2, (v, v2, type(v), type(v2), type(v) is type(v2)))

    def test_create_invalid_tuple(self):
        with self.assertRaises(TypeError):
            Tuple((int, int))

    def test_roundtrip_tuple(self):
        T = Tuple(str, bool, str)
        v = T(('1', False, ''))

        v2 = deserialize(T, serialize(T, v))

        self.assertEqual(v, v2)

    def test_roundtrip_alternative(self):
        A = Alternative("A", a0=dict(x_0=None))
        T = NamedTuple(x0=A, x1=bool)

        v = T(x0=A.a0(), x1=True)

        v2 = deserialize(T, serialize(T, v))

        self.assertEqual(v, v2)

    def test_serialize_doesnt_leak(self):
        T = TupleOf(int)

        def getMem():
            return psutil.Process().memory_info().rss / 1024 ** 2

        m0 = getMem()

        for passIx in range(100):
            for i in range(1000):
                t = T(list(range(i)))
                deserialize(T, serialize(T, t))

            self.assertTrue(getMem() < m0 + 100)

    def test_const_dict_of_tuple(self):
        K = NamedTuple(a=OneOf(float, int), b=OneOf(float, int))
        someKs = [K(a=0, b=0), K(a=1), K(a=10), K(b=10), K()]

        T = ConstDict(K, K)

        indexDict = {}
        x = T()

        numpy.random.seed(42)

        for _ in range(100):
            i1 = numpy.random.choice(len(someKs))
            i2 = numpy.random.choice(len(someKs))
            add = numpy.random.choice([False, True])

            if add:
                indexDict[i1] = i2
                x = x + {someKs[i1]: someKs[i2]}
            else:
                if i1 in indexDict:
                    del indexDict[i1]
                    x = x - (someKs[i1],)

            self.assertEqual(x, T({someKs[i]: someKs[v] for i, v in indexDict.items()}))
            for k in x:
                self.assertTrue(k in x)
                x[k]

    def test_conversion_of_binary_compatible(self):
        class T1(NamedTuple(a=int)):
            pass

        class T2(NamedTuple(a=int)):
            pass

        class T1Comp(NamedTuple(d=ConstDict(str, T1))):
            pass

        class T2Comp(NamedTuple(d=ConstDict(str, T1))):
            pass

        self.assertTrue(_types.isBinaryCompatible(T1Comp, T2Comp))
        self.assertTrue(_types.isBinaryCompatible(T1, T2))

    def test_binary_compatible_nested(self):
        def make():
            class Interior(NamedTuple(a=int)):
                pass

            class Exterior(NamedTuple(a=Interior)):
                pass

            return Exterior

        E1 = make()
        E2 = make()

        self.assertTrue(_types.isBinaryCompatible(E1, E2))

    def test_python_objects_in_tuples(self):
        class NormalPyClass:
            pass

        class NormalPySubclass(NormalPyClass):
            pass

        NT = NamedTuple(x=NormalPyClass, y=NormalPySubclass)

        nt = NT(x=NormalPyClass(), y=NormalPySubclass())
        self.assertIsInstance(nt.x, NormalPyClass)
        self.assertIsInstance(nt.y, NormalPySubclass)

    def test_construct_alternatives_with_positional_arguments(self):
        a = Alternative("A", HasOne={'a': str}, HasTwo={'a': str, 'b': str})

        with self.assertRaises(TypeError):
            a.HasTwo("hi")

        self.assertEqual(a.HasOne("hi"), a.HasOne(a="hi"))

        hasOne = a.HasOne("hi")
        self.assertEqual(a.HasOne(hasOne), hasOne)

        with self.assertRaises(TypeError):
            a.HasOne(a.HasTwo(a='1', b='b'))

    def test_unsafe_pointers_to_list_internals(self):
        x = ListOf(int)()
        x.resize(100)
        for i in range(len(x)):
            x[i] = i

        aPointer = x.pointerUnsafe(0)
        self.assertTrue(str(aPointer).startswith("(int*)0x"))

        self.assertEqual(aPointer.get(), x[0])
        aPointer.set(100)
        self.assertEqual(aPointer.get(), 100)
        self.assertEqual(x[0], 100)

        aPointer = aPointer + 10

        self.assertEqual(aPointer.get(), x[10])
        self.assertEqual(aPointer[10], x[20])
        aPointer.set(20)
        self.assertEqual(aPointer.get(), 20)
        self.assertEqual(x[10], 20)

        # this is OK because ints are POD.
        aPointer.initialize(30)
        self.assertEqual(x[10], 30)

    def test_pointer_to_has_no_len_and_is_not_iterable(self):
        x = ListOf(int)([1, 2])

        with self.assertRaises(TypeError):
            len(x.pointerUnsafe(0))

        with self.assertRaises(Exception):
            for i in x.pointerUnsafe(0):
                break

    def test_unsafe_pointers_to_uninitialized_list_items(self):
        # because this is testing unsafe operations, the test is
        # really just that we don't segfault!
        for _ in range(100):
            x = ListOf(TupleOf(int))()
            x.reserve(10)

            for i in range(x.reserved()):
                x.pointerUnsafe(i).initialize((i,))

            x.setSizeUnsafe(10)

        # now check that if we fail to set the size we'll leak the tuple
        aLeakedTuple = TupleOf(int)((1, 2, 3))
        x = ListOf(TupleOf(int))()
        x.reserve(1)
        x.pointerUnsafe(0).initialize(aLeakedTuple)
        x = None

        self.assertEqual(_types.refcount(aLeakedTuple), 2)

    def test_list_extend(self):
        LI = ListOf(int)
        LF = ListOf(float)

        li = LI([1, 2, 3])
        lf = LF([1.5, 2.5, 3.5])

        li.extend(li)
        self.assertEqual(li, [1, 2, 3, 1, 2, 3])

        lf.extend(lf)
        self.assertEqual(lf, [1.5, 2.5, 3.5, 1.5, 2.5, 3.5])

        lf.extend(li)
        self.assertEqual(lf, [1.5, 2.5, 3.5, 1.5, 2.5, 3.5, 1, 2, 3, 1, 2, 3])

        li = LI()
        li.extend(range(10))

        self.assertEqual(li, list(range(10)))

    def test_list_copy_operation_duplicates_list(self):
        T = ListOf(int)

        x = T([1, 2, 3])
        y = T(x)

        x[0] = 100

        self.assertNotEqual(y[0], 100)

    def test_list_and_tuple_conversion_to_numpy(self):
        for T in [ListOf(bool), TupleOf(bool)]:
            for arr in [
                    numpy.array([]),
                    numpy.array([0, 1, 2, 3, 4, 5]),
                    numpy.array([0, 1, 2, 3, 4, 5], 'int32'),
                    numpy.array([0, 1, 2, 3, 4, 5], 'int16'),
                    numpy.array([0, 1, 2, 3, 4, 5], 'bool')
            ]:
                self.assertEqual(T(arr), T(arr.tolist()))
                self.assertEqual(T(arr).toArray().tolist(), [bool(x) for x in arr.tolist()])

        for T in [ListOf(int), TupleOf(int)]:
            for arr in [
                    numpy.array([]),
                    numpy.array([1, 2, 3, 4, 5]),
                    numpy.array([1, 2, 3, 4, 5], 'int32'),
                    numpy.array([1, 2, 3, 4, 5], 'int16')
            ]:
                self.assertEqual(T(arr), T(arr.tolist()))
                self.assertEqual(T(arr).toArray().tolist(), arr.tolist())

        for T in [ListOf(float), TupleOf(float)]:
            for arr in [
                    numpy.array([]),
                    numpy.array([1, 2, 3, 4, 5], 'float64'),
                    numpy.array([1, 2, 3, 4, 5], 'float32')
            ]:
                self.assertEqual(T(arr), T(arr.tolist()))
                self.assertEqual(T(arr).toArray().tolist(), arr.tolist())

        self.assertEqual(str(ListOf(int)([1, 2, 3, 4]).toArray().dtype), 'int64')
        self.assertEqual(str(ListOf(Int32)([1, 2, 3, 4]).toArray().dtype), 'int32')
        self.assertEqual(str(ListOf(Int16)([1, 2, 3, 4]).toArray().dtype), 'int16')
        self.assertEqual(str(ListOf(Int8)([1, 2, 3, 4]).toArray().dtype), 'int8')

        self.assertEqual(str(ListOf(UInt64)([1, 2, 3, 4]).toArray().dtype), 'uint64')
        self.assertEqual(str(ListOf(UInt32)([1, 2, 3, 4]).toArray().dtype), 'uint32')
        self.assertEqual(str(ListOf(UInt16)([1, 2, 3, 4]).toArray().dtype), 'uint16')
        self.assertEqual(str(ListOf(UInt8)([1, 2, 3, 4]).toArray().dtype), 'uint8')

        self.assertEqual(str(ListOf(float)([1, 2, 3, 4]).toArray().dtype), 'float64')
        self.assertEqual(str(ListOf(Float32)([1, 2, 3, 4]).toArray().dtype), 'float32')

    def test_list_of_equality(self):
        x = ListOf(int)([1, 2, 3, 4])
        y = ListOf(int)([1, 2, 3, 5])

        self.assertEqual(x, x)
        self.assertNotEqual(x, y)

    def test_tuple_r_add(self):
        self.assertEqual(
            (1, 2, 4, 5, 6) + TupleOf(int)([1, 2]),
            (1, 2, 4, 5, 6, 1, 2)
        )

        self.assertEqual(
            [1, 2, 4, 5, 6] + TupleOf(int)([1, 2]),
            (1, 2, 4, 5, 6, 1, 2)
        )

        with self.assertRaises(TypeError):
            [1, 2, "hi", 5, 6] + TupleOf(int)([1, 2])

    def test_tuple_r_cmp(self):
        self.assertEqual(
            (1, 2, 3), TupleOf(int)([1, 2, 3])
        )

    def test_can_convert_numpy_scalars(self):
        self.assertEqual(OneOf(int, float)(numpy.int64(10)), 10)
        self.assertEqual(OneOf(int, float)(numpy.float64(10.5)), 10.5)

    def test_other_bitness_types(self):
        # verify we can cast around non-64-bit values in a way that matches numpy
        typeAndNumpyType = [
            (bool, numpy.bool),
            (Int8, numpy.int8),
            (Int16, numpy.int16),
            (Int32, numpy.int32),
            (UInt8, numpy.uint8),
            (UInt16, numpy.uint16),
            (UInt32, numpy.uint32),
            (UInt64, numpy.uint64),
            (Float32, numpy.float32),
            (float, numpy.float64)
        ]

        for ourType, numpyType in typeAndNumpyType:
            for candValue in [-1, 0, 1, 10, 100, 1000, 100000, 10000000, 10000000000]:
                self.assertEqual(int(ourType(candValue)), int(numpyType(candValue)), (ourType, candValue))
                self.assertEqual(float(ourType(candValue)), float(numpyType(candValue)), (ourType, candValue))

            for ourType2, numpyType2 in typeAndNumpyType:
                zeroOrTwoFloatTypes = sum([1 if 'float' in str(t) else 0 for t in [numpyType, numpyType2]]) in [0, 2]

                if zeroOrTwoFloatTypes:
                    for candValue in [-1, 0, 1, 10, 100, 1000, 100000, 10000000, 10000000000]:
                        self.assertEqual(
                            int(ourType(ourType2(candValue))),
                            int(numpyType(numpyType2(candValue))),
                            (ourType, ourType2, candValue)
                        )
                        self.assertEqual(
                            float(ourType(ourType2(candValue))),
                            float(numpyType(numpyType2(candValue))),
                            (ourType, ourType2, candValue)
                        )
                else:
                    # we convert from float to int as c++, which is different than numpy, which clips
                    # floats in a bizarre way. e.g.
                    #  numpy.int16(numpy.int64(numpy.float64(10000000000)))
                    # is not
                    #  numpy.int16(numpy.float64(10000000000))
                    pass

    def test_other_bitness_types_operators(self):
        def add(x, y):
            return x + y

        def div(x, y):
            return x / y

        def mod(x, y):
            return x % y

        def mul(x, y):
            return x * y

        def sub(x, y):
            return x - y

        def pow(x, y):
            return x ** y

        def bitand(x, y):
            return x & y

        def bitor(x, y):
            return x | y

        def bitxor(x, y):
            return x ^ y

        def less(x, y):
            return x < y

        def greater(x, y):
            return x > y

        def lesseq(x, y):
            return x <= y

        def greatereq(x, y):
            return x >= y

        def eq(x, y):
            return x == y

        def neq(x, y):
            return x != y

        otherTypes = [bool, Int8, Int16, Int32, UInt8, UInt16, UInt32, UInt64, Float32, float]

        for T1 in otherTypes:
            for T2 in otherTypes:
                for op in [less, greater, lesseq, greatereq, eq, neq]:
                    for i1 in [-1, 0, 1, 2]:
                        for i2 in [-1, 0, 1, 2]:
                            res = op(T1(i1), T2(i2))
                            promotedType = computeArithmeticBinaryResultType(T1, T2)

                            if promotedType is int:
                                # in the compiler (and in our internals, ints are 64 bits)
                                # but in python they are of arbitrary bitness. For this test
                                # to make sense, we need to compare the results to the promoted
                                # type we would use internally, which is an int64.
                                @Entrypoint
                                def promotedType(i: int):
                                    return i

                            proI1 = promotedType(T1(i1))
                            proI2 = promotedType(T2(i2))
                            self.assertEqual(res, op(proI1, proI2), (res, op, T1, T2, promotedType, proI1, proI2))

                for op in [add, mul, div, sub, mod]:
                    for i1 in [-1, 0, 1, 2, 10]:
                        for i2 in [-1, 0, 1, 2, 10]:
                            validOp = True
                            if op is div and (T1 is bool or T2 is bool):
                                validOp = False
                            if op in (div, mod) and i2 == 0:
                                validOp = False
                            elif op is pow and i2 < 0 and i2 < 1:
                                validOp = False

                            if validOp:
                                res = op(T1(i1), T2(i2))
                                promotedType = computeArithmeticBinaryResultType(T1, T2)
                                if op in [div]:
                                    promotedType = computeArithmeticBinaryResultType(promotedType, Float32)

                                if promotedType is int:
                                    # in the compiler (and in our internals, ints are 64 bits)
                                    # but in python they are of arbitrary bitness. For this test
                                    # to make sense, we need to compare the results to the promoted
                                    # type we would use internally, which is an int64.
                                    @Entrypoint
                                    def promotedType(i: int):
                                        return i

                                proI1 = promotedType(T1(i1))
                                proI2 = promotedType(T2(i2))
                                self.assertEqual(type(res), type(op(proI1, proI2)))
                                self.assertEqual(res, op(proI1, proI2), (op.__name__, T1, T2, i1, i2, proI1, proI2))

                if not floatness(T1) and not floatness(T2):
                    for op in [bitand, bitor, bitxor]:
                        res = op(T1(10), T2(10))
                        resType = type(res)
                        resType = {bool: bool, int: int, float: float}.get(resType, resType)

                        if T1 is bool and T2 is bool:
                            self.assertEqual(resType, bool if op in (bitor, bitand, bitxor) else int if op is not div else float)
                        else:
                            self.assertEqual(bitness(resType), max(bitness(T1), bitness(T2)))

                            if op is not div:
                                self.assertEqual(isSignedInt(resType), isSignedInt(T1) or isSignedInt(T2))

                            if bitness(T1) > 1 and bitness(T2) > 1:
                                self.assertEqual(res, op(10, 10))

    def test_comparing_arbitrary_objects(self):
        x = TupleOf(object)(["a"])
        y = TupleOf(object)([1])

        with self.assertRaises(TypeError):
            x < y

        self.assertEqual(x, x)
        self.assertEqual(y, y)
        self.assertNotEqual(x, y)

    def test_list_of_indexing_with_numpy_ints(self):
        x = ListOf(ListOf(int))([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(x[numpy.int64(0)][numpy.int64(0)], 1)

    def test_error_message_on_bad_dispatch(self):
        @Function
        def f(x: int):
            return x

        with self.assertRaisesRegex(TypeError, "str"):
            f("hi")

        with self.assertRaisesRegex(TypeError, "argname="):
            f(argname=1)

    def test_const_dict_comparison_more(self):
        N = NamedTuple(x=OneOf(None, int), y=OneOf(None, int))
        D = ConstDict(str, N)

        n1 = N(x=1, y=2)
        n2 = N(x=1, y=3)

        self.assertEqual(D({'a': n1}), D({'a': n1}))
        self.assertNotEqual(D({'a': n1}), D({'a': n2}))

    def test_mutable_dict(self):
        T = Dict(int, int)

        d = T()

        self.assertEqual(len(d), 0)

        with self.assertRaises(KeyError):
            d[0]

        d[0] = 10

        self.assertEqual(len(d), 1)
        self.assertEqual(d[0], 10)

        d[0] = 20

        self.assertEqual(len(d), 1)
        self.assertEqual(d[0], 20)

        for i in range(2000):
            d[i] = i

            if i % 100 == 0 and i:
                for x in range(len(d)):
                    self.assertEqual(d[x], x)

        for i in range(2000):
            if i % 100 == 0 and i:
                for x in range(i):
                    assert x not in d

                for x in range(i, 2000):
                    self.assertEqual(d[x], x)

            del d[i]

        # verify that adding and removing elements doesn't leak memory
        usage = currentMemUsageMb()
        for i in range(1000000):
            d[0] = i
            del d[0]
        self.assertLess(currentMemUsageMb(), usage+1)

    def test_mutable_dict_fuzz(self):
        native_d = Dict(int, int)()
        py_d = {}

        for dictSize in [10, 100, 1000, 10000]:
            for i in range(100000):
                z = numpy.random.choice(dictSize)

                self.assertEqual(z in py_d, z in native_d)

                if i % 3 == 0 or (i % 1000) > 900:
                    if z in py_d:
                        del py_d[z]
                        del native_d[z]
                else:
                    py_d[z] = i
                    native_d[z] = i

            for i in range(dictSize):
                self.assertEqual(z in py_d, z in native_d)

    def test_mutable_dict_refcounts(self):
        native_d = Dict(str, ListOf(int))()
        i = ListOf(int)()

        native_d["a"] = i

        self.assertEqual(_types.refcount(i), 2)

        native_d["b"] = i

        self.assertEqual(_types.refcount(i), 3)

        del native_d["a"]

        self.assertEqual(_types.refcount(i), 2)

        native_d["b"] = ListOf(int)()

        self.assertEqual(_types.refcount(i), 1)

        native_d["b"] = i

        self.assertEqual(_types.refcount(i), 2)

        native_d = None

        self.assertEqual(_types.refcount(i), 1)

    def test_mutable_dict_create_many(self):
        for ct in range(100):
            d = Dict(int, int)()
            for i in range(ct):
                d[i] = i + 1

    def test_mutable_dict_methods(self):
        d = Dict(int, int)({i: i+1 for i in range(10)})

        self.assertEqual(list(d.keys()), list(range(10)))
        self.assertEqual(list(d.values()), list(range(1, 11)))
        self.assertEqual(list(d.items()), [(i, i+1) for i in range(10)])

        for i in range(10):
            self.assertEqual(d.get(i), i+1)
            self.assertEqual(d.get(i, None), i+1)

        self.assertEqual(d.get(1000), None)
        self.assertEqual(d.get(1000, 123), 123)

        with self.assertRaises(TypeError):
            self.assertEqual(d.get("1000"), None)

    def test_mutable_dict_setdefault_bad_arguments(self):
        d = Dict(int, str)()

        with self.assertRaises(TypeError):
            d.setdefault(1, 2, 3)

    def test_mutable_dict_setdefault(self):
        d = Dict(int, str)()
        d[1] = "a"

        # check if this call doesn't change the dict
        # and returns already existing value
        v1 = d.setdefault(1, "b")
        self.assertEqual(v1, "a")
        self.assertEqual(d[1], "a")

        # check if this call sets the d[2]="b"
        # # and returns "b"
        v2 = d.setdefault(2, "b")
        self.assertEqual(v2, "b")
        self.assertEqual(d[2], "b")

        with self.assertRaisesRegex(
            TypeError,
            "Cannot construct a new str from an instance of NoneType"
        ):
            d.setdefault(3, None)

        self.assertEqual(d.setdefault(3), "")

    def test_mutable_dict_pop(self):
        d = Dict(int, str)()
        d[1] = 'a'

        self.assertEqual(d.pop(1), 'a')
        self.assertNotIn(1, d)

        with self.assertRaisesRegex(KeyError, "10"):
            d.pop(10)

        with self.assertRaisesRegex(
            TypeError,
            "Cannot upcast an object of type str to an instance of int"
        ):
            d.pop("hihi")

    def test_mutable_dict_pop_with_conversion(self):
        d = Dict(Tuple(int, str), Tuple(int, str))()
        d[(1, "hi")] = (1, "bye")
        d[(2, "hi")] = (2, "bye")

        self.assertEqual(d.pop((1, "hi")), (1, "bye"))
        self.assertEqual(d.pop(d.KeyType((2, "hi"))), (2, "bye"))
        self.assertTrue(len(d) == 0)

    def test_mutable_dict_setdefault_refcount(self):
        d = Dict(int, ListOf(int))()
        aList = ListOf(int)([1, 2, 3])

        self.assertEqual(_types.refcount(aList), 1)
        d[1] = aList
        self.assertEqual(_types.refcount(aList), 2)
        d.setdefault(2, aList)
        self.assertEqual(_types.refcount(aList), 3)
        a = d.setdefault(2, aList)
        self.assertEqual(_types.refcount(aList), 4)
        self.assertEqual(a, aList)
        a = None
        self.assertEqual(_types.refcount(aList), 3)
        d.setdefault(3, ListOf(int)([1]))
        self.assertEqual(_types.refcount(aList), 3)
        d.setdefault(3, aList)
        self.assertEqual(_types.refcount(aList), 3)

    def test_mutable_dict_iteration_order(self):
        d = Dict(int, int)()

        d[10] = 10
        d[1] = 1
        d[2] = 2

        self.assertEqual(list(d), [10, 1, 2])
        del d[1]
        self.assertEqual(list(d), [10, 2])

    def test_simplicity(self):
        isSimple = _types.isSimple

        self.assertTrue(isSimple(int))
        self.assertTrue(isSimple(Int32()))
        self.assertTrue(isSimple(Int16()))
        self.assertTrue(isSimple(Int8()))
        self.assertTrue(isSimple(UInt64()))
        self.assertTrue(isSimple(UInt32()))
        self.assertTrue(isSimple(UInt16()))
        self.assertTrue(isSimple(UInt8()))
        self.assertTrue(isSimple(str))
        self.assertTrue(isSimple(bytes))
        self.assertTrue(isSimple(bool))
        self.assertTrue(isSimple(float))

        class C(Class):
            pass

        self.assertFalse(isSimple(C))

        self.assertTrue(isSimple(ListOf(int)))
        self.assertFalse(isSimple(ListOf(C)))

        self.assertTrue(isSimple(TupleOf(int)))
        self.assertFalse(isSimple(TupleOf(C)))

        self.assertTrue(isSimple(ConstDict(int, int)))
        self.assertFalse(isSimple(ConstDict(C, int)))
        self.assertFalse(isSimple(ConstDict(int, C)))

        self.assertTrue(isSimple(Dict(int, int)))
        self.assertFalse(isSimple(Dict(C, int)))
        self.assertFalse(isSimple(Dict(int, C)))

        self.assertTrue(isSimple(Set(int)))
        self.assertFalse(isSimple(Set(C)))

        self.assertFalse(isSimple(Alternative("Alternative")))

        self.assertTrue(isSimple(NamedTuple(x=int)))
        self.assertFalse(isSimple(NamedTuple(x=C)))

        X = Forward("X")
        X = X.define(Alternative("X", X={'x': X}, Y={'i': int}))
        self.assertFalse(isSimple(X))
        self.assertFalse(isSimple(NamedTuple(x=X)))

        self.assertFalse(isSimple(OneOf(int, X)))
        self.assertTrue(isSimple(OneOf(int, float)))

    def test_oneof_picks_best_choice(self):
        T = OneOf(float, int, bool)

        self.assertIsInstance(T(1.5), float)
        self.assertIsInstance(T(1), int)
        self.assertIsInstance(T(True), bool)

    def test_dict_equality(self):
        for d in [{1: 2}, {1: 2, 3: 4}]:
            self.assertEqual(Dict(int, int)(d), d)

        self.assertNotEqual(Dict(int, int)({1: 2}), {'1': 2})
        self.assertNotEqual(Dict(int, int)({1: 2}), {1: '2'})
        self.assertNotEqual(Dict(int, int)({1: 2}), {2: 3})

        self.assertNotEqual(Dict(int, int)({1: 2}), {1: 2.5})

        T = Dict(OneOf(int, float), OneOf(int, float))
        self.assertEqual(T({1: 2.5}), {1: 2.5})

    def test_dict_equality_with_python_and_object(self):
        self.assertTrue(Dict(int, object)({1: 2}) == {1: 2})
        self.assertTrue(Dict(int, object)({1: (7, 8, 9)}) == {1: (7, 8, 9)})
        self.assertTrue(Dict(int, object)({1: 'two'}) == {1: 'two'})

    def test_const_dict_with_noncomparable_things(self):
        DictType = ConstDict(OneOf(int, str), int)

        aDict = DictType({1: 100, 'hi': 200, 'bye': 300})

        self.assertEqual(aDict[1], 100)
        self.assertEqual(aDict['hi'], 200)

    def test_const_dict_with_noncomparable_things_as_object(self):
        DictType = ConstDict(object, int)

        aDict = DictType({1: 100, 'hi': 200, 'bye': 300})

        self.assertEqual(aDict[1], 100)
        self.assertEqual(aDict['hi'], 200)

    def test_oneof_conversion(self):
        BrokenOutBool = OneOf(False, True, int)

        self.assertIs(type(BrokenOutBool(0)), int)
        self.assertIs(type(BrokenOutBool(False)), bool)

        BrokenOutBoolReordered = OneOf(int, False, True)

        self.assertIs(type(BrokenOutBoolReordered(0)), int)
        self.assertIs(type(BrokenOutBoolReordered(False)), bool)

    def test_set_constructor_identity(self):
        s = Set(int)
        self.assertEqual(s.__qualname__, "Set(int)")
        s = Set(float)
        self.assertEqual(s.__qualname__, "Set(float)")
        s = Set(str)
        self.assertEqual(s.__qualname__, "Set(str)")

        s1 = Set(int)([1])
        s2 = Set(int)([1])
        self.assertNotEqual(id(s2), id(s1))

    def test_set_update(self):
        s1 = Set(int)([1, 2, 3])

        s1.update([4, 5, 6])

        self.assertEqual(s1, Set(int)(range(1, 7)))

        s1.update(Set(int)([7, 8, 9]))

        self.assertEqual(s1, Set(int)(range(1, 10)))

    def test_set_len(self):
        s1 = Set(int)([1, 2, 3])
        s2 = Set(int)([1, 2, 3])
        self.assertEqual(len(s1), len(s2))

    def test_set_discard(self):
        s = Set(int)([1, 2, 3])
        s.discard(2)
        self.assertNotIn(2, s)
        self.assertRaises(TypeError, s.discard, [])

        s = Set(str)()
        s.add("hello")
        s.discard("hello")
        self.assertEqual(len(s), 0)

        # discard on empty or when key not found should not throw
        s = Set(int)()
        s.discard(1)

    def test_set_clear(self):
        s = Set(int)([1, 2, 3])
        s.clear()
        self.assertEqual(set(s), set())
        self.assertEqual(len(s), 0)

    def test_set_contains(self):
        letters = ['a', 'b', 'c']
        s1 = Set(str)()
        s2 = Set(str)()
        for c in letters:
            s1.add(c)
            s2.add(c)
        for c in letters:
            self.assertEqual(c in s1, c in s2)
        self.assertRaises(TypeError, s1.__contains__, [[]])

    def test_set_remove(self):
        s = Set(str)()
        s.add("a")
        s.add("b")
        s.remove("a")
        self.assertNotIn("a", s)
        self.assertRaises(KeyError, s.remove, "Q")
        self.assertRaises(TypeError, s.remove, [])

    def test_set_add(self):
        s = Set(int)()
        self.assertEqual(len(s), 0)
        s.add(1)
        self.assertIn(1, s)
        self.assertEqual(len(s), 1)
        s.add(2)
        self.assertEqual(len(s), 2)
        s.add(2)
        self.assertEqual(len(s), 2)
        self.assertRaises(TypeError, s.add, [])
        self.assertRaises(TypeError, s.add, 1.0)
        self.assertRaises(TypeError, s.add, "hello")

        for i in ([1, 2, 3], (1, 2, 3), {1, 2, 3}):
            s = Set(int)(i)
            self.assertEqual(len(s), 3)

    def test_set_pop(self):
        s = Set(int)([1, 2, 3])

        self.assertEqual(sorted([s.pop() for i in range(len(s))]), [1, 2, 3])

        with self.assertRaises(KeyError):
            s.pop()

    def test_set_refcounts(self):
        native_d = Dict(str, Set(int))()
        i = Set(int)()
        native_d["a"] = i
        self.assertEqual(_types.refcount(i), 2)
        native_d["b"] = i
        self.assertEqual(_types.refcount(i), 3)
        del native_d["a"]
        self.assertEqual(_types.refcount(i), 2)
        native_d = None
        self.assertEqual(_types.refcount(i), 1)

        d = Dict(str, Set(TupleOf(int)))()
        s = Set(TupleOf(int))()
        i = TupleOf(int)((1,))
        self.assertEqual(_types.refcount(i), 1)
        s.add(i)
        self.assertEqual(_types.refcount(i), 2)
        s.add(i)
        self.assertEqual(_types.refcount(i), 2)
        s.discard(i)
        self.assertEqual(_types.refcount(i), 1)
        d['a'] = d['b'] = Set(TupleOf(int))((i,))
        self.assertEqual(_types.refcount(i), 2)
        d = None
        self.assertEqual(_types.refcount(i), 1)
        s.add(i)
        self.assertEqual(_types.refcount(i), 2)
        d = Dict(str, Set(TupleOf(int)))()
        d['a'] = d['b'] = Set(TupleOf(int))((i,))
        self.assertEqual(_types.refcount(i), 3)
        s.clear()
        self.assertEqual(_types.refcount(i), 2)
        del d['a']
        self.assertEqual(_types.refcount(i), 2)
        del d['b']
        self.assertEqual(_types.refcount(i), 1)
        d = None

        # test several tuple adds into same set
        s.clear()
        self.assertEqual(len(s), 0)
        self.assertEqual(_types.refcount(i), 1)

        i = TupleOf(int)((1,))
        i2 = TupleOf(int)((2,))
        i3 = TupleOf(int)((3,))
        s.add(i)
        s.add(i2)
        s.add(i3)
        self.assertEqual(len(s), 3)
        self.assertEqual(_types.refcount(i), 2)
        self.assertEqual(_types.refcount(i2), 2)
        self.assertEqual(_types.refcount(i3), 2)
        s.remove(i2)
        self.assertEqual(_types.refcount(i), 2)
        self.assertEqual(_types.refcount(i2), 1)
        self.assertEqual(_types.refcount(i3), 2)
        s.remove(i3)
        self.assertEqual(_types.refcount(i), 2)
        self.assertEqual(_types.refcount(i2), 1)
        self.assertEqual(_types.refcount(i3), 1)
        s = None
        self.assertEqual(_types.refcount(i), 1)
        self.assertEqual(_types.refcount(i2), 1)
        self.assertEqual(_types.refcount(i3), 1)

        # test from constructor
        s = Set(TupleOf(int))((i, i2, i3))
        self.assertEqual(len(s), 3)
        self.assertEqual(_types.refcount(i), 2)
        self.assertEqual(_types.refcount(i2), 2)
        self.assertEqual(_types.refcount(i3), 2)
        s.clear()
        self.assertEqual(_types.refcount(i), 1)
        self.assertEqual(_types.refcount(i2), 1)
        self.assertEqual(_types.refcount(i3), 1)

    def test_set_equality(self):
        s = Set(str)()
        s.add('hello')
        other_s = Set(str)()
        other_s.add('world')
        another_s = Set(str)()
        another_s.add('hello')
        self.assertEqual(set(s), set(['hello']))
        self.assertEqual(s == 'hello', False)
        self.assertNotEqual(set(s), set(other_s))
        self.assertEqual(s != 'hello', True)
        self.assertEqual(s == other_s, False)
        self.assertEqual(s != other_s, True)
        self.assertEqual(s == another_s, True)

    def test_set_self_equality(self):
        s = Set(int)()
        self.assertEqual(s, s)

    def test_set_repr(self):
        repr_s = '{1, 2, 3}'
        s = Set(int)([1, 2, 3])
        self.assertEqual(repr(s), repr_s)

    def test_set_literal(self):
        s = Set(int)([1, 2, 3])
        t = {1, 2, 3}
        self.assertEqual(t, set(s))

        s = Set(str)(["a", "b", "c"])
        t = {"a", "b", "c"}
        self.assertEqual(t, set(s))

    def test_set_iterating(self):
        s = Set(int)()
        it = iter(s)
        self.assertRaises(StopIteration, next, it)

        s = Set(str)(["a", "b", "c"])
        it = iter(s)
        count = 0
        while True:
            try:
                next(it)
                count += 1
            except StopIteration:
                break
        self.assertEqual(count, len(s))

    def test_set_assign_from_existing_dict_key_nothrow(self):
        d = Dict(str, Set(int))()
        i = Set(int)()
        d['a'] = i
        d['a'] = i
        d['a'] = Set(int)()

    def test_set_uniquification(self):
        word = 'simsalabim'
        s = Set(str)()
        for i in word:
            s.add(i)
        ss = sorted(s)
        ds = sorted(dict.fromkeys(word))
        self.assertEqual(ss, ds)

    def test_set_copy(self):
        s = Set(int)([1])
        dup = s.copy()
        self.assertEqual(s, dup)
        self.assertNotEqual(id(s), id(dup))
        self.assertEqual(_types.refcount(s), 1)
        self.assertEqual(_types.refcount(dup), 1)
        dup2 = s.copy()
        self.assertEqual(_types.refcount(s), 1)
        s.add(5)
        self.assertEqual(len(s), 2)
        self.assertEqual(len(dup), 1)
        self.assertEqual(len(dup2), 1)
        self.assertNotIn(5, dup)
        self.assertNotIn(5, dup2)
        self.assertRaises(TypeError, s.copy, s)
        self.assertRaises(TypeError, s.copy, [])

    def test_set_construct_from_str(self):
        word = 'symbolic'
        s = Set(str)(word)
        self.assertEqual(len(s), 8)
        self.assertEqual(s == word, False)
        self.assertEqual(s != word, True)
        self.assertRaises(TypeError, s.add, [])
        self.assertRaises(TypeError, s.add, 1.0)
        for c in word:
            self.assertIn(c, s)

    def test_set_ops_throws_diff_type(self):
        s = Set(int)([1])
        self.assertRaises(TypeError, s.union, 1.0)
        self.assertRaises(TypeError, s.union, 'hello')
        self.assertRaises(TypeError, s.union, ListOf(str)(['hello']))
        self.assertRaises(TypeError, s.union, [[]])
        s = Set(str)('hello')
        self.assertRaises(TypeError, s.union, 1)
        self.assertRaises(TypeError, s.union, ListOf(int)([1]))
        self.assertRaises(TypeError, s.union, [[]])

        self.assertRaises(TypeError, s.difference, [[]])

    def test_set_union_refcounts(self):
        s = Set(int)([1])
        s2 = Set(int)([2])
        k = s.union(s2)
        self.assertEqual(_types.refcount(s), 1)
        self.assertEqual(_types.refcount(k), 1)
        self.assertNotIn(2, s)
        self.assertNotIn(1, s2)
        self.assertIn(1, k)
        self.assertIn(2, k)
        self.assertNotEqual(id(s), id(s2), id(k))

    def test_set_union(self):
        word = 'symbolic'
        word2 = 'word'
        s = Set(str)(word)
        u = s.union(Set(str)(word2))
        self.assertEqual(s, Set(str)(word))
        self.assertEqual(type(u), type(s))

        def _check(u, s, chars):
            self.assertEqual(len(u), len(s))
            self.assertEqual(len(u), len(chars))
            for c in chars:
                self.assertIn(c, u)
                self.assertIn(c, s)

        chars = 'abcd'
        u = Set(str)('abcba').union(Set(str)('cdc'))
        s = set(chars)
        _check(u, s, chars)
        chars = 'abcefg'
        u = Set(str)('abcba').union(Set(str)('efgfe'))
        s = set('abcefg')
        _check(u, s, chars)
        chars = 'abc'
        u = Set(str)('abcba').union(Set(str)('ccb'))
        s = set(chars)
        _check(u, s, chars)
        chars = 'abcef'
        u = Set(str)('abcba').union(Set(str)('ef'))
        s = set(chars)
        _check(u, s, chars)
        chars = 'abcefg'
        u = Set(str)('abcba').union(Set(str)('ef'), Set(str)('fg'))
        s = set(chars)
        _check(u, s, chars)

        s = Set(int)()
        self.assertEqual(s.union(Set(int)([1]), s, Set(int)([2])), Set(int)([1, 2]))

    def test_set_ops_with_other_containers(self):
        S = Set(str)
        # operators require type matching; but methods accept iterables
        # methods accept any number of arguments, except symmetric_difference, which requires exactly one argument

        self.assertEqual(S('abcba').union(), S('abc'))
        self.assertEqual(S('abc').union(S('bcd')), S('abc') | S('bcd'))
        for C in set, list, tuple, ListOf(str), TupleOf(str), S:
            self.assertEqual(S('abcba').union(C('cdc')), S('abcd'))
            self.assertEqual(S('abcba').union(C('ef'), C('fg')), S('abcefg'))
            with self.assertRaises(TypeError):
                S('abc').union(C([1, 2, 3]))
            if C is not S:
                with self.assertRaises(TypeError):
                    S('abc') | C('bcd')

        # intersection
        self.assertEqual(S('abcba').intersection(), S('abc'))
        self.assertEqual(S('abc').intersection(S('bcd')), S('abc') & S('bcd'))
        for C in set, list, tuple, ListOf(str), TupleOf(str), S:
            self.assertEqual(S('abcba').intersection(C('cdc')), S('cc'))
            self.assertEqual(S('abcba').intersection(C('efgfe')), S(''))
            self.assertEqual(S('abcba').intersection(C('ccb')), S('bc'))
            self.assertEqual(S('abcba').intersection(C('ef')), S(''))
            self.assertEqual(S('abcba').intersection(C('cbcf'), C('bag')), S('b'))
            with self.assertRaises(TypeError):
                S('abc').intersection(C([1, 2, 3]))
            if C is not S:
                with self.assertRaises(TypeError):
                    S('abc') & C('bcd')

        # difference
        self.assertEqual(S('abcba').difference(), S('abc'))
        self.assertEqual(S('abc').difference(S('bcd')), S('abc') - S('bcd'))
        for C in set, list, tuple, ListOf(str), TupleOf(str), S:
            self.assertEqual(S('abcba').difference(C('cdc')), S('ab'))
            self.assertEqual(S('abcba').difference(C('efgfe')), S('abc'))
            self.assertEqual(S('abcba').difference(C('ccb')), S('a'))
            self.assertEqual(S('abcba').difference(C('ef')), S('abc'))
            self.assertEqual(S('abcba').difference(C('ef'), C('ag'), C('a')), S('bc'))
            with self.assertRaises(TypeError):
                S('abc').difference(C([1, 2, 3]))
            if C is not S:
                with self.assertRaises(TypeError):
                    S('abc') - C('bcd')

        # symmetric difference
        with self.assertRaises(TypeError):
            S('abcba').symmetric_difference()
        self.assertEqual(S('abc').symmetric_difference(S('bcd')), S('abc') ^ S('bcd'))
        for C in set, list, tuple, ListOf(str), TupleOf(str), S:
            self.assertEqual(S('abcba').symmetric_difference(C('cdc')), S('abd'), C)
            self.assertEqual(S('abcba').symmetric_difference(C('efgfe')), S('abcefg'), C)
            self.assertEqual(S('abcba').symmetric_difference(C('ccb')), S('a'), C)
            self.assertEqual(S('abcba').symmetric_difference(C('ef')), S('abcef'), C)
            with self.assertRaises(TypeError):
                S('abc').symmetric_difference(C([1, 2, 3]))
            if C is not S:
                with self.assertRaises(TypeError):
                    S('abc') ^ C('bcd')
        with self.assertRaises(TypeError):
            S('abcba').symmetric_difference(set("cd"), set("ef"))

        # subset
        for C in set, list, tuple, ListOf(str), TupleOf(str), S:
            self.assertFalse(S('abcba').issubset(C('cdc')), C)
            self.assertFalse(S('abcba').issubset(C('efgfe')), C)
            self.assertTrue(S('abcba').issubset(C('abcdecad')), C)
            self.assertFalse(S('abcba').issubset(C('ccb')), C)

        self.assertFalse(S('abcba') <= S('cdc'))
        self.assertFalse(S('abcba') <= S('efgfe'))
        self.assertTrue(S('abcba') <= S('abcdecad'))
        self.assertFalse(S('abcba') <= S('ccb'))
        self.assertTrue(S('abcba') <= S('abcc'))

        self.assertFalse(S('abcba') < S('cdc'))
        self.assertFalse(S('abcba') < S('efgfe'))
        self.assertTrue(S('abcba') < S('abcdecad'))
        self.assertFalse(S('abcba') < S('ccb'))
        self.assertFalse(S('abcba') < S('abcc'))

        # superset
        for C in set, list, tuple, ListOf(str), TupleOf(str), S:
            self.assertFalse(S('abcba').issuperset(C('cdc')), C)
            self.assertFalse(S('abcba').issuperset(C('efgfe')), C)
            self.assertFalse(S('abcba').issuperset(C('abcdecad')), C)
            self.assertTrue(S('abcba').issuperset(C('ccb')), C)

        self.assertFalse(S('abcba') >= S('cdc'))
        self.assertFalse(S('abcba') >= S('efgfe'))
        self.assertFalse(S('abcba') >= S('abcdecad'))
        self.assertTrue(S('abcba') >= S('ccb'))
        self.assertTrue(S('abcba') >= S('abcc'))

        self.assertFalse(S('abcba') > S('cdc'))
        self.assertFalse(S('abcba') > S('efgfe'))
        self.assertFalse(S('abcba') > S('abcdecad'))
        self.assertTrue(S('abcba') > S('ccb'))
        self.assertFalse(S('abcba') > S('abcc'))

        # disjoint
        for C in set, list, tuple, ListOf(str), TupleOf(str), S:
            self.assertFalse(S('abcba').isdisjoint(C('cdc')), C)
            self.assertTrue(S('abcba').isdisjoint(C('efgfe')), C)
            self.assertFalse(S('abcba').isdisjoint(C('abcdecad')), C)
            self.assertFalse(S('abcba').isdisjoint(C('ccb')), C)

    def test_set_intersection(self):
        word1 = 'symbolic'
        word2 = 'words'
        alphabet = 'abcdefghijklmnopqrstuvwxyz'

        s1 = Set(str)(word1)
        s2 = Set(str)(word2)
        i = s1.intersection(s2)
        self.assertEqual(_types.refcount(s1), 1)
        self.assertEqual(_types.refcount(s2), 1)
        for c in alphabet:
            self.assertEqual(c in i, c in word1 and c in word2)

        self.assertEqual(s1, Set(str)(word1))
        self.assertEqual(type(i), Set(str))

        z = s1.intersection()
        self.assertNotEqual(id(s1), id(z))
        self.assertEqual(z, s1)
        self.assertEqual(_types.refcount(s1), 1)

    def test_set_difference(self):
        word1 = 'symbolic'
        word2 = 'symbolism'
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
        s1 = Set(str)(word1)
        s2 = Set(str)(word2)
        i = s1.difference(s2)
        i_operator = s1 - s2
        self.assertEqual(i, i_operator)
        for c in alphabet:
            self.assertEqual(c in i, c in word1 and c not in word2)

        self.assertEqual(s1, Set(str)(word1))
        self.assertEqual(type(i), Set(str))

    def test_set_symmetric_difference(self):
        word1 = 'symbolic'
        word2 = 'symbolism'
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
        s1 = Set(str)(word1)
        s2 = Set(str)(word2)
        i = s1.symmetric_difference(s2)
        i_operator = s1 ^ s2
        self.assertEqual(i, i_operator)
        for c in alphabet:
            self.assertEqual(c in i, (c in word1 and c not in word2) or (c not in word1 and c in word2))

        self.assertEqual(s1, Set(str)(word1))
        self.assertEqual(type(i), Set(str))

    def test_set_listof_tupleof_constructors(self):
        s1 = Set(int)(ListOf(int)([1, 1]))
        self.assertEqual(len(s1), 1)

        s2 = Set(int)(TupleOf(int)((1, 1)))
        self.assertEqual(len(s2), 1)

    def test_list_of_tuples_transpose(self):
        listOfTuples = ListOf(NamedTuple(x=int, y=str, z=bool))()
        listOfTuples.append(dict(x=1, y="hi", z=False))
        listOfTuples.append(dict(x=2, y="hihi", z=True))

        tupleOfLists = listOfTuples.transpose()

        self.assertEqual(tupleOfLists.x, [1, 2])
        self.assertEqual(tupleOfLists.y, ['hi', 'hihi'])
        self.assertEqual(tupleOfLists.z, [False, True])

    def test_const_dict_equality_with_python(self):
        CD = ConstDict(OneOf(int, str), OneOf(int, str))

        someDicts = [{}, {1: 2}, {1: 2, 3: 4}, {"a": 10, "b": "c"}]

        for d1 in someDicts:
            for d2 in someDicts:
                self.assertEqual(CD(d1) == CD(d2), d1 == d2)
                self.assertEqual(CD(d1) == d2, d1 == d2)
                self.assertEqual(d1 == CD(d2), d1 == d2)

    def test_alternative_reverse_operators(self):
        A = Alternative("A", a={'a': int}, b={'b': str},
                        __radd__=lambda lhs, rhs: "radd",
                        __rsub__=lambda lhs, rhs: "rsub",
                        __rmul__=lambda lhs, rhs: "rmul",
                        __rmatmul__=lambda lhs, rhs: "rmatmul",
                        __rtruediv__=lambda lhs, rhs: "rtruediv",
                        __rfloordiv__=lambda lhs, rhs: "rfloordiv",
                        __rmod__=lambda lhs, rhs: "rmod",
                        __rpow__=lambda lhs, rhs: "rpow",
                        __rlshift__=lambda lhs, rhs: "rlshift",
                        __rrshift__=lambda lhs, rhs: "rrshift",
                        __rand__=lambda lhs, rhs: "rand",
                        __rxor__=lambda lhs, rhs: "rxor",
                        __ror__=lambda lhs, rhs: "ror"
                        )

        values = [1, Int16(1), UInt64(1), 1.234, Float32(1.234), True, "abc",
                  ListOf(int)((1, 2)), ConstDict(str, str)({"a": "1"})]
        for v in values:
            self.assertEqual(v + A.a(), "radd")
            self.assertEqual(v - A.a(), "rsub")
            self.assertEqual(v * A.a(), "rmul")
            self.assertEqual(v @ A.a(), "rmatmul")
            self.assertEqual(v / A.a(), "rtruediv")
            self.assertEqual(v // A.a(), "rfloordiv")
            if type(v) != str:
                self.assertEqual(v % A.a(), "rmod")
            self.assertEqual(v ** A.a(), "rpow")
            self.assertEqual(v << A.a(), "rlshift")
            self.assertEqual(v >> A.a(), "rrshift")
            self.assertEqual(v & A.a(), "rand")
            self.assertEqual(v ^ A.a(), "rxor")
            self.assertEqual(v | A.a(), "ror")
            with self.assertRaises(Exception):
                A.a() + v
            with self.assertRaises(Exception):
                A.a() - v
            with self.assertRaises(Exception):
                A.a() * v
            with self.assertRaises(Exception):
                A.a() @ v
            with self.assertRaises(Exception):
                A.a() / v
            with self.assertRaises(Exception):
                A.a() // v
            with self.assertRaises(Exception):
                A.a() % v
            with self.assertRaises(Exception):
                A.a() ** v
            with self.assertRaises(Exception):
                A.a() << v
            with self.assertRaises(Exception):
                A.a() >> v
            with self.assertRaises(Exception):
                A.a() & v
            with self.assertRaises(Exception):
                A.a() ^ v
            with self.assertRaises(Exception):
                A.a() | v

    def test_alternative_missing_inplace_operators_fallback(self):
        A = Alternative("A", a={'a': int}, b={'b': str},
                        __add__=lambda self, other: "worked",
                        __sub__=lambda self, other: "worked",
                        __mul__=lambda self, other: "worked",
                        __matmul__=lambda self, other: "worked",
                        __truediv__=lambda self, other: "worked",
                        __floordiv__=lambda self, other: "worked",
                        __mod__=lambda self, other: "worked",
                        __pow__=lambda self, other: "worked",
                        __lshift__=lambda self, other: "worked",
                        __rshift__=lambda self, other: "worked",
                        __and__=lambda self, other: "worked",
                        __or__=lambda self, other: "worked",
                        __xor__=lambda self, other: "worked",
                        )

        v = A.a()
        v += 10
        self.assertEqual(v, "worked")
        v = A.a()
        v -= 10
        self.assertEqual(v, "worked")
        v = A.a()
        v *= 10
        self.assertEqual(v, "worked")
        v = A.a()
        v @= 10
        self.assertEqual(v, "worked")
        v = A.a()
        v /= 10
        self.assertEqual(v, "worked")
        v = A.a()
        v //= 10
        self.assertEqual(v, "worked")
        v = A.a()
        v %= 10
        self.assertEqual(v, "worked")
        v = A.a()
        v **= 10
        self.assertEqual(v, "worked")
        v = A.a()
        v <<= 10
        self.assertEqual(v, "worked")
        v = A.a()
        v >>= 10
        self.assertEqual(v, "worked")
        v = A.a()
        v &= 10
        self.assertEqual(v, "worked")
        v = A.a()
        v |= 10
        self.assertEqual(v, "worked")
        v = A.a()
        v ^= 10
        self.assertEqual(v, "worked")

    def test_list_and_tuple_of_compare(self):
        things = [
            ListOf(int)([1, 2]),
            ListOf(int)([2, 2]),
            ListOf(int)([1, 3]),
            ListOf(int)([1, 2, 3]),
            ListOf(int)([2, 2, 3]),
            ListOf(int)([1, 3, 3]),
            ListOf(float)([1, 2]),
            ListOf(float)([2, 2]),
            ListOf(float)([1, 3]),
            ListOf(float)([1, 2, 3]),
            ListOf(float)([2, 2, 3]),
            ListOf(float)([1, 3, 3]),
            ListOf(float)([1.0, 2.0]),
            TupleOf(int)([1, 2]),
            TupleOf(int)([2, 2]),
            TupleOf(int)([1, 3]),
            TupleOf(int)([1, 2, 3]),
            TupleOf(int)([2, 2, 3]),
            TupleOf(int)([1, 3, 3]),
            TupleOf(float)([1, 2]),
            TupleOf(float)([2, 2]),
            TupleOf(float)([1, 3]),
            TupleOf(float)([1, 2, 3]),
            TupleOf(float)([2, 2, 3]),
            TupleOf(float)([1, 3, 3]),
            TupleOf(float)([1.0, 2.0]),
            Tuple(int, float)((1, 2)),
            Tuple(int, float, int)((1, 2, 3)),
            Tuple(int, float)((2, 2)),
            Tuple(int, float, int)((2, 2, 3)),
            NamedTuple(x=int, y=float)((1, 2)),
            NamedTuple(x=int, y=float, z=float)((1, 2, 3)),
            NamedTuple(x=int, y=float)((2, 2)),
            NamedTuple(x=int, y=float, z=int)((2, 2, 3))
        ]

        for t1 in things:
            for t2 in things:
                print(t1, t2)
                # lists and tuples are not themselves comparable
                if ("List" in str(type(t1))) != ("List" in str(type(t2))):
                    with self.assertRaises(TypeError):
                        t1 < t2

                    with self.assertRaises(TypeError):
                        t1 > t2

                    with self.assertRaises(TypeError):
                        t1 <= t2

                    with self.assertRaises(TypeError):
                        t1 >= t2

                    self.assertTrue(t1 != t2)
                    self.assertFalse(t1 == t2)
                else:
                    typ = list if "List" in str(type(t1)) else tuple

                    t1Untyped = typ(t1)
                    t2Untyped = typ(t2)

                    self.assertEqual(t1 < t2, t1Untyped < t2Untyped)
                    self.assertEqual(t1 <= t2, t1Untyped <= t2Untyped)
                    self.assertEqual(t1 > t2, t1Untyped > t2Untyped)
                    self.assertEqual(t1 >= t2, t1Untyped >= t2Untyped)
                    self.assertEqual(t1 != t2, t1Untyped != t2Untyped)
                    self.assertEqual(t1 == t2, t1Untyped == t2Untyped)

    def test_docstrings_all_types(self):
        # TODO: actually test all types
        types = [
            bool,
            Int8,
            Int16,
            Int32,
            int,
            UInt8,
            UInt16,
            UInt32,
            UInt64,
            str,
            bytes,
            ListOf(str),
            TupleOf(int),
            Tuple(int, str, int),
            NamedTuple(a=int, b=str),
            Dict(int, str),
            ConstDict(int, str),
            Set(str),
            Alternative("a", s1={'f1': int, 'f2': str}, s2={'f3': bool}),
            Value(3),
            PointerTo(str),
        ]

        good = True
        for T in types:
            type_docstring = T.__doc__
            self.assertTrue(type_docstring, f"Type {T} missing docstring")
            for a in dir(T):
                m = getattr(T, a)
                if callable(m):
                    docstring = m.__doc__

                    # Ignore certain magic methods that don't always have docstrings, even for builtin types.
                    # And ignore some specific Alternative subtypes defined in the test cases.
                    if a in ['__format__', '__getnewargs__', 's1', 's2']:
                        continue
                    if not docstring:
                        print(f"Type {T} missing docstring for '{a}'")
                        good = False
                        continue

                    self.assertTrue(docstring, f"Type {T} missing docstring for '{a}'")

                    # Check for overlong lines
                    max_line_len = 80
                    if a in ['__index__']:  # Ignore some builtin docstrings with overlong lines.
                        continue
                    for l in docstring.splitlines():
                        if len(l) > max_line_len:
                            print(f"docstring line too long {len(l)} > {max_line_len} for '{T}.{a}':\n{l}")
                            good = False

        self.assertTrue(good)  # see output for specific problems

    @flaky(max_runs=3, min_passes=1)
    def test_list_of_uint8_from_bytes_perf(self):
        someBytes = b"asdf" * 1024 * 1024

        t0 = time.time()
        ListOf(UInt8)(someBytes)
        t1 = time.time()

        elapsed = (t1 - t0)

        # I get .001, but if we use the normal interpreter loop, .2
        assert elapsed < .02

    def test_iterate_dict_and_change_size_throws(self):
        x = Dict(int, int)({1: 2})

        with self.assertRaisesRegex(RuntimeError, "dictionary size changed"):
            for k in x:
                x[k + 1] = 2

    def test_iterate_set_and_change_size_throws(self):
        x = Set(int)([1])

        with self.assertRaisesRegex(RuntimeError, "set size changed"):
            for k in x:
                x.add(k + 1)

    def test_construct_named_tuple_with_other_named_tuple(self):
        # we should be matching the names up correctly
        T1 = NamedTuple(x=int, y=str)
        T2 = NamedTuple(y=str, x=int)
        T3 = NamedTuple(z=float, y=str, x=int)

        assert T1(T2(T1(x=10, y='hello'))) == T1(x=10, y='hello')

        # we can construct a bigger tuple from a smaller one
        assert T3(T2(x=10, y='hello')).y == 'hello'

        # but not in reverse
        with self.assertRaises(TypeError):
            T2(T3(x=10, y='hello'))

    def test_dict_update_refcounts(self):
        d = Dict(TupleOf(int), TupleOf(int))()
        k = TupleOf(int)([1])
        v = TupleOf(int)([1, 2])

        n = NamedTuple(k=TupleOf(int), v=TupleOf(int))(k=k, v=v)

        assert _types.refcount(k) == 2
        assert _types.refcount(v) == 2

        d.update({n.k: n.v})

        assert _types.refcount(k) == 3
        assert _types.refcount(v) == 3

        d.clear()

        assert _types.refcount(k) == 2
        assert _types.refcount(v) == 2

        d[k] = v
        assert _types.refcount(k) == 3
        assert _types.refcount(v) == 3

        d.pop(k)
        assert _types.refcount(k) == 2
        assert _types.refcount(v) == 2

        d[k] = v
        assert _types.refcount(k) == 3
        assert _types.refcount(v) == 3

        del d[k]
        assert _types.refcount(k) == 2
        assert _types.refcount(v) == 2

        d[k] = v
        assert _types.refcount(k) == 3
        assert _types.refcount(v) == 3

        d = None
        assert _types.refcount(k) == 2
        assert _types.refcount(v) == 2

        n = None
        assert _types.refcount(k) == 1
        assert _types.refcount(v) == 1

    def test_dict_from_const_dict_refcounts(self):
        D = Dict(TupleOf(int), TupleOf(int))
        CD = ConstDict(TupleOf(int), TupleOf(int))
        k1 = TupleOf(int)([1])
        k2 = TupleOf(int)([1])

        assert _types.refcount(k1) == 1
        assert _types.refcount(k2) == 1

        cd = CD({k1: k2})

        assert _types.refcount(k1) == 2
        assert _types.refcount(k2) == 2

        d = D(cd)
        assert _types.refcount(k1) == 3
        assert _types.refcount(k2) == 3

        cd = None
        assert _types.refcount(k1) == 2
        assert _types.refcount(k2) == 2

        d = None  # noqa
        assert _types.refcount(k1) == 1
        assert _types.refcount(k2) == 1

    def test_set_refcounts_tupleof_int(self):
        s = Set(TupleOf(int))()

        k1 = TupleOf(int)([1])
        k2 = TupleOf(int)([2])

        s.add(k1)
        assert _types.refcount(k1) == 2
        s.discard(k1)
        assert _types.refcount(k1) == 1

        s.update([k1, k2])
        s.discard(k1)
        s.discard(k2)

        assert _types.refcount(k1) == 1

    def test_const_dict_from_dict_refcounts(self):
        CD = ConstDict(TupleOf(int), TupleOf(int))

        k1 = TupleOf(int)([1])
        k2 = TupleOf(int)([2])

        d = Dict(TupleOf(int), TupleOf(int))()
        d[k1] = k2

        assert _types.refcount(k1) == 2
        assert _types.refcount(k2) == 2
        assert _types.refcount(d) == 1
        cd2 = CD(d)

        assert _types.refcount(k1) == 3
        assert _types.refcount(k2) == 3
        assert _types.refcount(d) == 1

        cd2 = None  # noqa

        assert _types.refcount(k1) == 2
        assert _types.refcount(k2) == 2

        d = None

        assert _types.refcount(k1) == 1
        assert _types.refcount(k2) == 1

    def test_list_of_constructor_from_numpy(self):
        array = numpy.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])

        assert ListOf(float)(array[0]) == ListOf(float)([0.0, 1.0, 2.0])

        assert ListOf(float)(array[:, 1]) == ListOf(float)([1.0, 4.0])

        assert ListOf(float)(array.transpose()[2]) == ListOf(float)([2.0, 5.0])

    def test_subclass_of(self):
        class C(Class):
            pass

        T = SubclassOf(C)

        assert T.Type is C

        with self.assertRaisesRegex(TypeError, 'Cannot construct a SubclassOf\\(C\\) from the type "hi"'):
            T("hi")

        with self.assertRaisesRegex(TypeError, "Cannot construct a SubclassOf\\(C\\) from the type int"):
            T(int)

        class D(C):
            pass

        class E(Class):
            pass

        with self.assertRaisesRegex(TypeError, "Cannot construct"):
            T(E)

        assert T(D) is D

        lst = ListOf(SubclassOf(C))()

        lst.append(D)
        lst.append(C)

        assert lst[0] is D
        assert lst[1] is C

        assert issubclass(T, SubclassOf)

        class F(D, Final):
            pass

        assert SubclassOf(F) is Value(F)
        assert SubclassOf(F)(F) is F

        lst = ListOf(SubclassOf(F))()
        lst.append(F)
        assert lst[0] == F
