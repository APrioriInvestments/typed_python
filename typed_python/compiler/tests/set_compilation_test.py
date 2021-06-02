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

from typed_python import (
    Set, ListOf, Entrypoint, Compiled, Tuple, TupleOf, NamedTuple, Dict, ConstDict, OneOf,
    UInt8
)
from typed_python.compiler.type_wrappers.set_wrapper import (
    set_union, set_intersection, set_difference,
    set_symmetric_difference, set_union_multiple, set_intersection_multiple, set_difference_multiple,
    set_disjoint, set_subset, set_subset_iterable, set_superset, set_proper_subset, set_equal, set_not_equal,
    set_update, set_intersection_update, set_difference_update, set_symmetric_difference_update,
    initialize_set_from_other_implicit_containers,
    initialize_set_from_other_upcast_containers
)

from flaky import flaky
import typed_python._types as _types
import time
import numpy
import unittest
import pytest


def result_or_exception(f, *p):
    try:
        return f(*p)
    except Exception as e:
        return str(e)


def some_fibs():
    a = 0
    b = 1
    while b < 100:
        yield b
        a, b = b, a + b


class TestSetCompilation(unittest.TestCase):
    def test_can_copy_set(self):
        @Entrypoint
        def f(x: Set(int)):
            y = x
            return y

        self.assertEqual(f(Set(int)({1, 2})), {1, 2})

        @Entrypoint
        def reversedNative(x: ListOf(Set(int))):
            res = ListOf(Set(int))()

            i = len(x) - 1
            while i >= 0:
                res.append(x[i])
                i -= 1

            return res

        for length in range(100):
            sets = [{x * 2 + 1} for x in range(length)]

            aList = ListOf(Set(int))(sets)

            refcounts = [_types.refcount(x) for x in aList]
            aListRev = reversedNative(aList)
            self.assertEqual(aListRev, list(reversed(sets)))
            aListRev = None

            refcounts2 = [_types.refcount(x) for x in aList]

            self.assertEqual(refcounts, refcounts2)

    def test_set_length(self):
        @Entrypoint
        def set_len(x):
            return len(x)

        x = Set(int)({1})

        self.assertEqual(set_len(x), 1)
        x.add(2)

        self.assertEqual(set_len(x), 2)

        x.remove(2)

        self.assertEqual(set_len(x), 1)

        x.clear()
        self.assertEqual(set_len(x), 0)

    def test_set_in(self):
        @Entrypoint
        def set_in(x, y):
            return y in x

        x = Set(int)()

        x.add(1)

        self.assertEqual(set_in(x, 1), True)
        self.assertEqual(set_in(x, 2), False)
        # TODO: Is this the desired behavior?
        with self.assertRaises(TypeError):
            set_in(x, 1.5)
        with self.assertRaises(TypeError):
            set_in(x, 1.0)

        x1 = Set(float)()

        x1.add(1.0)
        self.assertEqual(set_in(x1, 1.0), True)
        self.assertEqual(set_in(x1, 1.2), False)
        assert set_in(x1, 1)
        assert 1 in x1

    def test_set_add(self):
        S = Set(int)

        @Entrypoint
        def set_add(s, k):
            s.add(k)

        x = S({1})

        self.assertEqual(x, {1})
        set_add(x, 9)
        self.assertEqual(x, {1, 9})
        set_add(x, 4)
        self.assertEqual(x, {1, 4, 9})

        with self.assertRaises(TypeError):
            x.add(1.5)

        with self.assertRaises(TypeError):
            set_add(x, 1.5)

        x.clear()
        self.assertEqual(x, set())

        @Entrypoint
        def set_many(s, count):
            for i in range(count):
                s.add(i * i)

        set_many(x, 1000)
        self.assertEqual(x, {i * i for i in range(1000)})

        T = Set(str)
        y = T({"abc"})
        self.assertEqual(y, {"abc"})
        set_add(y, "def")
        self.assertEqual(y, {"abc", "def"})
        set_add(y, "ghi")
        self.assertEqual(y, {"abc", "def", "ghi"})
        y.clear()
        self.assertEqual(y, set())

        @Entrypoint
        def set_many2(s, count):
            for i in range(count):
                s.add(str(i * i))

        set_many2(y, 1000)
        self.assertEqual(y, {str(i * i) for i in range(1000)})

        @Compiled
        def f(x: str):
            s = Set(str)({"initial", "value"})
            s.add(x)
            s.add(x + x)
            return s

        self.assertEqual(f("a"), {"initial", "value", "a", "aa"})
        self.assertEqual(f("xyz" * 100), {"initial", "value", "xyz" * 100, "xyz" * 200})

    def test_set_remove_discard(self):
        @Entrypoint
        def set_remove(s, k):
            s.remove(k)

        @Entrypoint
        def set_discard(s, k):
            s.discard(k)

        s = Set(int)({1, 2, 3})
        with self.assertRaises(KeyError):
            set_remove(s, 4)
        set_remove(s, 2)
        self.assertEqual(s, {1, 3})
        set_remove(s, 1)
        self.assertEqual(s, {3})
        set_remove(s, 3)
        self.assertEqual(s, set())
        with self.assertRaises(KeyError):
            set_remove(s, 4)

        s = Set(str)({'aa', 'bb', 'cc'})
        with self.assertRaises(KeyError):
            set_remove(s, 'dd')
        set_remove(s, 'bb')
        self.assertEqual(s, {'aa', 'cc'})
        set_remove(s, 'aa')
        self.assertEqual(s, {'cc'})
        set_remove(s, 'cc')
        self.assertEqual(s, set())
        with self.assertRaises(KeyError):
            set_remove(s, 'dd')

        s = Set(int)({1, 2, 3})
        set_discard(s, 4)
        self.assertEqual(s, {1, 2, 3})
        set_remove(s, 2)
        self.assertEqual(s, {1, 3})
        set_remove(s, 1)
        self.assertEqual(s, {3})
        set_remove(s, 3)
        self.assertEqual(s, set())
        set_discard(s, 4)
        self.assertEqual(s, set())

        s = Set(str)({'aa', 'bb', 'cc'})
        set_discard(s, 'dd')
        self.assertEqual(s, {'aa', 'bb', 'cc'})
        set_remove(s, 'bb')
        self.assertEqual(s, {'aa', 'cc'})
        set_remove(s, 'aa')
        self.assertEqual(s, {'cc'})
        set_remove(s, 'cc')
        self.assertEqual(s, set())
        set_discard(s, 'dd')
        self.assertEqual(s, set())

    def test_set_clear(self):
        @Entrypoint
        def set_clear(s):
            s.clear()

        s1 = Set(int)()
        for i in range(1000):
            s1.add(i * i)
        self.assertEqual(len(s1), 1000)
        set_clear(s1)
        self.assertEqual(len(s1), 0)

        s2 = Set(str)()
        for i in range(1000, 2000):
            s2.add(str(i))
        self.assertEqual(len(s2), 1000)
        set_clear(s2)
        self.assertEqual(len(s2), 0)

    def test_set_assign_and_copy(self):

        @Entrypoint
        def set_assign_and_modify_original(s, x, y):
            s2 = s
            s.add(x)
            s.remove(y)
            return s2

        @Entrypoint
        def set_copy_and_modify_original(s, x, y):
            s2 = s.copy()
            s.add(x)
            s.remove(y)
            return s2

        s = Set(str)(set("abc"))
        self.assertEqual(set_assign_and_modify_original(s, 'q', 'b'), set("acq"))
        s = Set(str)(set("abc"))
        self.assertEqual(set_copy_and_modify_original(s, 'q', 'b'), set("abc"))

    def test_adding_to_sets(self):

        @Entrypoint
        def f(count):
            for salt in range(count):
                for count in range(10):
                    d = Set(str)()

                    for i in range(count):
                        d.add(str(salt) + "hi" + str(i))

                        for j in range(i):
                            if str(salt) + "hi" + str(j) not in d:
                                return False

            return True

        self.assertTrue(f(20000))

    def test_set_destructors(self):
        @Entrypoint
        def f():
            x = Set(str)({'h', 'i'})
            y = Set(int)({1, 2, 3})
            x.add('a')
            y.add(4)
            return "OK"

        f()

    def test_set_pop(self):
        @Entrypoint
        def set_pop(s):
            return s.pop()

        s = Set(str)()
        s.add('a')
        s.add('b')

        self.assertEqual(set_pop(s), 'a')
        self.assertEqual(s, {'b'})
        self.assertEqual(set_pop(s), 'b')
        self.assertEqual(s, set())
        with self.assertRaises(KeyError):
            set_pop(s)

        @Compiled
        def set_to_list(s: Set(int)) -> ListOf(int):
            result = ListOf(int)()
            try:
                while (True):
                    result.append(s.pop())
            except KeyError:
                return result
            return result

        original_set = Set(int)()
        for _ in range(10000):
            original_set.add(numpy.random.choice(1000000))
        temp_set = original_set.copy()
        new_set = Set(int)(set_to_list(temp_set))
        self.assertEqual(temp_set, set())
        self.assertEqual(original_set, new_set)

    @flaky(max_runs=3, min_passes=1)
    def test_set_perf(self):
        def set_copy_discard(s: Set(int), count: int):
            for i in range(2, count):
                s.add(i)
            for i in s.copy():
                for j in range(2, count):
                    s.discard(i * j)

        compiled_set_copy_discard = Compiled(set_copy_discard)

        t0 = time.time()
        aSet = Set(int)()
        set_copy_discard(aSet, 1000)

        t1 = time.time()
        aSet2 = Set(int)()
        compiled_set_copy_discard(aSet2, 1000)
        t2 = time.time()

        self.assertEqual(aSet, aSet2)

        ratio = (t1 - t0) / (t2 - t1)

        self.assertGreater(ratio, 6)

        print("Speedup was ", ratio)

    def test_set_binop(self):
        S = Set(int)
        sets = []
        bits = 4
        for s in range(1<<bits):
            one_set = set()
            for i in range(bits):
                if s & 1<<i:
                    one_set.add(i)
            sets.append(one_set)

        def union(x: S, y: S):
            return x.union(y)

        def intersection(x: S, y: S):
            return x.intersection(y)

        def difference(x: S, y: S):
            return x.difference(y)

        def symmetric_difference(x: S, y: S):
            return x.symmetric_difference(y)

        def op_union(x: S, y: S):
            return x | y

        def op_intersection(x: S, y: S):
            return x & y

        def op_difference(x: S, y: S):
            return x - y

        def op_symmetric_difference(x: S, y: S):
            return x ^ y

        fns = [union, op_union, intersection, op_intersection,
               difference, op_difference, symmetric_difference, op_symmetric_difference]
        for f in fns:
            for left in sets:
                for right in sets:
                    r1 = f(left, right)
                    r2 = f(S(left), S(right))  # retesting interpreter here
                    r3 = Compiled(f)(S(left), S(right))  # testing compiled code
                    self.assertEqual(r1, r2, (f, left, right))
                    self.assertEqual(r2, r3, (f, left, right))

        def equal(x: S, y: S):
            return x == y

        def notequal(x: S, y: S):
            return x != y

        def subset(x: S, y: S):
            return x.issubset(y)

        def superset(x: S, y: S):
            return x.issuperset(y)

        def op_subset(x: S, y: S):
            return x <= y

        def op_superset(x: S, y: S):
            return x >= y

        def isdisjoint(x: S, y: S):
            return x.isdisjoint(y)

        comparison_fns = [equal, notequal, subset, superset, op_subset, op_superset, isdisjoint]
        for f in comparison_fns:
            for left in sets:
                for right in sets:
                    r1 = f(left, right)
                    r2 = f(S(left), S(right))  # retesting interpreter here
                    r3 = Compiled(f)(S(left), S(right))  # testing compiled code
                    self.assertEqual(r1, r2, (f, left, right))
                    self.assertEqual(r2, r3, (f, left, right))

        def union0(x: S):
            return x.union()

        def intersection0(x: S):
            return x.intersection()

        def difference0(x: S):
            return x.difference()

        def symmetric_difference0(x: S):
            return x.symmetric_difference()

        for left in sets:
            for f in [union0, intersection0, difference0]:
                r1 = f(left)
                r2 = f(S(left))  # retesting interpreter here
                r3 = Compiled(f)(S(left))  # testing method with 0 arguments
                self.assertEqual(r1, r2)
                self.assertEqual(r2, r3)
            with self.assertRaises(TypeError):
                Compiled(symmetric_difference0)(S(left))

        def union2(x: S, y: S, z: S):
            return x.union(y, z)

        def intersection2(x: S, y: S, z: S):
            return x.intersection(y, z)

        def difference2(x: S, y: S, z: S):
            return x.difference(y, z)

        def symmetric_difference2(x: S, y: S, z: S):
            return x.symmetric_difference(y, z)

        for x in sets:
            for y in sets:
                for z in sets:
                    for f in [union2, intersection2, difference2]:
                        r1 = f(x, y, z)
                        r2 = f(S(x), S(y), S(z))  # retesting interpreter here
                        r3 = Compiled(f)(S(x), S(y), S(z))  # testing method with 2 arguments
                        self.assertEqual(r1, r2)
                        self.assertEqual(r2, r3)

                    with self.assertRaises(TypeError):
                        Compiled(symmetric_difference2)(S(x), S(y), S(z))

        def update(x: S, y: S):
            x.update(y)

        def intersection_update(x: S, y: S):
            x.intersection_update(y)

        def difference_update(x: S, y: S):
            x.difference_update(y)

        def symmetric_difference_update(x: S, y: S):
            return x.symmetric_difference_update(y)

        for x in sets:
            for y in sets:
                for f in [update, intersection_update, difference_update, symmetric_difference_update]:
                    x1 = x.copy()
                    x2 = S(x.copy())
                    x3 = S(x.copy())
                    f(x1, y)
                    f(x2, S(y))
                    Compiled(f)(x3, S(y))
                    self.assertEqual(x1, x2)
                    self.assertEqual(x1, x3)

        def update2(x: S, y: S, z: S):
            x.update(y, z)

        def intersection_update2(x: S, y: S, z: S):
            x.intersection_update(y, z)

        def difference_update2(x: S, y: S, z: S):
            x.difference_update(y, z)

        def symmetric_difference_update2(x: S, y: S, z: S):
            x.symmetric_difference_update(y, z)

        for x in sets:
            for y in sets:
                for z in sets:
                    for f in [update2, intersection_update2, difference_update2]:
                        x1 = x.copy()
                        x2 = S(x.copy())
                        x3 = S(x.copy())
                        f(x1, y, z)
                        f(x2, S(y), S(z))
                        Compiled(f)(x3, S(y), S(z))
                        self.assertEqual(x1, x2)
                        self.assertEqual(x1, x3)
                    with self.assertRaises(TypeError):
                        Compiled(symmetric_difference_update2)(S(x), S(y), S(z))

    def test_set_iterable_comparisons(self):
        S = Set(int)

        def issubset(x, y):
            return x.issubset(y)

        def issuperset(x, y):
            return x.issuperset(y)

        def isdisjoint(x, y):
            return x.isdisjoint(y)

        test_cases = [
            {1, 2, 3, 5, 8, 13, 21, 34, 55, 89},
            {1, 2, 3, 5, 13, 21, 34, 55, 89},
            {1, 2, 3, 5, 9, 8, 13, 21, 34, 55, 89},
            {1, 2, 3, 5, 9, 13, 21, 34, 55, 89},
            {2, 3, 5, 8, 13, 21, 34, 55, 89},
            {1, 2, 3, 5, 8, 13, 21, 34, 55},
            {1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 90},
            {0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89},
            {},
        ]
        for f in [issubset, issuperset, isdisjoint]:
            for s in test_cases:
                r1 = f(S(s), some_fibs())
                r2 = Entrypoint(f)(S(s), some_fibs())
                self.assertEqual(r1, r2)

                r1 = f(S(s), range(3))
                r2 = Entrypoint(f)(S(s), range(3))
                self.assertEqual(r1, r2)

                r1 = f(S(s), range(10))
                r2 = Entrypoint(f)(S(s), range(10))
                self.assertEqual(r1, r2)

        fail_cases = [
            Set(str)("abc"),
            ListOf(str)("abc"),
            TupleOf(str)("abc"),
            Dict(str, int)({}.fromkeys("xyz", 1)),
            ConstDict(str, int)({}.fromkeys("xyz", 1)),
        ]
        for f in [issubset, issuperset, isdisjoint]:
            s = S({1, 2, 3})
            for v in fail_cases:
                with self.assertRaises(TypeError):
                    f(s, v)
                with self.assertRaises(TypeError):
                    Entrypoint(f)(s, v)

    def test_set_iterable_updates(self):
        S = Set(int)

        def update(x, y):
            x.update(y)
            return x

        def intersection_update(x, y):
            x.intersection_update(y)
            return x

        def difference_update(x, y):
            x.difference_update(y)
            return x

        def symmetric_difference_update(x, y):
            x.symmetric_difference_update(y)
            return x

        def is_subset(x, y):
            return x.issubset(y)

        def is_superset(x, y):
            return x.issuperset(y)

        def is_disjoint(x, y):
            return x.isdisjoint(y)

        test_sets = [
            {-1, -2},
            set(range(20)),
            {1, 3, 5},
            {-1, 3, 5},
            set(range(20)),
            set(some_fibs()),
            set(some_fibs()) | {0},
            set(some_fibs()).difference({5}),
        ]
        test_iterables = [
            [1, 2, 2, 3],
            [-1, -1, 2, 2, 3],
            (89, 4, 89),
            ListOf(int)([1, 4, 4, 5, 5]),
            TupleOf(int)(some_fibs()),
            Dict(int, int)({}.fromkeys(range(20), 1)),
            ConstDict(int, int)({}.fromkeys(range(20), 1)),
        ]
        for f in [update, intersection_update, difference_update, symmetric_difference_update, is_subset, is_superset, is_disjoint]:
            for s in test_sets:
                for v in test_iterables:
                    s0 = s.copy()
                    r0 = f(s0, v)

                    s1 = S(s)
                    r1 = f(s1, v)
                    self.assertEqual(s0, s1)  # test interpreted side effect
                    self.assertEqual(r0, r1)  # test interpreted return value

                    s2 = S(s)
                    r2 = Entrypoint(f)(s2, v)
                    self.assertEqual(s0, s2)  # test compiled side effect
                    self.assertEqual(r0, r2)  # test compiled return value

                # now test range() and some_fibs() separately
                s0 = s.copy()
                r0 = f(s0, range(10))

                s1 = S(s)
                r1 = f(s1, range(10))
                self.assertEqual(s0, s1, (f, s, "range()"))  # test interpreted result
                self.assertEqual(r0, r1, (f, s, "range()"))  # test interpreted result

                s2 = S(s)
                r2 = Entrypoint(f)(s2, range(10))
                self.assertEqual(s0, s2, (f, s, "range()"))  # test compiled result
                self.assertEqual(r0, r2, (f, s, "range()"))  # test compiled result

                s0 = s.copy()
                r0 = f(s0, some_fibs())

                s1 = S(s)
                r1 = f(s1, some_fibs())
                self.assertEqual(s0, s1, (f, s, "some_fibs()"))  # test interpreted result
                self.assertEqual(r0, r1, (f, s, "some_fibs()"))  # test interpreted result

                s2 = S(s)
                r2 = Entrypoint(f)(s2, some_fibs())
                self.assertEqual(s0, s2, (f, s, "some_fibs()"))  # test compiled result
                self.assertEqual(r0, r2, (f, s, "some_fibs()"))  # test compiled result

        fail_iterables = [
            [[]],
            ['a', 1, 2, 3],
            ['x', -1, -1, 2, 3],
            ('b', 2, 'b'),
            ListOf(OneOf(int, str))(['q', 1, 4, 4, 5]),
            TupleOf(OneOf(int, str))(('a', 1, 2, 3)),
            Dict(str, int)({'a': 1, 'b': 2, 'c': 3}),
            ConstDict(str, int)({'a': 1, 'b': 2, 'c': 3}),
        ]
        for f in [update, intersection_update, difference_update, symmetric_difference_update, is_subset, is_superset, is_disjoint]:
            s = {1, 2, 3}
            for v in fail_iterables:
                print("TEST", v)
                s1 = S(s)
                with self.assertRaises(TypeError):
                    f(s1, v)

                s2 = S(s)
                with self.assertRaises(TypeError):
                    Entrypoint(f)(s2, v)

    @flaky(max_runs=3, min_passes=1)
    @pytest.mark.skipif('sys.platform=="darwin"')
    def test_set_binop_perf(self):
        for T in [int, str]:
            S = Set(T)

            sets = []
            bits = 6
            for s in range(1 << bits):
                one_set = set()
                for i in range(bits):
                    if s & 1 << i:
                        one_set.add(T(i))
                sets.append(one_set)

            def union(x: S, y: S):
                return x.union(y)

            def intersection(x: S, y: S):
                return x.intersection(y)

            def difference(x: S, y: S):
                return x.difference(y)

            def symmetric_difference(x: S, y: S):
                return x.symmetric_difference(y)

            sets.append({T(x) for x in range(99)})
            sets.append({T(x) for x in range(100)})
            sets.append({T(x) for x in range(0, 100, 2)})
            sets.append({T(x) for x in range(1, 100, 2)})
            sets.append({T(x) for x in range(9999)})
            sets.append({T(x) for x in range(10000)})
            sets.append({T(x) for x in range(0, 10000, 2)})
            sets.append({T(x) for x in range(1, 10000, 2)})
            ourSets = [S(s) for s in sets]
            for f, threshold in [(union, 0.7), (intersection, 0.1), (difference, 0.4), (symmetric_difference, .5)]:
                repeat = 2
                a1 = 0
                a2 = 0
                c_f = Compiled(f)

                t0 = time.time()
                for _ in range(repeat):
                    for x in sets:
                        for y in sets:
                            result = f(x, y)
                            a1 += len(result)

                t1 = time.time()
                for _ in range(repeat):
                    for x in ourSets:
                        for y in ourSets:
                            result = c_f(x, y)
                            a2 += len(result)

                t2 = time.time()
                ratio = (t1 - t0) / (t2 - t1)

                self.assertEqual(a1, a2, (f, S))

                print("Speedup was", (t1 - t0), "/", (t2-t1), "=", ratio, "for", f, "on", S, ": ")

                # performance could be improved
                self.assertGreater(ratio, threshold)

    def test_set_iteration(self):
        def set_to_list(s: Set(str)) -> ListOf(str):
            result = ListOf(str)()
            for i in s:
                result.append(i)
            return result

        s = Set(str)("iteration")
        r1 = set_to_list(s)
        r2 = Compiled(set_to_list)(s)
        self.assertEqual(r1, r2)

    def test_set_refcounting(self):
        TOI = TupleOf(int)
        aTup = TOI((1, 2, 3))
        aTup2 = TOI((1, 2, 3, 4))
        aTup3 = TOI((1, 2, 3, 4, 5))

        x = Set(TOI)()
        x.add(aTup2)
        x.add(aTup)
        x.add(aTup3)

        @Entrypoint
        def addItem(s, k):
            s.add(k)

        @Entrypoint
        def removeItem(s, k):
            s.remove(k)

        @Entrypoint
        def getItem(s, k):
            return k in s

        self.assertTrue(getItem(x, aTup))

        addItem(x, aTup)

        self.assertTrue(getItem(x, aTup))
        removeItem(x, aTup2)
        self.assertFalse(getItem(x, aTup2))

        self.assertEqual(_types.refcount(aTup), 2)
        self.assertEqual(_types.refcount(aTup2), 1)
        self.assertEqual(_types.refcount(aTup3), 2)

        x = None

        self.assertEqual(_types.refcount(aTup), 1)
        self.assertEqual(_types.refcount(aTup2), 1)
        self.assertEqual(_types.refcount(aTup3), 1)

    def test_set_pop_many(self):
        @Entrypoint
        def f(x: Set(str)):
            keys = ListOf(str)()

            for key in x:
                keys.append(key)

            for _ in keys:
                x.pop()

        x = Set(str)()

        for i in range(1000):
            x.add(str(i))
        self.assertEqual(len(x), 1000)

        f(x)

        self.assertEqual(len(x), 0)

    def test_set_up_and_down(self):
        @Entrypoint
        def f(targets):
            x = Set(int)()

            for target in targets:
                for i in range(len(x), target):
                    assert i not in x
                    x.add(i)
                    assert i in x
                    assert i + 1 not in x

                for i in range(target, len(x)):
                    assert i in x
                    x.remove(i)
                    assert i not in x

                assert len(x) == target

        C = 10
        for i1 in range(C):
            for i2 in range(C):
                for i3 in range(C):
                    for i4 in range(C):
                        for i5 in range(C):
                            f(ListOf(int)([i1, i2, i3, i4, i5]))

    def test_set_fuzz(self):
        # try adding and removing items repeatedly, in an effort to fill the table up
        @Entrypoint
        def f(actions: ListOf(Tuple(bool, int))):
            x = Set(int)()

            for thing in actions:
                if thing[0]:
                    x.add(thing[1])
                else:
                    x.discard(thing[1])

        f([(True, 3), (False, 17), (False, 3), (True, 34), (True, 36), (True, 0), (False, 0),
           (False, 38), (False, 34), (True, 11), (True, 37), (True, 33), (False, 4), (True, 16)])

        for length in range(5, 15):
            print("Trying length ", length)
            for trials in range(10000):
                actions = []

                for i in range(length):
                    actions.append((numpy.random.uniform() > .5, numpy.random.choice(40)))

                try:
                    f(actions)
                except Exception:
                    print(actions)
                    raise

    def test_set_destructor_complex(self):
        NT = NamedTuple(x=TupleOf(str), y=TupleOf(str))
        aTup = TupleOf(str)(['a'])

        tupRefcount = _types.refcount(aTup)

        x = ListOf(Set(NT))()
        x.append(Set(NT)([NT(x=aTup, y=aTup)]))

        @Entrypoint
        def resizeZero(x):
            x.resize(0)

        resizeZero(x)

        self.assertEqual(tupRefcount, _types.refcount(aTup))

    def test_set_with_neg_one(self):
        # negative one is special because it hashes to -1. Python
        # treats a -1 as an error code (indicating there was
        # an exception). We don't do the same thing, so lets make
        # sure we handle things that hash to -1 correctly
        s = Set(int)()

        @Entrypoint
        def set_add(s, x):
            s.add(x)

        @Entrypoint
        def set_remove(s, x):
            s.remove(x)

        @Entrypoint
        def set_in(x, s):
            return x in s

        self.assertFalse(set_in(-1, s))
        set_add(s, -1)
        self.assertTrue(set_in(-1, s))
        set_remove(s, -1)
        self.assertFalse(set_in(-1, s))

    def test_set_internal_functions(self):
        # These internal functions are already tested, as compiled python functions, in other tests
        # But these tests ensure that codecov counts these functions as "tested"

        s1 = {1, 2, 3}
        s2 = {2, 3, 4, 5}
        self.assertEqual(set_union(s1, s2), s1.union(s2))
        self.assertEqual(set_union(s2, s1), s2.union(s1))
        self.assertEqual(set_intersection(s1, s2), s1.intersection(s2))
        self.assertEqual(set_intersection(s2, s1), s2.intersection(s1))
        self.assertEqual(set_difference(s1, s2), s1.difference(s2))
        self.assertEqual(set_difference(s2, s1), s2.difference(s1))
        self.assertEqual(set_symmetric_difference(s1, s2), s1.symmetric_difference(s2))
        self.assertEqual(set_symmetric_difference(s2, s1), s2.symmetric_difference(s1))

        s3 = {4, 5, 6}
        self.assertEqual(set_union_multiple(s1, s2, s3), s1.union(s2, s3))
        self.assertEqual(set_intersection_multiple(s1, s2, s3), s1.intersection(s2, s3))
        self.assertEqual(set_difference_multiple(s1, s2, s3), s1.difference(s2, s3))

        s4 = {2, 3}
        self.assertEqual(set_disjoint(s3, s4), s3.isdisjoint(s4))
        self.assertEqual(set_disjoint(s3, s3), s3.isdisjoint(s3))
        self.assertEqual(set_subset(s3, s4), s3 <= s4)
        self.assertEqual(set_subset(s3, s3), s3 <= s3)
        self.assertEqual(set_superset(s4, s3), s4 >= s3)
        self.assertEqual(set_superset(s3, s3), s3 >= s3)
        self.assertEqual(set_subset_iterable(s3, s4), s3 <= s4)
        self.assertEqual(set_proper_subset(s3, s4), s3 < s4)
        self.assertEqual(set_equal(s3, s4), s3 == s4)
        self.assertEqual(set_not_equal(s3, s4), s3 != s4)

        set_update(s1, s2)
        set_intersection_update(s1, [5, 6])
        set_difference_update(s1, [3, 7])
        set_symmetric_difference_update(s1, [2, 8])

        class P_mock:
            ElementType = Set(int)

            def initialize(self, src=None):
                self.s = set()

            def destroy(self):
                pass

            def get(self):
                return self.s

        p1 = P_mock()
        initialize_set_from_other_implicit_containers(p1, [1, 2], True)
        initialize_set_from_other_upcast_containers(p1, [1, 2], True)

    def test_compiled_set_constructors(self):
        def f_set(x):
            return Set(int)(x)

        def f_listof(x):
            return ListOf(int)(x)

        def f_tupleof(x):
            return TupleOf(int)(x)

        test_values = [
            range(10),
            {1, 2},
            [3, 3, 4, 4],
            (5, 6, 5, 6),
            {101: 1, 202: 2},
            Set(int)({1, 2}),
            ListOf(int)([3, 3, 4, 4]),
            TupleOf(int)((5, 6, 5, 6)),
            Tuple(int, int, int, int)((7, 8, 8, 7)),
            Dict(int, int)({101: 1, 202: 2}),
            ConstDict(int, int)({101: 1, 202: 2}),
            ListOf(float)([3.0, 3.5, 4.0, 4.0]),
            TupleOf(float)((5.0, 6.0, 5.0, 6.0)),
            Tuple(float, float, float, float)((7.0, 8.0, 8.0, 7.0)),
            Dict(float, int)({101.0: 1, 102.0: 2}),
            Set(float)({1.0, 2.0}),
        ]

        test_values_mismatch = [
            Set(str)({"a", "b"}),
            ListOf(str)(["c", "c", "d", "d"]),
            TupleOf(str)(("e", "f", "e", "f")),
            Tuple(str, int, int, int)(("g", 8, 8, 7)),
            Dict(str, int)({"x": 1, "y": 2}),
        ]

        # let's test all these types, not just Set
        for f in [f_set, f_listof, f_tupleof]:
            for v in test_values:
                r1 = f(v)
                r2 = Entrypoint(f)(v)
                if r1 != r2:
                    raise Exception(
                        f"Calling {f.__name__} on {v} of type {type(v)} produced different"
                        f" values under entrypoint: {r1} != {r2}"
                    )
                self.assertEqual(r1, r2)

            r1 = f(some_fibs())
            r2 = Entrypoint(f)(some_fibs())
            self.assertEqual(r1, r2)

            for v in test_values_mismatch:
                print(f, v)
                with self.assertRaises(TypeError):
                    f(v)
                with self.assertRaises(TypeError):
                    Entrypoint(f)(v)

        S = Set(TupleOf(TupleOf(int)))

        def f_set_t(x):
            return S(x)

        t1 = TupleOf(TupleOf(int))(((1, 2), (3, 0)))
        t2 = TupleOf(TupleOf(int))(((4, 5, 5), (6,)))

        r1 = f_set_t({t1, t2})
        r2 = Entrypoint(f_set_t)({t1, t2})
        self.assertEqual(r1, r2)

        t3 = TupleOf(TupleOf(OneOf(int, str)))(((7, 7, 7), (8, 9)))
        r1 = f_set_t({t1, t2, t3})
        r2 = Entrypoint(f_set_t)({t1, t2, t3})
        self.assertEqual(r1, r2)

        t4 = TupleOf(TupleOf(OneOf(int, str)))(((7, 7, 7), (8, "x")))
        with self.assertRaises(TypeError):
            f_set_t({t1, t2, t4})
        with self.assertRaises(TypeError):
            Entrypoint(f_set_t)({t1, t2, t4})

    def test_set_aliasing(self):
        T = Set(int)

        t1 = T()
        t2 = T(t1)

        t1.add(10)

        assert len(t2) == 0

    def test_set_aliasing_compiled(self):
        T = Set(int)

        @Entrypoint
        def dup(x):
            return T(x)

        assert dup.resultTypeFor(T).typeRepresentation == T

        t1 = T()
        t2 = dup(t1)

        t1.add(10)

        assert len(t2) == 0

    def test_set_adding_allows_container_upcast(self):
        T = Set(TupleOf(float))

        def f():
            aT = T()

            # these work
            aT.add((1, 2, 3))
            aT.add([1, 2, 3])
            aT.add(TupleOf(UInt8)([1, 2, 3]))

            return len(aT) == 1

        assert f()
        assert Entrypoint(f)()

        aT = T()
        with self.assertRaises(TypeError):
            aT.add([1, 2, "3"])

        with self.assertRaises(TypeError):
            @Entrypoint
            def addIt():
                aT = T()
                aT.add([1, 2, "3"])
            addIt()

    def test_set_size_change_during_iteration_raises(self):
        @Entrypoint
        def checkIt():
            aSet = Set(int)()
            aSet.add(1)

            for x in aSet:
                aSet.add(x+1)

        with self.assertRaisesRegex(RuntimeError, "set size changed"):
            checkIt()
