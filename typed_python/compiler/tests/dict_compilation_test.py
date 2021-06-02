#   Copyright 2017-2019 typed_python Authors
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

from typed_python import Dict, ListOf, Tuple, TupleOf, Entrypoint, OneOf, Set
import typed_python._types as _types
import unittest
import time
import threading
from flaky import flaky
import numpy
import numpy.random


class TestDictCompilation(unittest.TestCase):
    def test_can_copy_dict(self):
        @Entrypoint
        def f(x):
            y = x
            return y

        self.assertEqual(f(Dict(int, int)({1: 2})), {1: 2})

        @Entrypoint
        def reversed(x: ListOf(Dict(int, int))):
            res = ListOf(Dict(int, int))()

            i = len(x) - 1
            while i >= 0:
                res.append(x[i])
                i -= 1

            return res

        for length in range(100):
            dicts = [{x: x * 2 + 1} for x in range(length)]

            aList = ListOf(Dict(int, int))(dicts)

            refcounts = [_types.refcount(x) for x in aList]
            aListRev = reversed(aList)
            self.assertEqual(aListRev, reversed(ListOf(Dict(int, int))(dicts)))
            aListRev = None

            refcounts2 = [_types.refcount(x) for x in aList]

            self.assertEqual(refcounts, refcounts2)

    def test_dict_contains(self):
        @Entrypoint
        def isIn(x, d):
            if x in d:
                return True
            return False

        @Entrypoint
        def isNotIn(x, d):
            if x not in d:
                return True
            return False

        d = Dict(str, TupleOf(int))()
        d['hi'] = (1, 2, 3)

        self.assertTrue(isIn("hi", d))
        self.assertFalse(isNotIn("hi", d))

        self.assertFalse(isIn("boo", d))
        self.assertTrue(isNotIn("boo", d))

    def test_dict_length(self):
        @Entrypoint
        def dict_len(x):
            return len(x)

        x = Dict(int, int)({1: 2})

        self.assertEqual(dict_len(x), 1)
        x[2] = 3

        self.assertEqual(dict_len(x), 2)

        del x[1]

        self.assertEqual(dict_len(x), 1)

    def test_dict_getitem(self):
        @Entrypoint
        def dict_getitem(x, y):
            return x[y]

        x = Dict(int, int)()

        x[1] = 2

        self.assertEqual(dict_getitem(x, 1), 2)

        with self.assertRaisesRegex(KeyError, "2"):
            dict_getitem(x, 2)

    def test_dict_get_default(self):
        @Entrypoint
        def dict_get(x, y):
            return x.get(y, -1.5)

        x = Dict(int, int)()

        x[1] = 2

        self.assertEqual(dict_get(x, 1), 2)
        self.assertEqual(dict_get(x, 2), -1.5)

        with self.assertRaises(TypeError):
            self.assertEqual(dict_get(x, 1.5), 2)

        with self.assertRaises(TypeError):
            self.assertEqual(x[1.5], 2)

    def test_dict_get_nodefault(self):
        @Entrypoint
        def dict_get(x, y):
            return x.get(y)

        x = Dict(int, int)()

        x[1] = 2

        self.assertEqual(dict_get(x, 1), 2)
        self.assertEqual(x.get(1), 2)

        with self.assertRaises(TypeError):
            self.assertEqual(x.get(1.5), 2)

        with self.assertRaises(TypeError):
            self.assertEqual(dict_get(x, 1.5), 2)

        self.assertEqual(dict_get(x, 2), None)
        self.assertEqual(x.get(2), None)

    def test_dict_setitem(self):
        @Entrypoint
        def dict_setitem(d, k, v):
            d[k] = v

        x = Dict(int, str)()

        x[1] = '2'

        dict_setitem(x, 1, '3')

        self.assertEqual(x, {1: '3'})

        dict_setitem(x, 2, '300')

        self.assertEqual(x, {1: '3', 2: '300'})

        with self.assertRaises(TypeError):
            dict_setitem(x, 1.5, '200')

        with self.assertRaises(TypeError):
            x[1.5] = '200'

        @Entrypoint
        def dict_setmany(d, count):
            for i in range(count):
                d[i] = str(i * i)

        dict_setmany(x, 1000)

        self.assertEqual(x, {i: str(i*i) for i in range(1000)})

    def test_dict_with_oneof_keys(self):
        d = Dict(OneOf(None, int), int)()

        d[None] = 10
        d[20] = 20

        @Entrypoint
        def lookup(d, v):
            return d.get(v)

        self.assertEqual(lookup(d, None), 10)
        self.assertEqual(lookup(d, 30), None)
        self.assertEqual(lookup(d, 20), 20)

        self.assertEqual(d.get(None), 10)
        self.assertEqual(d.get(30), None)
        self.assertEqual(d.get(20), 20)

    def test_dict_position_same(self):
        def check(someInts):
            dInterp = Dict(int, int)()
            dCompil = Dict(int, int)()

            for i in someInts:
                dInterp[i] = i

            @Entrypoint
            def putThemIn(d, ints):
                for i in ints:
                    d[i] = i

            putThemIn(dCompil, ListOf(int)(someInts))

            for i in someInts:
                assert dCompil[i] == i

            @Entrypoint
            def checkThem(d, ints):
                for i in ints:
                    assert d[i] == i

            checkThem(dInterp, ListOf(int)(someInts))

        numpy.random.seed(42)

        i = 2
        while i < 100000:
            i = int(i * 1.5)
            check(range(i))
            check(numpy.random.choice(1000000, size=i).tolist())

    def test_adding_to_dicts(self):
        @Entrypoint
        def f(count):
            for salt in range(count):
                for count in range(10):
                    d = Dict(str, int)()

                    for i in range(count):
                        d[str(salt) + "hi" + str(i)] = i

                        for j in range(i):
                            if (str(salt) + "hi" + str(j)) not in d:
                                return False
                            if d[str(salt) + "hi" + str(j)] != j:
                                return False

            return True

        self.assertTrue(f(20000))

    def test_dicts_in_dicts(self):
        @Entrypoint
        def f():
            d = Dict(str, Dict(str, float))()
            d["hi"] = Dict(str, float)()
            d["hi"]["good"] = 100.0
            d["bye"] = Dict(str, float)()
            d2 = Dict(str, Dict(str, float))() # noqa
            return d

        for _ in range(1000):
            d = f()

        self.assertEqual(d['hi']['good'], 100.0)

    def test_dict_destructors(self):
        @Entrypoint
        def f():
            d = Dict(str, Dict(str, float))()
            d["hi"] = Dict(str, float)()
            return "OK"

        f()

    def test_dict_setdefault(self):
        @Entrypoint
        def dict_setdefault(d, k, v):
            return d.setdefault(k, v)

        x = Dict(int, str)()
        x[1] = "a"

        # This should not change the dictionary, and return "a"
        v1 = dict_setdefault(x, 1, "b")
        self.assertEqual(v1, "a")
        self.assertEqual(x, {1: "a"})

        # This should set x[2]="b" and return "b"
        v2 = dict_setdefault(x, 2, "b")
        self.assertEqual(v2, "b")
        self.assertEqual(x, {1: "a", 2: "b"})

    def test_dict_setdefault_noarg(self):
        @Entrypoint
        def dict_setdefault(d, k):
            return d.setdefault(k)

        x = Dict(int, str)()
        x[1] = "a"

        # This should not change the dictionary, and return "a"
        v1 = dict_setdefault(x, 1)
        self.assertEqual(v1, "a")
        self.assertEqual(x, {1: "a"})

        # This should set x[2]="" and return ""
        v2 = dict_setdefault(x, 2)
        self.assertEqual(v2, "")
        self.assertEqual(x, {1: "a", 2: ""})

    def test_dict_pop(self):
        @Entrypoint
        def dict_pop(d, k):
            return d.pop(k)

        @Entrypoint
        def dict_pop_2(d, k, v):
            return d.pop(k, v)

        d = Dict(int, str)()
        d[1] = "a"
        d[2] = "b"

        self.assertEqual(dict_pop(d, 1), 'a')
        self.assertEqual(dict_pop_2(d, 2, 'asdf'), 'b')
        self.assertEqual(dict_pop_2(d, 2, 'asdf'), 'asdf')

        with self.assertRaisesRegex(KeyError, "10"):
            dict_pop(d, 10)

    def test_dict_with_different_types(self):
        """Check if the dictionary with different types
        supports proper key and type conversion.
        """
        @Entrypoint
        def dict_setvalue(d, k, v):
            d[k] = v

        x = Dict(int, str)()
        x[1] = "a"
        self.assertEqual(x, {1: "a"})

        x = Dict(str, int)()
        x["a"] = 1
        self.assertEqual(x, {"a": 1})

    def test_dict_del(self):
        @Entrypoint
        def dict_delitem(d, k):
            del d[k]

        x = Dict(int, int)()
        x[1] = 2
        x[2] = 3

        dict_delitem(x, 1)

        self.assertEqual(x, {2: 3})

    @flaky(max_runs=3, min_passes=1)
    def test_dict_read_write_perf(self):
        def dict_setmany(d, count, passes):
            for _ in range(passes):
                for i in range(count):
                    if i in d:
                        d[i] += i
                    else:
                        d[i] = i

        compiled_setmany = Entrypoint(dict_setmany)

        t0 = time.time()
        aDict = Dict(int, int)()
        dict_setmany(aDict, 10000, 100)
        t1 = time.time()

        aDict2 = Dict(int, int)()
        compiled_setmany(aDict2, 1, 1)

        t2 = time.time()
        aDict2 = Dict(int, int)()
        compiled_setmany(aDict2, 10000, 100)
        t3 = time.time()

        self.assertEqual(aDict, aDict2)

        ratio = (t1 - t0) / (t3 - t2)

        # I get about 6.5
        self.assertGreater(ratio, 3)

        print("Speedup was ", ratio, ". compiled time was ", t3 - t2)

    @flaky(max_runs=3, min_passes=1)
    def test_dict_read_write_perf_releases_gil(self):
        def dict_setmany(d, count, passes):
            for _ in range(passes):
                for i in range(count):
                    if i in d:
                        d[i] += i
                    else:
                        d[i] = i

        compiled_setmany = Entrypoint(dict_setmany)

        # make sure we compile this immediately
        aDictToForceCompilation = Dict(int, int)()
        compiled_setmany(aDictToForceCompilation, 1, 1)

        # test it with one core
        t0 = time.time()
        aDict = Dict(int, int)()
        compiled_setmany(aDict, 10000, 100)
        t1 = time.time()

        # test it with 2 cores
        threads = [threading.Thread(target=compiled_setmany, args=(Dict(int, int)(), 10000, 100)) for _ in range(2)]
        t2 = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        t3 = time.time()

        slowdownRatio = (t3 - t2) / (t1 - t0)

        self.assertGreater(slowdownRatio, .9)
        self.assertLess(slowdownRatio, 1.75)

        print("Multicore slowdown factor was ", slowdownRatio)

    def test_iteration(self):
        def iterateDirect(d):
            res = ListOf(type(d).KeyType)()

            for elt in d:
                res.append(elt)

            return res

        def iterateKeys(d):
            res = ListOf(type(d).KeyType)()

            for elt in d.keys():
                res.append(elt)

            return res

        def iterateValues(d):
            res = ListOf(type(d).ValueType)()

            for elt in d.values():
                res.append(elt)

            return res

        def iterateItems(d):
            res = ListOf(Tuple(type(d).KeyType, type(d).ValueType))()

            for elt in d.items():
                res.append(elt)

            return res

        for iterate in [iterateDirect, iterateKeys, iterateValues, iterateItems]:
            iterateCompiled = Entrypoint(iterate)

            d = Dict(int, int)()

            for i in range(100):
                self.assertEqual(iterateCompiled(d), iterate(d))
                d[i] = i

            for i in range(50):
                self.assertEqual(iterateCompiled(d), iterate(d))
                del d[i * 2]

            self.assertEqual(iterateCompiled(d), iterate(d))

    def test_refcounting(self):
        TOI = TupleOf(int)
        x = Dict(TOI, TOI)()
        aTup = TOI((1, 2, 3))
        aTup2 = TOI((1, 2, 3, 4))
        aTup3 = TOI((1, 2, 3, 4, 5))

        x[aTup] = aTup2

        @Entrypoint
        def setItem(d, k, v):
            d[k] = v

        @Entrypoint
        def getItem(d, k):
            return d[k]

        self.assertEqual(getItem(x, aTup), aTup2)

        setItem(x, aTup, aTup3)

        self.assertEqual(getItem(x, aTup), aTup3)

        self.assertEqual(_types.refcount(aTup), 2)
        self.assertEqual(_types.refcount(aTup3), 2)

    @flaky(max_runs=3, min_passes=1)
    def test_dict_hash_perf_compiled(self):
        @Entrypoint
        def f(dictToLookupIn, items, passes):
            counts = ListOf(int)()
            counts.resize(len(items))

            for passIx in range(passes):
                for i in range(len(items)):
                    if items[i] in dictToLookupIn:
                        counts[i] += 1

            return counts

        aDict = Dict(str, int)()
        someStrings = ListOf(str)()

        for i in range(100):
            someStrings.append(str(i))
            aDict[someStrings[-1]] = i

        t0 = time.time()
        f(aDict, someStrings, 1)
        print(time.time() - t0, "to compile the function.")

        t0 = time.time()
        f(aDict, someStrings, 1000000)
        print(time.time() - t0, "to lookup 100mm strings")

    def test_dict_clear_compiles(self):
        T = Dict(str, str)

        @Entrypoint
        def clearIt(d):
            d.clear()

        for i in range(100):
            d = T()

            for passIx in range(3):
                for j in range(i + 1):
                    d[str(j)] = str(j)

                for j in range(i + 1):
                    assert str(j) in d

                    if j % 4 in (0, 1, 2):
                        del d[str(j)]

                clearIt(d)

            self.assertEqual(len(d), 0)

            self.assertTrue("0" not in d)

            d["1"] = "1"
            self.assertTrue("1" in d)

    def test_dict_update_compiles(self):
        T = Dict(str, str)

        @Entrypoint
        def updateCompiled(d, d2):
            d.update(d2)

        someDicts = [
            {str(i): str(i % mod) for i in range(iVals)}
            for iVals in range(10) for mod in range(2, 10)
        ]

        for d1 in someDicts:
            for d2 in someDicts:
                aDict = T(d1)
                aDict.update(d2)

                aDict2 = T(d1)
                updateCompiled(aDict2, T(d2))

                self.assertEqual(aDict, aDict2)

    def test_dict_pop_many(self):
        @Entrypoint
        def f(x: Dict(int, int)):
            keys = ListOf(int)()

            for key in x:
                keys.append(key)

            for key in keys:
                assert key in x
                x.pop(key)

        x = Dict(int, int)()

        for i in range(100):
            x[int(i)] = 0

        f(x)

        self.assertEqual(len(x), 0)

    def test_dict_up_and_down(self):
        @Entrypoint
        def f(targets):
            x = Dict(int, int)()

            for target in targets:
                for i in range(len(x), target):
                    assert i not in x
                    x[i] = i
                    assert i in x
                    assert i + 1 not in x

                for i in range(target, len(x)):
                    assert i in x
                    x.pop(i)
                    assert i not in x

                assert len(x) == target

        C = 10
        for i1 in range(C):
            for i2 in range(C):
                for i3 in range(C):
                    for i4 in range(C):
                        for i5 in range(C):
                            f(ListOf(int)([i1, i2, i3, i4, i5]))

    def test_dict_fuzz(self):
        # try adding and removing items repeatedly, in an effort to fill the table up
        @Entrypoint
        def f(actions: TupleOf(Tuple(bool, int))):
            x = Dict(int, int)()

            for thing in actions:
                if thing[0]:
                    x[thing[1]] = 1
                else:
                    x.pop(thing[1], None)

        # this sequence exposed a severe bug in the dict_wrapper. The code below should
        # find others if the bug reappears
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

    def test_dict_of_int_with_neg_one(self):
        # negative one is special because it hashes to -1. Python
        # treats a -1 as an error code (indicating there was
        # an exception). We don't do the same thing, so lets make
        # sure we handle things that hash to -1 correctly
        d = Dict(int, int)()

        @Entrypoint
        def set(d, x, y):
            d[x] = y

        @Entrypoint
        def get(d, x):
            return d[x]

        d[-1] = 2

        set(d, -1, 2)

        self.assertEqual(get(d, -1), 2)

    def test_dict_of_float_with_neg_one(self):
        # negative one is special because it hashes to -1. Python
        # treats a -1 as an error code (indicating there was
        # an exception). We don't do the same thing, so lets make
        # sure we handle things that hash to -1 correctly
        d = Dict(float, int)()

        @Entrypoint
        def set(d, x, y):
            d[x] = y

        @Entrypoint
        def get(d, x):
            return d[x]

        d[-1.0] = 2

        set(d, -1.0, 2)

        self.assertEqual(get(d, -1.0), 2)

    def test_dict_assign_and_copy(self):

        @Entrypoint
        def dict_assign_and_modify_original(d, x, y):
            d2 = d
            d[x] = 7
            del d[y]
            return d2

        @Entrypoint
        def dict_copy_and_modify_original(d, x, y):
            d2 = d.copy()
            d[x] = 7
            del d[y]
            return d2

        d = Dict(str, int)({'a': 1, 'b': 3, 'c': 5})
        self.assertEqual(dict_assign_and_modify_original(d, 'q', 'b'), {'a': 1, 'c': 5, 'q': 7})
        d = Dict(str, int)({'a': 1, 'b': 3, 'c': 5})
        self.assertEqual(dict_copy_and_modify_original(d, 'q', 'b'), {'a': 1, 'b': 3, 'c': 5})

    def test_dict_del_refcounts(self):
        T = Dict(int, ListOf(int))

        aDict = T()

        aListOf = ListOf(int)()

        aDict[1] = aListOf

        assert _types.refcount(aListOf) == 2

        del aDict[1]

        assert _types.refcount(aListOf) == 1

        aDict[1] = aListOf

        assert _types.refcount(aListOf) == 2

        @Entrypoint
        def delOne(d):
            del d[1]

        delOne(aDict)

        assert _types.refcount(aListOf) == 1

    def test_dict_aliasing(self):
        T = Dict(int, int)

        t1 = T()
        t2 = T(t1)

        t1[10] = 20

        assert len(t2) == 0

    def test_dict_aliasing_compiled(self):
        T = Dict(int, int)

        @Entrypoint
        def dup(x):
            return T(x)

        assert dup.resultTypeFor(T).typeRepresentation == T

        t1 = T()
        t2 = dup(t1)

        t1[10] = 20

        assert len(t2) == 0

    def test_dict_assign_untyped_containers(self):
        T = Dict(int, ListOf(int))

        aDict = T()

        aDict[10] = []

        @Entrypoint
        def setIt(d, x):
            d[10] = x

        setIt(aDict, [1])

        assert len(aDict[10]) == 1

    def test_dict_assign_untyped_sets(self):
        T = Dict(int, Set(int))

        aDict = T()

        aDict[10] = set()

        @Entrypoint
        def setIt(d, x):
            d[10] = x

        setIt(aDict, set([1]))

        assert len(aDict[10]) == 1

    def test_dict_compiled_equality_with_python_and_object(self):
        def f_compare(x, y):
            return x == y

        c_compare = Entrypoint(f_compare)

        cases = [
            (Dict(int, object), {1: 2}),
            (Dict(int, object), {1: (7, 8, 9)}),
            (Dict(int, object), {1: 'one'})
        ]

        for T, v in cases:
            r1 = f_compare(T(v), v)
            r2 = c_compare(T(v), v)
            self.assertEqual(r2, True)
            self.assertEqual(r1, r2)

    def test_dict_size_change_during_iteration_raises(self):
        @Entrypoint
        def checkIt():
            aDict = Dict(int, int)()
            aDict[1] = 2

            for x in aDict:
                aDict[x+1] = 2

        with self.assertRaisesRegex(RuntimeError, "dictionary size changed"):
            checkIt()

    def test_dict_collisions(self):
        aDict = Dict(int, int)()

        CT = 1000000

        for i in range(CT):
            aDict[i] = i

        for i in range(CT, CT * 10, CT // 10):
            t0 = time.time()

            i in aDict

            if time.time() - t0 > 1e-5:
                print(i, time.time() - t0)
