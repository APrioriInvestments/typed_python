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

from typed_python import Dict, ListOf, Tuple
import typed_python._types as _types
from nativepython import SpecializedEntrypoint
import unittest
import time
import threading

class TestDictCompilation(unittest.TestCase):
    def test_can_copy_dict(self):
        @SpecializedEntrypoint
        def f(x: Dict(int, int)):
            y = x
            return y

        self.assertEqual(f({1: 2}), {1: 2})

        @SpecializedEntrypoint
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
            self.assertEqual(aListRev, list(reversed(dicts)))
            aListRev = None

            refcounts2 = [_types.refcount(x) for x in aList]

            self.assertEqual(refcounts, refcounts2)

    def test_dict_length(self):
        @SpecializedEntrypoint
        def dict_len(x):
            return len(x)

        x = Dict(int, int)({1: 2})

        self.assertEqual(dict_len(x), 1)
        x[2] = 3

        self.assertEqual(dict_len(x), 2)

        del x[1]

        self.assertEqual(dict_len(x), 1)

    def test_dict_getitem(self):
        @SpecializedEntrypoint
        def dict_getitem(x, y):
            return x[y]

        x = Dict(int, int)()

        x[1] = 2

        self.assertEqual(dict_getitem(x, 1), 2)

        with self.assertRaisesRegex(Exception, "Key doesn't exist"):
            dict_getitem(x, 2)

    def test_dict_setitem(self):
        @SpecializedEntrypoint
        def dict_setitem(d, k, v):
            d[k] = v

        x = Dict(int, int)()

        x[1] = 2

        dict_setitem(x, 1, 3)

        self.assertEqual(x, {1: 3})

        dict_setitem(x, 2, 300)

        self.assertEqual(x, {1: 3, 2: 300})

        @SpecializedEntrypoint
        def dict_setmany(d, count):
            for i in range(count):
                d[i] = i * i

        dict_setmany(x, 1000)

        self.assertEqual(x, {i: i*i for i in range(1000)})

    def test_dict_del(self):
        @SpecializedEntrypoint
        def dict_delitem(d, k):
            del d[k]

        x = Dict(int, int)()
        x[1] = 2
        x[2] = 3

        dict_delitem(x, 1)

        self.assertEqual(x, {2: 3})

    def test_dict_read_write_perf(self):
        def dict_setmany(d, count, passes):
            for _ in range(passes):
                for i in range(count):
                    if i in d:
                        d[i] += i
                    else:
                        d[i] = i

        compiled_setmany = SpecializedEntrypoint(dict_setmany)

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

        print("Speedup was ", ratio)

    def test_dict_read_write_perf_releases_gil(self):
        def dict_setmany(d, count, passes):
            for _ in range(passes):
                for i in range(count):
                    if i in d:
                        d[i] += i
                    else:
                        d[i] = i

        compiled_setmany = SpecializedEntrypoint(dict_setmany)

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
        self.assertLess(slowdownRatio, 1.5)

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
            iterateCompiled = SpecializedEntrypoint(iterate)

            d = Dict(int, int)()

            for i in range(100):
                self.assertEqual(iterateCompiled(d), iterate(d))
                d[i] = i

            for i in range(50):
                self.assertEqual(iterateCompiled(d), iterate(d))
                del d[i * 2]

            self.assertEqual(iterateCompiled(d), iterate(d))
