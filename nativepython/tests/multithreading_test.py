#   Copyright 2019 Nativepython Authors
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
import threading
import os


def thread_apply(f, argtuples):
    threads = []
    results = {}

    def doit(f, ix, *args):
        results[ix] = f(*args)

    for ix, a in enumerate(argtuples):
        threads.append(threading.Thread(target=doit, args=(f, ix) + a))

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    return [results.get(i) for i in range(len(argtuples))]


def Compiled(f):
    f = Function(f)
    return Runtime.singleton().compile(f)


class AClass(Class):
    x = Member(int)


class TestMultithreading(unittest.TestCase):
    def test_gil_is_released(self):
        @Compiled
        def f(x: int):
            res = 0.0
            for i in range(x):
                res += i
            return res

        ratios = []
        for _ in range(10):
            t0 = time.time()
            res1 = thread_apply(f, [(100000000,)])
            t1 = time.time()
            res2 = thread_apply(f, [(100000000,), (100000001,)])
            t2 = time.time()

            first = t1 - t0
            second = t2 - t1

            ratios.append(second/first)

        ratios = sorted(ratios)

        ratio = ratios[5]

        #expect the ratio to be close to 1, but have some error margin, especially on Travis
        #where we may be running in a multitenant environment
        if os.environ.get('TRAVIS_CI', None):
            self.assertTrue(ratio >= .8 and ratio < 1.75, ratio)
        else:
            self.assertTrue(ratio >= .9 and ratio < 1.1, ratio)

    def test_refcounts_of_objects_across_boundary(self):
        class Object:
            pass
        anObject = Object()

        A = Alternative("A", X={'x': int}, Y={'y': int})

        for instance in [
                TupleOf(int)((1, 2, 3)),
                ListOf(int)((1, 2, 3)),
                #Dict(int,int)({1:2,3:4}),
                ConstDict(int, int)({1: 2, 3: 4}),
                AClass(),
                #anObject,
                A.X(x=10)
                ]:
            self.refcountsTest(instance)

    def refcountsTest(self, instance):
        typeOfInstance = type(instance)

        @Compiled
        def rapidlyIncAndDecref(x: typeOfInstance):
            y = x
            for _ in range(1000000):
                y = x
            return x

        thread_apply(rapidlyIncAndDecref, [(instance,)] * 10)

        self.assertEqual(_types.refcount(instance), 1)

    def test_serialize_is_parallel(self):
        x = ListOf(int)()
        x.resize(1000000)
        sc = SerializationContext({})

        def f():
            for i in range(10):
                sc.deserialize(sc.serialize(x))

        ratios = []
        for i in range(10):
            t0 = time.time()
            thread_apply(f, [()])
            t1 = time.time()
            thread_apply(f, [(), ()])
            t2 = time.time()

            first = t1 - t0
            second = t2 - t1

            ratios.append(second/first)

        ratios = sorted(ratios)

        ratio = ratios[5]

        #expect the ratio to be close to 1, but have some error margin, especially on Travis
        #where we don't really get two cores
        if os.environ.get('TRAVIS_CI', None):
            self.assertTrue(ratio >= .8 and ratio < 1.75, ratios)
        else:
            self.assertTrue(ratio >= .8 and ratio < 1.2, ratios)
