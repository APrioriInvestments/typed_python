#   Copyright 2019 typed_python Authors
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
import os
import threading
import time
import unittest

from flaky import flaky
from typed_python import (
    Class, Member, Alternative, TupleOf, ListOf, ConstDict, SerializationContext,
    Entrypoint, Compiled, localVariableTypesKnownToCompiler
)

import typed_python._types as _types


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


class AClass(Class):
    x = Member(int)


class TestMultithreading(unittest.TestCase):
    @flaky(max_runs=3, min_passes=1)
    def test_gil_is_released(self):
        @Compiled
        def f(x: int):
            res = 0.0
            for i in range(x):
                res += i
            return res

        ratios = []
        for _1 in range(10):
            t0 = time.time()
            thread_apply(f, [(100000000,)])
            t1 = time.time()
            thread_apply(f, [(100000000,), (100000001,)])
            t2 = time.time()

            first = t1 - t0
            second = t2 - t1

            ratios.append(second/first)

        ratios = sorted(ratios)

        ratio = ratios[5]

        # expect the ratio to be close to 1, but have some error margin, especially on Travis
        # where we may be running in a multitenant environment
        if os.environ.get('TRAVIS_CI', None):
            self.assertTrue(ratio >= .7 and ratio < 1.75, ratio)
        else:
            self.assertTrue(ratio >= .9 and ratio < 1.1, ratio)

    def test_refcounts_of_objects_across_boundary(self):
        class Object:
            pass
        _ = Object()

        A = Alternative("A", X={'x': int}, Y={'y': int})

        for instance in [
                TupleOf(int)((1, 2, 3)),
                ListOf(int)((1, 2, 3)),
                # Dict(int,int)({1:2,3:4}),
                ConstDict(int, int)({1: 2, 3: 4}),
                AClass(),
                # anObject,
                A.X(x=10)
        ]:
            self.refcountsTest(instance)

    def refcountsTest(self, instance):
        typeOfInstance = type(instance)

        @Compiled
        def rapidlyIncAndDecref(x: typeOfInstance):
            _ = x
            for _1 in range(1000000):
                _ = x
            return x

        thread_apply(rapidlyIncAndDecref, [(instance,)] * 10)

        self.assertEqual(_types.refcount(instance), 1)

    def test_serialize_is_parallel(self):
        if os.environ.get('TRAVIS_CI', None):
            return

        T = ListOf(int)
        x = T()
        x.resize(1000000)
        sc = SerializationContext().withoutCompression()

        def f():
            for i in range(10):
                sc.deserialize(sc.serialize(x, T), T)

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

        # expect the ratio to be close to 1, but have some error margin
        self.assertTrue(ratio >= .8 and ratio < 1.2, ratios)

    def test_can_access_locks_in_compiler_with_locks_as_obj(self):
        lock = threading.Lock()
        recursiveLock = threading.RLock()

        @Compiled
        def lockFun(l: object):
            with l:
                return 10

        # these methods will hit 'l' using the interpreter
        self.assertEqual(lockFun(lock), 10)
        self.assertEqual(lockFun(recursiveLock), 10)

        self.assertFalse(lock.locked())

    def test_can_access_locks_in_compiler_with_typed_locks(self):
        lock = threading.Lock()
        recursiveLock = threading.RLock()

        @Compiled
        def lockFun(l: threading.Lock):
            with l:
                return 10

        @Compiled
        def recursiveLockFun(l: threading.RLock):
            with l:
                return 10

        # these methods will hit the lock objects directly, bypassing
        # the interpreter and using C code.
        self.assertEqual(lockFun(lock), 10)
        self.assertEqual(recursiveLockFun(recursiveLock), 10)

        self.assertFalse(lock.locked())

    @flaky(max_runs=3, min_passes=1)
    def test_lock_perf(self):
        lock = threading.Lock()
        recursiveLock = threading.RLock()

        aList = ListOf(int)([0])

        @Compiled
        def lockFun(l: threading.Lock, aList: ListOf(int), count: int):
            print("I KNOW THESE AS ", localVariableTypesKnownToCompiler())
            for _ in range(count):
                with l:
                    aList[0] += 1

        @Compiled
        def recursiveLockFun(l: threading.RLock, aList: ListOf(int), count: int):
            for _ in range(count):
                with l:
                    aList[0] += 1

        t0 = time.time()
        lockFun(lock, aList, 1000000)
        t1 = time.time()
        recursiveLockFun(recursiveLock, aList, 1000000)
        t2 = time.time()

        # I get around 0.02 for this, which is 50mm locks / second when there is no
        # contention.
        self.assertLess(t1 - t0, .1)
        self.assertLess(t2 - t1, .1)

    def test_lock_works(self):
        lock = threading.Lock()

        aList = ListOf(int)([0])

        @Entrypoint
        def loopWithLock(l, aList, count):
            for _ in range(count):
                with l:
                    aList[0] += 1

        ct = 1000000
        threads = [threading.Thread(target=loopWithLock, args=(lock, aList, ct)) for _ in range(4)]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # this test will fail (badly) if you remove the lock because we won't add up
        # to the right amount.
        self.assertEqual(ct * 4, aList[0])

    @flaky(max_runs=3, min_passes=1)
    def test_lock_with_separate_locks_perf(self):
        @Entrypoint
        def loopWithLock(l, aList, count):
            for _ in range(count):
                with l:
                    aList[0] += 1

        def timeFor(threadCount, ct):
            t0 = time.time()

            threads = [threading.Thread(target=loopWithLock, args=(threading.Lock(), ListOf(int)([0]), ct)) for _ in range(threadCount)]

            for t in threads:
                t.start()

            for t in threads:
                t.join()

            return time.time() - t0

        # prime the compiler
        timeFor(1, 10)

        oneThread = timeFor(1, 1000000)
        twoThreads = timeFor(2, 1000000)

        # we should see that we don't really change performance, because our test is using separate
        # locks for each of the two threads in the second case. I get almost exactly 1.0 for this,
        # but on Travis, because we don't get a dedicated box, we can get more than 1. If the lock
        # is held as 'object', you'd see 2.0 or higher, so this still verifies that we are
        # getting c-level parallelism at this threshold.
        self.assertLess(twoThreads / oneThread, 1.65, (oneThread, twoThreads))
