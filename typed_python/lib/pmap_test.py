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

import unittest
import os

from typed_python.lib.pmap import pmap
from typed_python import ListOf, Entrypoint
import time


def isPrime(p):
    x = 2
    while x * x <= p:
        if p % x == 0:
            return False
        x = x + 1
    return True


@Entrypoint
def isPrimeLC(x):
    res = ListOf(bool)()
    for p in x:
        res.append(isPrime(p))
    return res


class TestMap(unittest.TestCase):
    def test_pmap_correct(self):
        def addOne(x):
            return x + 1

        self.assertEqual(
            pmap(ListOf(int)([1, 2, 3]), addOne, int),
            [2, 3, 4]
        )

    def test_pmap_perf(self):
        # disable this test on travis, as extra cores aren't guaranteed.
        if os.environ.get('TRAVIS_CI', None) is not None:
            return

        someInts = ListOf(int)()
        for i in range(100000):
            someInts.append(100000000 + i)

        outInts = pmap(someInts, isPrime, bool)
        outInts = pmap(someInts, isPrime, bool)
        isPrimeLC(someInts[:10])

        t0 = time.time()
        outInts = pmap(someInts, isPrime, bool)
        t1 = time.time()
        outIntsSeq = isPrimeLC(someInts)
        t2 = time.time()

        print(t1 - t0, " to do 100k little jobs")
        print(t2 - t1, " to do it sequentially")
        speedup = (t2 - t1) / (t1 - t0)
        print(speedup, " parallelism")

        self.assertEqual(outInts, outIntsSeq)

        # I get about 4x on a decent box.
        self.assertGreater(speedup, 1.5)
