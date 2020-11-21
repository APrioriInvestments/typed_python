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
import time
import gc
import pytest

from typed_python import Entrypoint, ListOf
from typed_python.test_util import currentMemUsageMb


class TestGeneratorsAndComprehensions(unittest.TestCase):
    def test_list_comp(self):
        @Entrypoint
        def listComp(x):
            return [a + 1 for a in range(x)]

        lst = listComp(10)

        assert isinstance(lst, list)
        assert lst == [a + 1 for a in range(10)]

    def test_list_from_listcomp(self):
        @Entrypoint
        def listComp(x):
            return ListOf(int)([a + 1 for a in range(x)])

        lst = listComp(10)

        assert isinstance(lst, ListOf(int))
        assert lst == [a + 1 for a in range(10)]

    def test_list_from_listcomp_perf(self):
        @Entrypoint
        def listCompSumConverted(x):
            aLst = ListOf(int)([a + 1 for a in range(x)])
            res = 0
            for a in aLst:
                res += a
            return res

        @Entrypoint
        def listCompSumMasquerade(x):
            aLst = [a + 1 for a in range(x)]
            res = 0
            for a in aLst:
                res += a
            return res

        def listCompSumUntyped(x):
            aLst = [a + 1 for a in range(x)]
            res = 0
            for a in aLst:
                res += a
            return res

        listCompSumConverted(1000)
        listCompSumMasquerade(1000)

        t0 = time.time()
        listCompSumConverted(1000000)
        convertedTime = time.time() - t0

        t0 = time.time()
        listCompSumMasquerade(1000000)
        masqueradeTime = time.time() - t0

        t0 = time.time()
        listCompSumUntyped(1000000)
        untypedTime = time.time() - t0

        # they should be about the same
        assert .75 <= convertedTime / masqueradeTime <= 1.25

        # but python is much slower. I get about 30 x.
        assert untypedTime / convertedTime > 10

    def executeInLoop(self, f, duration=.25, threshold=1.0):
        gc.collect()
        memUsage = currentMemUsageMb()

        t0 = time.time()

        count = 0
        while time.time() - t0 < duration:
            f()
            count += 1

        gc.collect()
        print("count=", count, "allocated=", currentMemUsageMb() - memUsage)
        self.assertLess(currentMemUsageMb() - memUsage, threshold)

    @pytest.mark.skipif('sys.platform=="darwin"')
    def test_listcomp_doesnt_leak(self):
        @Entrypoint
        def listComp(x):
            return [a + 1 for a in range(x)]

        @Entrypoint
        def sumListComp(x):
            l = listComp(x)
            res = 0
            for val in l:
                res += val
            return res

        # burn it in
        sumListComp(1000000)
        self.executeInLoop(lambda: sumListComp(1000000), duration=.1, threshold=20.0)
        self.executeInLoop(lambda: sumListComp(1000000), duration=.25, threshold=1.0)
