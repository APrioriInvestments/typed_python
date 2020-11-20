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

from typed_python import Entrypoint
from typed_python.test_util import currentMemUsageMb


class TestGeneratorsAndComprehensions(unittest.TestCase):
    def test_list_comp(self):
        @Entrypoint
        def listComp(x):
            return [a + 1 for a in range(x)]

        lst = listComp(10)

        assert isinstance(lst, list)
        assert lst == [a + 1 for a in range(10)]

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
