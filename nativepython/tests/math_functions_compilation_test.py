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

import unittest
import math
import time
import numpy

from typed_python import (
    Float32, UInt64, UInt32, UInt16, UInt8, Int32, Int16, Int8
)

from nativepython import SpecializedEntrypoint


@SpecializedEntrypoint
def compiledHash(x):
    return hash(x)


class TestMathFunctionsCompilation(unittest.TestCase):
    def test_math_functions(self):
        for funToTest in [
            lambda x: math.isinf(x),
            lambda x: math.isfinite(x),
            lambda x: math.isnan(x)
        ]:
            compiled = SpecializedEntrypoint(funToTest)

            values = []
            for val in [0, 1, -1]:
                for T in [UInt64, UInt32, UInt16, UInt8, Int32, Int16, Int8, int, float, bool]:
                    values.append(T(val))

            for val in [math.nan, math.inf]:
                for T in [float, Float32]:
                    values.append(T(val))

            for val in values:
                self.assertEqual(compiled(val), funToTest(val), val)

    def test_math_functions_perf(self):
        def checkMany(x: float, i: int):
            count = 0
            for _ in range(i):
                if math.isfinite(x):
                    count += 1
            return count

        checkManyCompiled = SpecializedEntrypoint(checkMany)

        checkManyCompiled(0.0, 1)

        count = 1000000

        ones = numpy.ones(count)

        t0 = time.time()
        checkMany(0.0, count)
        t1 = time.time()
        checkManyCompiled(0.0, count)
        t2 = time.time()
        numpy.isfinite(ones).sum()
        t3 = time.time()

        speedup = (t1 - t0) / (t2 - t1)

        # I get about 30x
        print("speedup is", speedup)

        speedupVsNumpy = (t3 - t2) / (t2 - t1)

        # I get about .9x, so we're a little slower than numpy but not much
        print("speedup vs numpy is", speedupVsNumpy)
