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

import unittest
import math
import time
import numpy

from typed_python import (
    Float32, UInt64, UInt32, UInt16, UInt8, Int32, Int16, Int8
)

from typed_python import Entrypoint


@Entrypoint
def compiledHash(x):
    return hash(x)


class TestMathFunctionsCompilation(unittest.TestCase):
    def test_entrypoint_overrides(self):
        @Entrypoint
        def f(x):
            return (type(x), int(x))

        r1 = f(UInt8(-1))
        r2 = f(-1)
        r3 = f(0)
        print(r1)
        print(r2)
        print(r3)

    def test_math_functions(self):
        for funToTest in [
            lambda x: math.isinf(x),
            lambda x: math.isfinite(x),
            lambda x: math.isnan(x)
        ]:
            compiled = Entrypoint(funToTest)

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

        checkManyCompiled = Entrypoint(checkMany)

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

    def test_math_transcendental_fns(self):
        def callcos(x):
            return math.cos(x)

        def callsin(x):
            return math.sin(x)

        def calllog(x):
            return math.log(x)

        def calllog2(x):
            return math.log2(x)

        def calllog10(x):
            return math.log10(x)

        def callexp(x):
            return math.exp(x)

        def callsqrt(x):
            return math.sqrt(x)

        def calltanh(x):
            return math.tanh(x)

        for mathFun in [callcos, callsin, calltanh, calllog, calllog2, calllog10, callexp, callsqrt]:
            compiled = Entrypoint(mathFun)

            self.assertEqual(compiled.resultTypeFor(float).typeRepresentation, float)
            self.assertEqual(compiled.resultTypeFor(Float32).typeRepresentation, Float32)

            for v in [0.5, 0.6, 1.0]:
                self.assertEqual(compiled(v), mathFun(v))
                self.assertIsInstance(compiled(v), float)

                self.assertLess(abs(float(compiled(Float32(v))) - mathFun(v)), 1e-6)
                self.assertIsInstance(compiled(Float32(v)), Float32)

    def test_math_other_float(self):
        def callfabs(x):
            return math.fabs(x)

        def callcopysign1(x):
            return math.copysign(x, type(x)(1.0))

        def callcopysign2(x):
            return math.copysign(x, type(x)(-1.0))

        for mathFun in [callfabs, callcopysign1, callcopysign2]:
            compiled = Entrypoint(mathFun)

            self.assertEqual(compiled.resultTypeFor(float).typeRepresentation, float)
            self.assertEqual(compiled.resultTypeFor(Float32).typeRepresentation, Float32)

            for v in [-1234.0, -12.34, -1.0, -0.5, 0.0, 0.5, 1.0, 12.34, 1234.0]:
                self.assertEqual(compiled(v), mathFun(v))
                self.assertIsInstance(compiled(v), float)

                self.assertLess(abs(float(compiled(Float32(v))) - mathFun(v)), 1e-6)
                self.assertIsInstance(compiled(Float32(v)), Float32)
