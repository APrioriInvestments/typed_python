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
        def callacos(x):
            return math.acos(x)

        def callacosh(x):
            return math.acosh(x)

        def callasin(x):
            return math.asin(x)

        def callasinh(x):
            return math.asinh(x)

        def callatan(x):
            return math.atan(x)

        def callatanh(x):
            return math.atanh(x)

        def callcos(x):
            return math.cos(x)

        def callcosh(x):
            return math.cosh(x)

        def callerf(x):
            return math.erf(x)

        def callerfc(x):
            return math.erfc(x)

        def callexp(x):
            return math.exp(x)

        def callexpm1(x):
            return math.expm1(x)

        def callgamma(x):
            return math.gamma(x)

        def calllgamma(x):
            return math.lgamma(x)

        def calllog(x):
            return math.log(x)

        def calllog1p(x):
            return math.log1p(x)

        def calllog2(x):
            return math.log2(x)

        def calllog10(x):
            return math.log10(x)

        def callpow1(x):
            return math.pow(x, type(x)(2.0))

        def callpow2(x):
            return math.pow(x, type(x)(0.8))

        def callpow3(x):
            return math.pow(x, type(x)(-2.5))

        def callsin(x):
            return math.sin(x)

        def callsinh(x):
            return math.sinh(x)

        def callsqrt(x):
            return math.sqrt(x)

        def calltan(x):
            return math.tan(x)

        def calltanh(x):
            return math.tanh(x)

        # added in 3.7:
        # def callremainder(x, y):
        #     return math.remainder(x, y)

        def callpow(x, y):
            return math.pow(x, y)

        def callatan2(x, y):
            return math.atan2(x, y)

        for mathFun in [callacos, callacosh, callasin, callasinh,
                        callcos, callcosh, callsin, callsinh,
                        callatan, callatanh, calltan, calltanh,
                        callerf, callerfc, callgamma, calllgamma,
                        calllog, calllog1p, calllog2, calllog10,
                        callexp, callexpm1,
                        callpow1, callpow2, callpow3,
                        callsqrt]:
            compiled = Entrypoint(mathFun)

            self.assertEqual(compiled.resultTypeFor(float).typeRepresentation, float)
            self.assertEqual(compiled.resultTypeFor(Float32).typeRepresentation, Float32)

            for v in [-2.34, -1.0, -0.6, -0.5, 0.0, 0.5, 0.6, 1.0, 2.34]:
                raisesValueError = False
                try:
                    r1 = mathFun(v)
                except ValueError:
                    raisesValueError = True

                if raisesValueError:
                    with self.assertRaises(ValueError):
                        compiled(v)
                    with self.assertRaises(ValueError):
                        compiled(Float32(v))
                else:
                    r2 = compiled(v)
                    self.assertIsInstance(r2, float)
                    if r1 == 0.0:
                        self.assertLess(abs(r2 - r1), 1e-10, (mathFun, v, r1, r2))
                    else:
                        self.assertLess(abs((r2 - r1) / r1), 1e-10, (mathFun, v, r1, r2))

                    r3 = compiled(Float32(v))
                    self.assertIsInstance(r3, Float32, (mathFun, v))
                    if r1 == 0.0:
                        self.assertLess(abs(r3 - r1), 1e-6, (mathFun, v, r1, r3))
                    else:
                        self.assertLess(abs((r3 - r1) / r1), 1e-6, (mathFun, v, r1, r3))

        for mathFun in [callpow, callatan2]:
            compiled = Entrypoint(mathFun)
            self.assertEqual(compiled.resultTypeFor(float, float).typeRepresentation, float)
            self.assertEqual(compiled.resultTypeFor(Float32, float).typeRepresentation, float)
            self.assertEqual(compiled.resultTypeFor(float, Float32).typeRepresentation, float)
            self.assertEqual(compiled.resultTypeFor(Float32, Float32).typeRepresentation, Float32)

            for v1 in [-5.4, -3.0, -1.0, -0.6, -0.5, 0.0, 0.5, 0.6, 1.0, 3.0, 5.4]:
                for v2 in [-5.4, -3.0, -1.0, -0.6, -0.5, 0.0, 0.5, 0.6, 1.0, 3.0, 5.4]:
                    raisesValueError = False
                    try:
                        r1 = mathFun(v1, v2)
                    except ValueError:
                        raisesValueError = True

                    if raisesValueError:
                        with self.assertRaises(ValueError):
                            compiled(v1, v2)
                        with self.assertRaises(ValueError):
                            compiled(Float32(v1), Float32(v2))
                    else:
                        r2 = compiled(v1, v2)
                        self.assertEqual(r1, r2)
                        r3 = compiled(Float32(v1), Float32(v2))
                        if r1 == 0.0:
                            self.assertLess(abs(r3 - r1), 1e-6, (mathFun, v1, v2, r1, r3))
                        else:
                            self.assertLess(abs((r3 - r1)/r1), 1e-6, (mathFun, v1, v2, r1, r3))

    def test_math_other_float(self):
        def callfabs(x):
            return math.fabs(x)

        def callceil(x):
            return math.ceil(x)

        def callfloor(x):
            return math.floor(x)

        def calltrunc(x):
            return math.trunc(x)

        def callcopysign1(x):
            return math.copysign(x, type(x)(1.0))

        def callcopysign2(x):
            return math.copysign(x, type(x)(-1.0))

        def calldegrees(x):
            return math.degrees(x)

        def callradians(x):
            return math.radians(x)

        for mathFun in [callfabs, callcopysign1, callcopysign2, callceil, callfloor, calltrunc, calldegrees, callradians]:
            compiled = Entrypoint(mathFun)

            self.assertEqual(compiled.resultTypeFor(float).typeRepresentation, float)
            self.assertEqual(compiled.resultTypeFor(Float32).typeRepresentation, Float32)

            for v in [-1234.5, -12.34, -1.0, -0.5, 0.0, 0.5, 1.0, 12.34, 1234.5]:
                r1 = mathFun(v)
                r2 = compiled(v)
                self.assertIsInstance(r2, float)
                self.assertEqual(r1, r2, (mathFun, v))

                r3 = compiled(Float32(v))
                self.assertIsInstance(r3, Float32)
                if r1 == 0.0:
                    self.assertLess(abs(r3 - r1), 1e-6, (mathFun, v, r1, r3))
                else:
                    self.assertLess(abs((r3 - r1) / r1), 1e-6, (mathFun, v, r1, r3))

        def callhypot(x, y):
            return math.hypot(x, y)

        for mathFun in [callhypot]:
            compiled = Entrypoint(mathFun)

            self.assertEqual(compiled.resultTypeFor(float, float).typeRepresentation, float)
            self.assertEqual(compiled.resultTypeFor(Float32, float).typeRepresentation, float)
            self.assertEqual(compiled.resultTypeFor(float, Float32).typeRepresentation, float)
            self.assertEqual(compiled.resultTypeFor(Float32, Float32).typeRepresentation, Float32)

            for v1 in [-1234.5, -12.34, -1.0, -0.5, 0.0, 0.5, 1.0, 12.34, 1234.5]:
                for v2 in [-1234.5, -12.34, -1.0, -0.5, 0.0, 0.5, 1.0, 12.34, 1234.5]:
                    r1 = mathFun(v1, v2)
                    r2 = compiled(v1, v2)
                    self.assertIsInstance(r2, float)
                    self.assertEqual(r1, r2, (mathFun, v1, v2))

                    r3 = compiled(Float32(v1), Float32(v2))
                    self.assertIsInstance(r3, Float32)
                    if r1 == 0.0:
                        self.assertLess(abs(r3 - r1), 1e-6, (mathFun, v1, v2, r1, r3))
                    else:
                        self.assertLess(abs((r3 - r1) / r1), 1e-6, (mathFun, v1, v2, r1, r3))
