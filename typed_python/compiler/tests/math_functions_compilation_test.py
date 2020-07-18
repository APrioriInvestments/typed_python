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
import math
import time
import numpy

from typed_python import (
    Float32, UInt64, UInt32, UInt16, UInt8, Int32, Int16, Int8, ListOf, TupleOf
)

from typed_python import Entrypoint


import typed_python.compiler
typeWrapper = lambda t: typed_python.compiler.python_object_representation.typedPythonTypeToTypeWrapper(t)


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
        def f_acos(x):
            return math.acos(x)

        def f_acosh(x):
            return math.acosh(x)

        def f_asin(x):
            return math.asin(x)

        def f_asinh(x):
            return math.asinh(x)

        def f_atan(x):
            return math.atan(x)

        def f_atanh(x):
            return math.atanh(x)

        def f_cos(x):
            return math.cos(x)

        def f_cosh(x):
            return math.cosh(x)

        def f_erf(x):
            return math.erf(x)

        def f_erfc(x):
            return math.erfc(x)

        def f_exp(x):
            return math.exp(x)

        def f_expm1(x):
            return math.expm1(x)

        def f_gamma(x):
            return math.gamma(x)

        def f_lgamma(x):
            return math.lgamma(x)

        def f_log(x):
            return math.log(x)

        def f_log1p(x):
            return math.log1p(x)

        def f_log2(x):
            return math.log2(x)

        def f_log10(x):
            return math.log10(x)

        def f_pow1(x):
            return math.pow(x, type(x)(2.0))

        def f_pow2(x):
            return math.pow(x, type(x)(0.8))

        def f_pow3(x):
            return math.pow(x, type(x)(-2.5))

        def f_sin(x):
            return math.sin(x)

        def f_sinh(x):
            return math.sinh(x)

        def f_sqrt(x):
            return math.sqrt(x)

        def f_tan(x):
            return math.tan(x)

        def f_tanh(x):
            return math.tanh(x)

        def f_pow(x, y):
            return math.pow(x, y)

        def f_atan2(x, y):
            return math.atan2(x, y)

        for mathFun in [f_acos, f_acosh, f_asin, f_asinh,
                        f_cos, f_cosh, f_sin, f_sinh,
                        f_atan, f_atanh, f_tan, f_tanh,
                        f_erf, f_erfc, f_gamma, f_lgamma,
                        f_log, f_log1p, f_log2, f_log10,
                        f_exp, f_expm1,
                        f_pow1, f_pow2, f_pow3,
                        f_sqrt]:
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

        for mathFun in [f_pow, f_atan2]:
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

    def test_math_other_one(self):
        def f_fabs(x):
            return math.fabs(x)

        def f_ceil(x):
            return math.ceil(x)

        def f_floor(x):
            return math.floor(x)

        def f_trunc(x):
            return math.trunc(x)

        def f_copysign1(x):
            return math.copysign(x, type(x)(1.0))

        def f_copysign2(x):
            return math.copysign(x, type(x)(-1.0))

        def f_degrees(x):
            return math.degrees(x)

        def f_radians(x):
            return math.radians(x)

        def f_factorial(x):
            return math.factorial(x)

        for mathFun in [f_fabs, f_copysign1, f_copysign2, f_ceil, f_floor, f_trunc, f_degrees, f_radians]:
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

        for mathFun in [f_factorial]:
            compiled = Entrypoint(mathFun)

            self.assertEqual(compiled.resultTypeFor(int).typeRepresentation, int)
            self.assertEqual(compiled.resultTypeFor(Int32).typeRepresentation, int)
            self.assertEqual(compiled.resultTypeFor(UInt8).typeRepresentation, int)

            for v in range(21):
                r1 = mathFun(v)
                r2 = compiled(v)
                self.assertIsInstance(r2, int)
                self.assertEqual(r1, r2, (mathFun, v))

            with self.assertRaises(ValueError):
                compiled(-1)
            with self.assertRaises(ValueError):
                compiled(-1.0)
            with self.assertRaises(ValueError):
                compiled(3.5)

            for v in range(34):
                r1 = mathFun(v)
                r2 = compiled(Float32(v))
                self.assertIsInstance(r2, Float32)
                self.assertTrue(abs(Float32(r1) - r2) / Float32(r1) < 1e-6, (mathFun, v))

            for v in range(170):
                r1 = mathFun(v)
                r2 = compiled(float(v))
                self.assertIsInstance(r2, float)
                self.assertTrue(abs(r1 - r2) / r1 < 1e-15, (mathFun, v))

    def test_math_frexp_modf(self):
        def f_frexp(x):
            return math.frexp(x)

        def f_modf(x):
            return math.modf(x)

        for mathFun in [f_frexp, f_modf]:
            compiled = Entrypoint(mathFun)

            self.assertEqual(compiled.resultTypeFor(float).typeRepresentation, typeWrapper(tuple).typeRepresentation)
            self.assertEqual(compiled.resultTypeFor(Float32).typeRepresentation, typeWrapper(tuple).typeRepresentation)
            for v in [-1e30/7, -1234.5, -12.34, -1.0, -0.5, 0.0, 0.5, 1.0, 12.34, 1234.5, 1e30/7]:
                r1 = mathFun(v)
                r2 = compiled(v)
                self.assertIsInstance(r2[0], float)
                self.assertIsInstance(r2[1], float if mathFun is f_modf else int)
                self.assertEqual(r1, r2, (mathFun, v))

                r3 = compiled(Float32(v))
                self.assertTrue(type(r3[0]) is Float32)
                # self.assertIsInstance(r2[0], Float32) fails for some reason
                self.assertTrue(type(r3[1]) is (Float32 if mathFun is f_modf else int))
                for i in range(2):
                    self.assertLess(abs((r1[i] - r3[i])/r1[i]) if r1[i] else abs(r1[i] - r3[i]), 1e-6)

    def test_math_other_two(self):
        def f_hypot(x, y):
            return math.hypot(x, y)

        def f_fmod(x, y):
            return math.fmod(x, y)

        test_values = [-1234.5, -12.34, -1.0, -0.5, 0.0, 0.5, 1.0, 12.34, 1234.5]
        for mathFun in [f_hypot, f_fmod]:
            compiled = Entrypoint(mathFun)

            self.assertEqual(compiled.resultTypeFor(float, float).typeRepresentation, float)
            self.assertEqual(compiled.resultTypeFor(Float32, float).typeRepresentation, float)
            self.assertEqual(compiled.resultTypeFor(float, Float32).typeRepresentation, float)
            self.assertEqual(compiled.resultTypeFor(Float32, Float32).typeRepresentation, Float32)

            for v1 in test_values:
                for v2 in test_values:
                    raisesValueError = False
                    try:
                        r1 = mathFun(v1, v2)
                    except ValueError:
                        raisesValueError = True

                    if raisesValueError:
                        with self.assertRaises(ValueError):
                            compiled(v1, v2)
                    else:
                        r2 = compiled(v1, v2)
                        self.assertIsInstance(r2, float)
                        self.assertEqual(r1, r2, (mathFun, v1, v2))

                    if raisesValueError:
                        with self.assertRaises(ValueError):
                            compiled(Float32(v1), Float32(v2))
                    else:
                        r3 = compiled(Float32(v1), Float32(v2))
                        self.assertIsInstance(r3, Float32)
                        if r1 == 0.0:
                            self.assertLess(abs(r3 - r1), 1e-6, (mathFun, v1, v2, r1, r3))
                        else:
                            self.assertLess(abs((r3 - r1) / r1), 5e-5, (mathFun, v1, v2, r1, r3))

        def f_isclose(x, y, r, a):
            if r is None and a is None:
                return math.isclose(x, y)
            if r is None:
                return math.isclose(x, y, abs_tol=a)
            if a is None:
                return math.isclose(x, y, rel_tol=r)
            return math.isclose(x, y, rel_tol=r, abs_tol=a)

        for mathFun in [f_isclose]:
            compiled = Entrypoint(mathFun)
            for v1 in test_values:
                for v2 in [v1,
                           v1 + 1, v1 - 1,
                           v1 + 1.1e-5, v1 - 1.1e-5,
                           v1 + 9e-6, v1 - 9e-6,
                           v1 + 1.1e-9, v1 - 1.1e-9,
                           v1 + 9e-10, v1 - 9e-10]:
                    for rel_tol in [None, 1, 1e-3, 1e-5, 1e-9]:
                        for abs_tol in [None, 0, 1e-7, 1e-5, 1e-3, 1]:
                            r1 = mathFun(v1, v2, rel_tol, abs_tol)
                            r2 = compiled(v1, v2, rel_tol, abs_tol)
                            self.assertIsInstance(r2, bool)
                            self.assertEqual(r1, r2, (mathFun, v1, v2, rel_tol, abs_tol))

                            # don't bother testing Float32 if rel_tol is too small
                            if rel_tol is not None and rel_tol < 1e-5:
                                continue

                            # default rel_tol is different for Float32
                            r3 = mathFun(v1, v2, 1e-5 if rel_tol is None else rel_tol, abs_tol)
                            r4 = compiled(Float32(v1), Float32(v2), rel_tol, abs_tol)
                            self.assertIsInstance(r4, bool)
                            self.assertEqual(r3, r4, (mathFun, v1, v2, rel_tol, abs_tol))

        def f_gcd(x, y):
            return math.gcd(x, y)

        for mathFun in [f_gcd]:
            compiled = Entrypoint(mathFun)
            self.assertEqual(compiled.resultTypeFor(int, int).typeRepresentation, UInt64)
            self.assertEqual(compiled.resultTypeFor(Int32, Int32).typeRepresentation, UInt64)
            self.assertEqual(compiled.resultTypeFor(UInt32, Int16).typeRepresentation, UInt64)
            self.assertEqual(compiled.resultTypeFor(Int8, UInt64).typeRepresentation, UInt64)
            self.assertEqual(compiled.resultTypeFor(UInt64, UInt64).typeRepresentation, UInt64)

            test_values = [-2**63+1, -35, 0, 1, 2, 3, 12, 15, 81, 90, 360, 1000, 1080, 2**16 * 3 * 5 * 7, 2**63-1]
            for v1 in test_values:
                for v2 in test_values:
                    raisesTypeError = False
                    try:
                        r1 = mathFun(v1, v2)
                    except TypeError:
                        raisesTypeError = True
                    if raisesTypeError:
                        with self.assertRaises(TypeError):
                            compiled(v1, v2)
                    else:
                        r2 = compiled(v1, v2)
                        self.assertEqual(r1, r2, (mathFun, v1, v2))

        def f_ldexp(x, y):
            return math.ldexp(x, y)

        for mathFun in [f_ldexp]:
            compiled = Entrypoint(mathFun)
            self.assertEqual(compiled.resultTypeFor(float, int).typeRepresentation, float)
            self.assertEqual(compiled.resultTypeFor(Float32, int).typeRepresentation, Float32)
            self.assertEqual(compiled.resultTypeFor(float, Int32).typeRepresentation, float)
            self.assertEqual(compiled.resultTypeFor(Float32, Int32).typeRepresentation, Float32)

            for v1 in [-123.45, -2.0, -1.0, -0.999, -0.7, -0.5, 0.0, 0.5, 0.7, 0.999, 1.0, 2.0, 123.45]:
                for v2 in range(-10, 10):
                    r1 = mathFun(v1, v2)
                    r2 = compiled(v1, v2)

                    self.assertEqual(r1, r2, (mathFun, v1, v2))

                    r3 = Float32(r1)
                    r4 = compiled(Float32(v1), Float32(v2))
                    self.assertEqual(r3, r4, (mathFun, v1, v2))

    def test_math_fsum(self):
        def f_fsum(iterable):
            return math.fsum(iterable)

        compiled = Entrypoint(f_fsum)

        test_cases = [
            [1, 1e100, 1, -1e100],
            ([1, 1e100] * 10 + [1, -1e100] * 10) * 10,
            range(100),
            range(-1000, 1000, 5),
            ListOf(float)([1.1, 2.2, 3.3]),
            TupleOf(float)([1.1, 2.2, 3.3])
        ]
        for v in test_cases:
            r1 = math.fsum(v)
            r2 = compiled(v)
            self.assertEqual(r1, r2)

        with self.assertRaises(ValueError):
            compiled([0.1, 0.2, 0.3, "abc"])

        with self.assertRaises(TypeError):
            compiled(1234)  # not iterable

    def test_math_constants(self):
        def all_constants(x):
            return (type(x), math.pi, math.e, math.tau, math.inf, math.nan)

        compiled = Entrypoint(all_constants)
        r1 = all_constants(1.0)
        r2 = compiled(1.0)
        self.assertTrue(math.isnan(r1[5]))
        self.assertTrue(math.isnan(r2[5]))
        # Note r1 != r2 because nan != nan
        self.assertEqual(r1[0:-1], r2[0:-1])

    def test_math_functions_perf_other(self):
        count = 1000000
        element = lambda i: -5.0 + i / (count / 10.0)
        domain = [element(i) for i in range(count)]
        a = numpy.fromiter(domain, numpy.float64)

        # intrinsic llvm
        def many_sin(n: int):
            sum = 0.0
            for i in range(n):
                sum += math.sin(element(i))
            return sum

        # intrinsic llvm
        def many_exp(n: int):
            sum = 0.0
            for i in range(n):
                sum += math.exp(element(i))
            return sum

        # C++
        def many_cosh(n: int):
            sum = 0.0
            for i in range(n):
                sum += math.cosh(element(i))
            return sum

        # native operations
        def many_hypot(n: int):
            sum = 0.0
            for i in range(n):
                sum += math.hypot(element(i), 2.0)
            return sum

        for f, numpy_f in [(many_sin, numpy.sin), (many_exp, numpy.exp), (many_cosh, numpy.cosh), (many_hypot, numpy.hypot)]:
            compiled = Entrypoint(f)
            compiled(1)

            t0 = time.time()
            r0 = f(count)
            t1 = time.time()
            r1 = compiled(count)
            t2 = time.time()
            if numpy_f is numpy.hypot:
                r2 = numpy_f(a, 2.0).sum()
            else:
                r2 = numpy_f(a).sum()
            t3 = time.time()

            if r0 != r1:
                pass
            if r0 != r2:
                pass

            speedup = (t1 - t0) / (t2 - t1)

            print(f"{f.__name__} speedup is", speedup)

            speedupVsNumpy = (t3 - t2) / (t2 - t1)

            print(f"{f.__name__} speedup vs numpy is", speedupVsNumpy)

    def test_math_float_overload_order(self):
        @Entrypoint
        def f(x):
            return type(x)(x + 1e-8)

        @Entrypoint
        def g(x):
            return type(x)(x + 1e-8)

        r1 = f(1.0)
        r2 = f(Float32(1.0))
        print(r1, r2)

        r3 = g(Float32(1.0))
        r4 = g(1.0)
        print(r3, r4)

        self.assertEqual(r1, r4)
        self.assertEqual(r2, r3)

    def test_math_int_overload_order(self):
        for T1 in [UInt8, UInt16, UInt32, UInt64, Int8, Int16, Int32, int]:
            for T2 in [UInt8, UInt16, UInt32, UInt64, Int8, Int16, Int32, int]:
                @Entrypoint
                def f(x):
                    return type(x)(x)

                @Entrypoint
                def g(x):
                    return type(x)(x)

                if T1 == T2:
                    continue

                r1 = f(T1(0))
                r2 = f(T2(0))
                print(r1, r2)

                r3 = g(T2(0))
                r4 = g(T1(0))
                print(r3, r4)

                self.assertEqual(type(r1), type(r4))
                self.assertEqual(type(r2), type(r3))

                del f
                del g
