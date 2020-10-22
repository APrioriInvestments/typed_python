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

import typed_python.compiler
from typed_python.compiler.type_wrappers.wrapper import Wrapper
import typed_python.compiler.native_ast as native_ast
from typed_python import Float32, Tuple, ListOf
from typed_python.compiler.conversion_level import ConversionLevel
import typed_python.compiler.type_wrappers.runtime_functions as runtime_functions
from typed_python.compiler.type_wrappers.tuple_wrapper import MasqueradingTupleWrapper

from math import (
    acos,
    acosh,
    asin,
    asinh,
    atan,
    atan2,
    atanh,
    ceil,
    copysign,
    cos,
    cosh,
    degrees,
    # e,
    erf,
    erfc,
    exp,
    expm1,
    fabs,
    factorial,
    floor,
    fmod,
    frexp,
    fsum,
    gamma,
    gcd,
    hypot,
    # inf,
    isclose,
    isfinite,
    isinf,
    isnan,
    ldexp,
    lgamma,
    log,
    log10,
    log1p,
    log2,
    modf,
    # nan,
    pi,
    pow,
    radians,
    sin,
    sinh,
    sqrt,
    tan,
    tanh,
    # tau,
    trunc
    # added in 3.7:
    # remainder
    # added in 3.8:
    # comb
    # dist
    # isqrt
    # perm
    # prod
)


def sumIterable(iterable):
    """Full precision summation using multiple floats for intermediate values.
    Code modified from msum() at code.activestate.com/recipes/393090/, which
    is licensed under the PSF License.
    Original comment below.
    """
    # Rounded x+y stored in hi with the round-off stored in lo.  Together
    # hi+lo are exactly equal to x+y.  The inner loop applies hi/lo summation
    # to each partial so that the list of partial sums remains exact.
    # Depends on IEEE-754 arithmetic guarantees.  See proof of correctness at:
    # www-2.cs.cmu.edu/afs/cs/project/quake/public/papers/robust-arithmetic.ps

    partials = ListOf(float)()
    for item in iterable:
        # float(item) would convert str to float, so __float__ is better
        if not isinstance(item, float):
            try:
                item = item.__float__()  # does not convert str to float
            except AttributeError:
                raise TypeError('must be real number, not str')
        x = float(item)  # I know item is a float now, but force compiler to know this too

        i = 0
        for y in partials:
            if abs(x) < abs(y):
                t = x
                x = y
                y = t
            hi = x + y
            lo = y - (hi - x)
            if lo:
                partials[i] = lo
                i += 1
            x = hi
        popindex = len(partials) - 1
        while popindex >= i:
            partials.pop()
            popindex -= 1
        partials.append(x)
    ret = 0.0
    for p in partials:
        ret += p
    return ret


class MathFunctionWrapper(Wrapper):
    is_pod = True
    is_empty = False
    is_pass_by_ref = False

    SUPPORTED_FUNCTIONS = (acos, acosh, asin, asinh, atan, atan2, atanh,
                           copysign, cos, cosh, degrees, erf, erfc, exp, expm1, fabs, factorial,
                           fmod, frexp, fsum, hypot, gamma, gcd, isclose, isnan, isfinite, isinf, ldexp, lgamma,
                           log, log2, log10, log1p, modf, pow, radians,
                           sin, sinh, sqrt, tan, tanh)

    def __init__(self, mathFun):
        assert mathFun in self.SUPPORTED_FUNCTIONS
        super().__init__(mathFun)

    def getNativeLayoutType(self):
        return native_ast.Type.Void()

    @Wrapper.unwrapOneOfAndValue
    def convert_call(self, context, expr, args, kwargs):
        # map py function to (c++ function, return type, check_inf)
        # check_inf is set when we need to check for infinite return values and possibly raise an exception
        # This would be needed when the function is compiled to an llvm intrinsic.
        fns_1 = {
            acos: (runtime_functions.acos64, float, False),
            acosh: (runtime_functions.acosh64, float, False),
            asin: (runtime_functions.asin64, float, False),
            asinh: (runtime_functions.asinh64, float, False),
            atan: (runtime_functions.atan64, float, False),
            atanh: (runtime_functions.atanh64, float, False),
            ceil: (runtime_functions.ceil64, float, False),
            cos: (runtime_functions.cos64, float, False),
            cosh: (runtime_functions.cosh64, float, False),
            degrees: (None, float, False),
            erf: (runtime_functions.erf64, float, False),
            erfc: (runtime_functions.erfc64, float, False),
            exp: (runtime_functions.exp64, float, True),
            expm1: (runtime_functions.expm1_64, float, False),
            fabs: (runtime_functions.fabs64, float, False),
            factorial: (runtime_functions.factorial64, float, False),
            floor: (runtime_functions.floor64, float, False),
            frexp: (runtime_functions.frexp64, None, False),
            gamma: (runtime_functions.gamma64, float, False),
            isfinite: (runtime_functions.isfinite_float64, bool, False),
            isinf: (runtime_functions.isinf_float64, bool, False),
            isnan: (runtime_functions.isnan_float64, bool, False),
            lgamma: (runtime_functions.lgamma64, float, False),
            log: (runtime_functions.log64, float, False),
            log1p: (runtime_functions.log1p64, float, False),
            log2: (runtime_functions.log2_64, float, False),
            log10: (runtime_functions.log10_64, float, False),
            modf: (runtime_functions.modf64, float, False),
            radians: (None, float, False),
            sin: (runtime_functions.sin64, float, False),
            sinh: (runtime_functions.sinh64, float, True),
            sqrt: (runtime_functions.sqrt64, float, False),
            tan: (runtime_functions.tan64, float, False),
            tanh: (runtime_functions.tanh64, float, False),
        }
        fns_2 = {
            atan2: (runtime_functions.atan2_64, float, False),
            copysign: (runtime_functions.copysign64, float, False),
            fmod: (runtime_functions.fmod64, float, False),
            hypot: (None, float, False),
            isclose: (runtime_functions.isclose64, bool, False),
            pow: (runtime_functions.pow64, float, True),
            trunc: (runtime_functions.trunc64, float, False),
        }
        if len(args) == 2 and not kwargs and self.typeRepresentation is ldexp:
            arg1 = args[0]
            arg2 = args[1]
            arg1 = arg1.toFloatAs()
            if arg1 is None:
                return None
            arg2 = arg2.convert_to_type(int, ConversionLevel.Signature)  # want a real 'int' here
            if arg2 is None:
                return context.pushException(TypeError, "Expected an int as second argument to ldexp.")
            return context.pushPod(float, runtime_functions.ldexp64.call(arg1.nonref_expr, arg2.nonref_expr))

        if len(args) == 2 and not kwargs and self.typeRepresentation is gcd:
            arg1 = args[0]
            arg2 = args[1]

            # gcd uses __index__, not __int__, to process integer arguments
            arg1 = arg1.toIndex()
            if arg1 is None:
                return None
            arg2 = arg2.toIndex()
            if arg2 is None:
                return None

            return context.pushPod(int, runtime_functions.gcd.call(arg1.nonref_expr, arg2.nonref_expr))

        if len(args) == 2 and self.typeRepresentation in fns_2 and (not kwargs or self.typeRepresentation is isclose):
            func, outT, check_inf = fns_2[self.typeRepresentation]

            arg1 = args[0]
            arg2 = args[1]
            argType1 = arg1.expr_type.typeRepresentation
            argType2 = arg2.expr_type.typeRepresentation
            if argType1 is not float:
                arg1 = arg1.toFloatAs()
                if arg1 is None:
                    return None
            if argType2 is not float:
                arg2 = arg2.toFloatAs()
                if arg2 is None:
                    return None

            if self.typeRepresentation is fmod:
                with context.ifelse(arg2.nonref_expr.eq(0.0)) as (ifTrue, ifFalse):
                    with ifTrue:
                        context.pushException(ValueError, "math domain error")
            elif self.typeRepresentation is hypot:
                # calculate this one with sqrt
                func = runtime_functions.sqrt64
                return context.pushPod(outT, func.call(arg1.nonref_expr.mul(arg1.nonref_expr).add(arg2.nonref_expr.mul(arg2.nonref_expr))))
            elif self.typeRepresentation is isclose:
                func = runtime_functions.isclose64

                if "rel_tol" in kwargs:
                    rel_tol = kwargs["rel_tol"].convert_to_type(float, ConversionLevel.Implicit)
                    if rel_tol is None:
                        return None
                else:
                    if args[0].expr_type.typeRepresentation is Float32 and args[1].expr_type.typeRepresentation is Float32:
                        rel_tol = native_ast.const_float_expr(1e-5)  # with Float32, default needs to be larger
                    else:
                        rel_tol = native_ast.const_float_expr(1e-9)

                if "abs_tol" in kwargs:
                    abs_tol = kwargs["abs_tol"].convert_to_type(float, ConversionLevel.Implicit)
                    if abs_tol is None:
                        return None
                else:
                    abs_tol = native_ast.const_float_expr(0.0)

                return context.pushPod(bool, func.call(arg1.nonref_expr, arg2.nonref_expr, rel_tol, abs_tol))
            elif self.typeRepresentation is pow:
                with context.ifelse(arg1.nonref_expr.lte(0.0)) as (ifTrue, ifFalse):
                    with ifTrue:  # arg1 <= 0
                        with context.ifelse(arg1.nonref_expr.eq(0.0)) as (ifTrue2, ifFalse2):
                            with ifTrue2:  # arg1 == 0
                                with context.ifelse(arg2.nonref_expr.lt(0.0)) as (ifTrue3, ifFalse3):
                                    with ifTrue3:  # arg1 == 0, arg2 < 0
                                        context.pushException(ValueError, "math domain error")
                            with ifFalse2:  # arg1 < 0
                                with context.ifelse(runtime_functions.isinf_float64.call(arg1.nonref_expr)) as (ifTrue4, ifFalse4):
                                    with ifFalse4:  # arg1 < 0 and arg1 is finite
                                        f_floor = runtime_functions.floor64
                                        with context.ifelse(
                                                f_floor.call(arg2.nonref_expr).sub(arg2.nonref_expr).neq(0.0)
                                        ) as (ifTrue5, ifFalse5):
                                            with ifTrue5:  # arg1 < 0, arg1 is finite, arg2 not an integer
                                                context.pushException(ValueError, "math domain error")

            ret = context.pushPod(outT, func.call(arg1.nonref_expr, arg2.nonref_expr))
            if check_inf:
                f_isinf = runtime_functions.isinf_float64
                with context.ifelse(f_isinf.call(ret.nonref_expr)) as (ifTrue, ifFalse):
                    with ifTrue:
                        f_isfinite = runtime_functions.isfinite_float64
                        with context.ifelse(
                                f_isfinite.call(arg1.nonref_expr).bitand(f_isfinite.call(arg2.nonref_expr))
                        ) as (ifTrue2, ifFalse2):
                            with ifTrue2:
                                context.pushException(OverflowError, 'math range error')
            return ret

        # handle integer factorial here
        if len(args) == 1 and not kwargs and self.typeRepresentation is factorial:
            arg = args[0]
            if not arg.expr_type.is_arithmetic:
                return context.pushException(TypeError, f"must be real number, not {arg.expr_type}")

            argType = arg.expr_type.typeRepresentation
            if argType not in (Float32, float):
                if argType not in (int,):
                    arg = arg.convert_to_type(int, ConversionLevel.New)
                    if arg is None:
                        return None
                with context.ifelse(arg.nonref_expr.lt(0)) as (ifTrue, ifFalse):
                    with ifTrue:
                        context.pushException(ValueError, "factorial() not defined for negative values")
                return context.pushPod(int, runtime_functions.factorial.call(arg.nonref_expr))
            # let Float32 and float args fall through to later section, which will handle float factorials

        if len(args) == 1 and not kwargs and self.typeRepresentation is fsum:
            arg = args[0]
            return context.call_py_function(sumIterable, (arg,), {})

        # math functions with 1 argument
        if len(args) == 1 and self.typeRepresentation in fns_1 and not kwargs:
            func, outT, check_inf = fns_1[self.typeRepresentation]

            arg = args[0].toFloatAs()
            if arg is None:
                return None

            # check argument
            if self.typeRepresentation in (cos, sin, tan):
                with context.ifelse(runtime_functions.isinf_float64.call(arg.nonref_expr)) as (ifTrue, ifFalse):
                    with ifTrue:
                        context.pushException(ValueError, "math domain error")
            elif self.typeRepresentation is acos:
                with context.ifelse(arg.nonref_expr.gt(1.0)) as (ifTrue, ifFalse):
                    with ifTrue:
                        context.pushException(ValueError, "math domain error")
                with context.ifelse(arg.nonref_expr.lt(-1.0)) as (ifTrue, ifFalse):
                    with ifTrue:
                        context.pushException(ValueError, "math domain error")
            elif self.typeRepresentation is acosh:
                with context.ifelse(arg.nonref_expr.lt(1.0)) as (ifTrue, ifFalse):
                    with ifTrue:
                        context.pushException(ValueError, "math domain error")
            elif self.typeRepresentation is asin:
                with context.ifelse(arg.nonref_expr.gt(1.0)) as (ifTrue, ifFalse):
                    with ifTrue:
                        context.pushException(ValueError, "math domain error")
                with context.ifelse(arg.nonref_expr.lt(-1.0)) as (ifTrue, ifFalse):
                    with ifTrue:
                        context.pushException(ValueError, "math domain error")
            elif self.typeRepresentation is atanh:
                with context.ifelse(arg.nonref_expr.gte(1.0)) as (ifTrue, ifFalse):
                    with ifTrue:
                        context.pushException(ValueError, "math domain error")
                with context.ifelse(arg.nonref_expr.lte(-1.0)) as (ifTrue, ifFalse):
                    with ifTrue:
                        context.pushException(ValueError, "math domain error")
            elif self.typeRepresentation is degrees:
                # calculate this one without c++
                return context.pushPod(outT, arg.nonref_expr.mul(native_ast.const_float_expr(180 / pi)))
            elif self.typeRepresentation is factorial:
                with context.ifelse(arg.nonref_expr.gte(0.0)) as (ifTrue, ifFalse):
                    with ifTrue:  # arg >= 0
                        f_floor = runtime_functions.floor64
                        with context.ifelse(f_floor.call(arg.nonref_expr).sub(arg.nonref_expr).neq(0.0)) as (ifTrue2, ifFalse2):
                            with ifTrue2:  # arg >= 0, arg not an integer
                                context.pushException(ValueError, "factorial() only accepts integral values")
                    with ifFalse:  # arg < 0
                        context.pushException(ValueError, "factorial() not defined for negative values")
            elif self.typeRepresentation is frexp:
                outT = MasqueradingTupleWrapper(Tuple(float, int))
                return_tuple = native_ast.Expression.StackSlot(name=".return_tuple", type=outT.layoutType)
                context.pushEffect(func.call(arg.nonref_expr, return_tuple.elemPtr(0).cast(native_ast.VoidPtr)))
                return typed_python.compiler.typed_expression.TypedExpression(context, return_tuple, outT, True)
            elif self.typeRepresentation is gamma:
                with context.ifelse(arg.nonref_expr.lte(0.0)) as (ifTrue, ifFalse):
                    with ifTrue:  # arg <= 0
                        with context.ifelse(runtime_functions.isinf_float64.call(arg.nonref_expr)) as (ifTrue2, ifFalse2):
                            with ifTrue2:  # arg == -inf   Note: inf is ok but not -inf
                                context.pushException(ValueError, "math domain error")
                        f_floor = runtime_functions.floor64
                        with context.ifelse(f_floor.call(arg.nonref_expr).sub(arg.nonref_expr).eq(0.0)) as (ifTrue2, ifFalse2):
                            with ifTrue2:  # arg <= 0, arg an integer
                                context.pushException(ValueError, "math domain error")
            elif self.typeRepresentation is lgamma:
                with context.ifelse(arg.nonref_expr.lte(0.0)) as (ifTrue, ifFalse):
                    with ifTrue:  # arg <= 0
                        f_floor = runtime_functions.floor64
                        with context.ifelse(f_floor.call(arg.nonref_expr).sub(arg.nonref_expr).eq(0.0)) as (ifTrue2, ifFalse2):
                            with ifTrue2:  # arg <= 0, arg an integer
                                context.pushException(ValueError, "math domain error")
            elif self.typeRepresentation is log:
                with context.ifelse(arg.nonref_expr.lte(0.0)) as (ifTrue, ifFalse):
                    with ifTrue:  # arg <= 0
                        context.pushException(ValueError, "math domain error")
            elif self.typeRepresentation is log1p:
                with context.ifelse(arg.nonref_expr.lte(-1.0)) as (ifTrue, ifFalse):
                    with ifTrue:  # arg <= -1
                        context.pushException(ValueError, "math domain error")
            elif self.typeRepresentation is log2:
                with context.ifelse(arg.nonref_expr.lte(0.0)) as (ifTrue, ifFalse):
                    with ifTrue:  # arg <= 0
                        context.pushException(ValueError, "math domain error")
            elif self.typeRepresentation is log10:
                with context.ifelse(arg.nonref_expr.lte(0.0)) as (ifTrue, ifFalse):
                    with ifTrue:
                        context.pushException(ValueError, "math domain error")
            elif self.typeRepresentation is radians:
                # calculate this one without c++
                return context.pushPod(outT, arg.nonref_expr.mul(native_ast.const_float_expr(pi / 180)))
            elif self.typeRepresentation is modf:
                outT = MasqueradingTupleWrapper(Tuple(float, float))
                return_tuple = native_ast.Expression.StackSlot(name=".return_tuple", type=outT.layoutType)
                context.pushEffect(func.call(arg.nonref_expr, return_tuple.elemPtr(0).cast(native_ast.VoidPtr)))
                return typed_python.compiler.typed_expression.TypedExpression(self, return_tuple, outT, True)
            elif self.typeRepresentation is sqrt:
                with context.ifelse(arg.nonref_expr.lt(0.0)) as (ifTrue, ifFalse):
                    with ifTrue:  # arg < 0
                        context.pushException(ValueError, "math domain error")

            ret = context.pushPod(outT, func.call(arg.nonref_expr))
            if check_inf:
                f_isinf = runtime_functions.isinf_float64
                with context.ifelse(f_isinf.call(ret.nonref_expr)) as (ifTrue, ifFalse):
                    with ifTrue:
                        f_isfinite = runtime_functions.isfinite_float64
                        with context.ifelse(f_isfinite.call(arg.nonref_expr)) as (ifTrue2, ifFalse2):
                            with ifTrue2:
                                context.pushException(OverflowError, 'math range error')
            return ret

        return super().convert_call(context, expr, args, kwargs)
