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

from typed_python.compiler.type_wrappers.wrapper import Wrapper
import typed_python.compiler.native_ast as native_ast
from typed_python import Float32
import typed_python.compiler.type_wrappers.runtime_functions as runtime_functions

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
    # factorial,
    floor,
    # fmod,
    # frexp,
    # fsum,
    gamma,
    # gcd,
    hypot,
    # inf,
    # isclose,
    isfinite,
    isinf,
    isnan,
    # ldexp,
    lgamma,
    log,
    log10,
    log1p,
    log2,
    # modf,
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
)


class MathFunctionWrapper(Wrapper):
    is_pod = True
    is_empty = False
    is_pass_by_ref = False

    SUPPORTED_FUNCTIONS = (acos, acosh, asin, asinh, atan, atan2, atanh,
                           copysign, cos, cosh, degrees, erf, erfc, exp, expm1, fabs,
                           hypot, gamma, isnan, isfinite, isinf, lgamma,
                           log, log2, log10, log1p, pow, radians,
                           sin, sinh, sqrt, tan, tanh)

    def __init__(self, mathFun):
        assert mathFun in self.SUPPORTED_FUNCTIONS
        super().__init__(mathFun)

    def getNativeLayoutType(self):
        return native_ast.Type.Void()

    def convert_call(self, context, expr, args, kwargs):
        if len(args) == 2 and not kwargs:
            arg1 = args[0]
            arg2 = args[1]
            argType1 = arg1.expr_type.typeRepresentation
            argType2 = arg2.expr_type.typeRepresentation
            if argType1 not in (Float32, float):
                arg1 = arg1.convert_to_type(float)
                if arg1 is None:
                    return None
                argType1 = float
            if argType2 not in (Float32, float):
                arg2 = arg2.convert_to_type(float)
                if arg2 is None:
                    return None
                argType2 = float
            if argType1 == Float32 and argType2 == float:
                arg1 = arg1.convert_to_type(float)
                argType1 = float
            if argType1 == float and argType2 == Float32:
                arg2 = arg2.convert_to_type(float)
                argType2 = float
            assert argType1 == argType2

            outT = argType1  # default behavior is to return same type as arguments
            if self.typeRepresentation is copysign:
                func = runtime_functions.copysign32 if argType1 is Float32 else runtime_functions.copysign64
                outT = argType1
            elif self.typeRepresentation is pow:
                f_func = runtime_functions.floor32 if argType1 is Float32 else runtime_functions.floor64
                with context.ifelse(arg1.nonref_expr.lte(0.0)) as (ifTrue, ifFalse):
                    with ifTrue:  # arg1 <= 0.0
                        with context.ifelse(arg1.nonref_expr.eq(0.0)) as (ifTrue2, ifFalse2):
                            with ifTrue2:  # arg1 == 0.0
                                with context.ifelse(arg2.nonref_expr.lt(0.0)) as (ifTrue3, ifFalse3):
                                    with ifTrue3:
                                        context.pushException(ValueError, "math domain error")
                            with ifFalse2:  # arg1 < 0.0
                                with context.ifelse(f_func.call(arg2.nonref_expr).sub(arg2.nonref_expr).eq(0.0)) as (ifTrue4, ifFalse4):
                                    with ifFalse4:
                                        context.pushException(ValueError, "math domain error")
                func = runtime_functions.pow32 if argType1 is Float32 else runtime_functions.pow64
            elif self.typeRepresentation is atan2:
                func = runtime_functions.atan2_32 if argType1 is Float32 else runtime_functions.atan2_64
            elif self.typeRepresentation is hypot:
                func = runtime_functions.sqrt32 if argType1 is Float32 else runtime_functions.sqrt64
                return context.pushPod(outT, func.call(arg1.nonref_expr.mul(arg1.nonref_expr).add(arg2.nonref_expr.mul(arg2.nonref_expr))))
            # elif self.typeRepresentation is remainder:  # added in 3.7
            #     func = runtime_functions.remainder32 if argType1 is Float32 else runtime_functions.remainder64
            else:
                assert False, "Unreachable"

            return context.pushPod(outT, func.call(arg1.nonref_expr, arg2.nonref_expr))
        elif len(args) == 1 and not kwargs:
            arg = args[0]

            if not arg.expr_type.is_arithmetic:
                return context.pushException(TypeError, f"must be a real number, not {arg.expr_type}")

            argType = arg.expr_type.typeRepresentation

            if argType not in (Float32, float):
                if self.typeRepresentation in (isnan, isinf):
                    return context.constant(False)
                elif self.typeRepresentation in (isfinite,):
                    return context.constant(True)
                elif self.typeRepresentation in (log, log2, log10, exp, cos, sin, sqrt, fabs):
                    arg = arg.convert_to_type(float)
                    if arg is None:
                        return None
                    argType = float

            outT = argType  # default behavior is to return same type as argument
            if self.typeRepresentation is acos:
                with context.ifelse(arg.nonref_expr.gt(1.0)) as (ifTrue, ifFalse):
                    with ifTrue:
                        context.pushException(ValueError, "math domain error")
                with context.ifelse(arg.nonref_expr.lt(-1.0)) as (ifTrue, ifFalse):
                    with ifTrue:
                        context.pushException(ValueError, "math domain error")
                func = runtime_functions.acos32 if argType is Float32 else runtime_functions.acos64
            elif self.typeRepresentation is acosh:
                with context.ifelse(arg.nonref_expr.lt(1.0)) as (ifTrue, ifFalse):
                    with ifTrue:
                        context.pushException(ValueError, "math domain error")
                func = runtime_functions.acosh32 if argType is Float32 else runtime_functions.acosh64
            elif self.typeRepresentation is asin:
                with context.ifelse(arg.nonref_expr.gt(1.0)) as (ifTrue, ifFalse):
                    with ifTrue:
                        context.pushException(ValueError, "math domain error")
                with context.ifelse(arg.nonref_expr.lt(-1.0)) as (ifTrue, ifFalse):
                    with ifTrue:
                        context.pushException(ValueError, "math domain error")
                func = runtime_functions.asin32 if argType is Float32 else runtime_functions.asin64
            elif self.typeRepresentation is asinh:
                func = runtime_functions.asinh32 if argType is Float32 else runtime_functions.asinh64
            elif self.typeRepresentation is atan:
                func = runtime_functions.atan32 if argType is Float32 else runtime_functions.atan64
            elif self.typeRepresentation is atanh:
                with context.ifelse(arg.nonref_expr.gte(1.0)) as (ifTrue, ifFalse):
                    with ifTrue:
                        context.pushException(ValueError, "math domain error")
                with context.ifelse(arg.nonref_expr.lte(-1.0)) as (ifTrue, ifFalse):
                    with ifTrue:
                        context.pushException(ValueError, "math domain error")
                func = runtime_functions.atanh32 if argType is Float32 else runtime_functions.atanh64
            elif self.typeRepresentation is ceil:
                func = runtime_functions.ceil32 if argType is Float32 else runtime_functions.ceil64
            elif self.typeRepresentation is cos:
                func = runtime_functions.cos32 if argType is Float32 else runtime_functions.cos64
            elif self.typeRepresentation is cosh:
                func = runtime_functions.cosh32 if argType is Float32 else runtime_functions.cosh64
            elif self.typeRepresentation is degrees:
                if argType is Float32:
                    return context.pushPod(outT, arg.nonref_expr.mul(native_ast.const_float32_expr(180 / pi)))
                else:
                    return context.pushPod(outT, arg.nonref_expr.mul(native_ast.const_float_expr(180 / pi)))
            elif self.typeRepresentation is erf:
                func = runtime_functions.erf32 if argType is Float32 else runtime_functions.erf64
            elif self.typeRepresentation is erfc:
                func = runtime_functions.erfc32 if argType is Float32 else runtime_functions.erfc64
            elif self.typeRepresentation is exp:
                func = runtime_functions.exp32 if argType is Float32 else runtime_functions.exp64
            elif self.typeRepresentation is expm1:
                func = runtime_functions.expm1_32 if argType is Float32 else runtime_functions.expm1_64
            elif self.typeRepresentation is fabs:
                func = runtime_functions.fabs32 if argType is Float32 else runtime_functions.fabs64
            elif self.typeRepresentation is floor:
                func = runtime_functions.floor32 if argType is Float32 else runtime_functions.floor64
            elif self.typeRepresentation is gamma:
                f_func = runtime_functions.floor32 if argType is Float32 else runtime_functions.floor64
                with context.ifelse(arg.nonref_expr.lte(0.0)) as (ifTrue, ifFalse):
                    with ifTrue:
                        with context.ifelse(f_func.call(arg.nonref_expr).sub(arg.nonref_expr).eq(0.0)) as (ifTrue2, ifFalse2):
                            with ifTrue2:
                                context.pushException(ValueError, "math domain error")
                func = runtime_functions.gamma32 if argType is Float32 else runtime_functions.gamma64
            elif self.typeRepresentation is isnan:
                func = runtime_functions.isnan_float32 if argType is Float32 else runtime_functions.isnan_float64
                outT = bool
            elif self.typeRepresentation is isfinite:
                func = runtime_functions.isfinite_float32 if argType is Float32 else runtime_functions.isfinite_float64
                outT = bool
            elif self.typeRepresentation is isinf:
                func = runtime_functions.isinf_float32 if argType is Float32 else runtime_functions.isinf_float64
                outT = bool
            elif self.typeRepresentation is lgamma:
                f_func = runtime_functions.floor32 if argType is Float32 else runtime_functions.floor64
                with context.ifelse(arg.nonref_expr.lte(0.0)) as (ifTrue, ifFalse):
                    with ifTrue:
                        with context.ifelse(f_func.call(arg.nonref_expr).sub(arg.nonref_expr).eq(0.0)) as (ifTrue2, ifFalse2):
                            with ifTrue2:
                                context.pushException(ValueError, "math domain error")
                func = runtime_functions.lgamma32 if argType is Float32 else runtime_functions.lgamma64
            elif self.typeRepresentation is log:
                with context.ifelse(arg.nonref_expr.gt(0.0)) as (ifTrue, ifFalse):
                    with ifFalse:
                        context.pushException(ValueError, "math domain error")
                func = runtime_functions.log32 if argType is Float32 else runtime_functions.log64
            elif self.typeRepresentation is log1p:
                with context.ifelse(arg.nonref_expr.gt(-1.0)) as (ifTrue, ifFalse):
                    with ifFalse:
                        context.pushException(ValueError, "math domain error")
                func = runtime_functions.log1p32 if argType is Float32 else runtime_functions.log1p64
            elif self.typeRepresentation is log2:
                with context.ifelse(arg.nonref_expr.gt(0.0)) as (ifTrue, ifFalse):
                    with ifFalse:
                        context.pushException(ValueError, "math domain error")
                func = runtime_functions.log2_32 if argType is Float32 else runtime_functions.log2_64
            elif self.typeRepresentation is log10:
                with context.ifelse(arg.nonref_expr.gt(0.0)) as (ifTrue, ifFalse):
                    with ifFalse:
                        context.pushException(ValueError, "math domain error")
                func = runtime_functions.log10_32 if argType is Float32 else runtime_functions.log10_64
            elif self.typeRepresentation is radians:
                if argType is Float32:
                    return context.pushPod(outT, arg.nonref_expr.mul(native_ast.const_float32_expr(pi / 180)))
                else:
                    return context.pushPod(outT, arg.nonref_expr.mul(native_ast.const_float_expr(pi / 180)))
            elif self.typeRepresentation is sin:
                func = runtime_functions.sin32 if argType is Float32 else runtime_functions.sin64
            elif self.typeRepresentation is sinh:
                func = runtime_functions.sinh32 if argType is Float32 else runtime_functions.sinh64
            elif self.typeRepresentation is sqrt:
                with context.ifelse(arg.nonref_expr.gte(0.0)) as (ifTrue, ifFalse):
                    with ifFalse:
                        context.pushException(ValueError, "math domain error")
                func = runtime_functions.sqrt32 if argType is Float32 else runtime_functions.sqrt64
            elif self.typeRepresentation is tan:
                func = runtime_functions.tan32 if argType is Float32 else runtime_functions.tan64
            elif self.typeRepresentation is tanh:
                func = runtime_functions.tanh32 if argType is Float32 else runtime_functions.tanh64
            elif self.typeRepresentation is trunc:
                func = runtime_functions.trunc32 if argType is Float32 else runtime_functions.trunc64
            else:
                assert False, "Unreachable"

            return context.pushPod(outT, func.call(arg.nonref_expr))

        return super().convert_call(context, expr, args, kwargs)
