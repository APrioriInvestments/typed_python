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
from typed_python import Float32, Float64
import typed_python.compiler.type_wrappers.runtime_functions as runtime_functions

from math import isnan, isfinite, isinf


class MathFunctionWrapper(Wrapper):
    is_pod = True
    is_empty = False
    is_pass_by_ref = False

    SUPPORTED_FUNCTIONS = (isnan, isfinite, isinf)

    def __init__(self, mathFun):
        assert mathFun in (isnan, isfinite, isinf)
        super().__init__(mathFun)

    def getNativeLayoutType(self):
        return native_ast.Type.Void()

    def convert_call(self, context, expr, args, kwargs):
        if len(args) == 1 and not kwargs:
            if not args[0].expr_type.is_arithmetic:
                return context.pushException(TypeError, f"must be a real number, not {args[0].expr_type}")

            argType = args[0].expr_type.typeRepresentation

            if argType not in (Float32, Float64):
                if self.typeRepresentation in (isnan, isinf):
                    return context.constant(False)
                if self.typeRepresentation in (isfinite,):
                    return context.constant(True)

            if self.typeRepresentation is isnan:
                func = runtime_functions.isnan_float32 if argType is Float32 else runtime_functions.isnan_float64
            elif self.typeRepresentation is isfinite:
                func = runtime_functions.isfinite_float32 if argType is Float32 else runtime_functions.isfinite_float64
            elif self.typeRepresentation is isinf:
                func = runtime_functions.isinf_float32 if argType is Float32 else runtime_functions.isinf_float64
            else:
                assert False, "Unreachable"

            return context.pushPod(bool, func.call(args[0].nonref_expr))

        return super().convert_call(context, expr, args, kwargs)
