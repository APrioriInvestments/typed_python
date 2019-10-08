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

from typed_python import Int64, Float64, Bool, String
from typed_python.compiler import native_ast
from typed_python.compiler.type_wrappers.wrapper import Wrapper
from typed_python.compiler.type_wrappers.python_free_object_wrapper import PythonFreeObjectWrapper


class IsinstanceWrapper(Wrapper):
    is_pod = True
    is_empty = False
    is_pass_by_ref = False

    def __init__(self):
        super().__init__(len)

    def getNativeLayoutType(self):
        return native_ast.Type.Void()

    @Wrapper.unwrapOneOfAndValue
    def convert_call(self, context, expr, args, kwargs):
        if len(args) == 2 and not kwargs:
            if isinstance(args[1].expr_type, PythonFreeObjectWrapper):
                # this is the compile-time path
                argType = args[0].expr_type.typeRepresentation

                if argType == Int64:
                    argType = int
                if argType == Float64:
                    argType = float
                if argType == Bool:
                    argType = bool
                if argType == String:
                    argType = str

                if not isinstance(argType, type):
                    return context.constant(False)

                if not isinstance(args[1].expr_type.typeRepresentation.Value, type):
                    return context.pushException(TypeError, "second argument to isinstance must be a type.")

                return context.constant(
                    issubclass(argType, args[1].expr_type.typeRepresentation.Value)
                )
            else:
                # we should be actually trying to figure out what type we have on the RHS
                # either because it's something like OneOf(Value(str), Value(float)), or its 'object'
                # and dispatch accordingly
                raise Exception(
                    "We don't yet know how to handle 'isinstance' unless the right-hand-side is known at compile time."
                )

        return super().convert_call(context, expr, args, kwargs)
