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

from typed_python.internals import (
    isCompiled,
    typeKnownToCompiler,
    localVariableTypesKnownToCompiler
)
from typed_python.compiler.type_wrappers.wrapper import Wrapper
import typed_python.compiler.native_ast as native_ast
import typed_python


class IsCompiledWrapper(Wrapper):
    is_pod = True
    is_empty = False
    is_pass_by_ref = False

    def __init__(self):
        super().__init__(isCompiled)

    def getNativeLayoutType(self):
        return native_ast.Type.Void()

    def convert_call(self, context, expr, args, kwargs):
        if args or kwargs:
            context.pushException(TypeError, "isCompiled() accepts no arguments")

        return context.constant(True)


class TypeKnownToCompiler(Wrapper):
    is_pod = True
    is_empty = False
    is_pass_by_ref = False

    def __init__(self):
        super().__init__(typeKnownToCompiler)

    def getNativeLayoutType(self):
        return native_ast.Type.Void()

    def convert_call(self, context, expr, args, kwargs):
        if len(args) != 1 or kwargs:
            context.pushException(TypeError, "typeKnownToCompiler() accepts 1 positional argument")

        return typed_python.compiler.python_object_representation.pythonObjectRepresentation(
            context,
            args[0].expr_type.typeRepresentation
        )


class LocalVariableTypesKnownToCompiler(Wrapper):
    is_pod = True
    is_empty = False
    is_pass_by_ref = False

    def __init__(self):
        super().__init__(localVariableTypesKnownToCompiler)

    def getNativeLayoutType(self):
        return native_ast.Type.Void()

    def convert_call(self, context, expr, args, kwargs):
        if args or kwargs:
            context.pushException(TypeError, "localVariableTypesKnownToCompiler() accepts no arguments")

        return context.constant(dict(context.variableStates._types), allowArbitrary=True)
