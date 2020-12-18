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


class PythonFreeFunctionWrapper(Wrapper):
    is_pod = True
    is_empty = True
    is_pass_by_ref = False

    def __init__(self, f):
        super().__init__(f)

    def getNativeLayoutType(self):
        return native_ast.Type.Void()

    def convert_call(self, context, left, args, kwargs):
        return context.call_py_function(self.typeRepresentation, args, kwargs)

    def convert_to_type_with_target(self, context, instance, targetVal, conversionLevel, mayThrowOnFailure=False):
        if targetVal.expr_type.typeRepresentation is object:
            return context.constant(self.typeRepresentation, allowArbitrary=True)

        if conversionLevel.isNewOrHigher() and targetVal.expr_type.typeRepresentation is str:
            targetVal.convert_copy_initialize(
                context.constant(str(self.typeRepresentation))
            )
            return context.constant(True)

        return super().convert_to_type_with_target(context, instance, targetVal, conversionLevel, mayThrowOnFailure)

    def convert_repr(self, context, instance):
        return context.constant(repr(self.typeRepresentation))

    def convert_typeof(self, context, instance):
        return context.constant(type(self.typeRepresentation))
