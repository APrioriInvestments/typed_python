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
from typed_python import _types
import typed_python.compiler

typeWrapper = lambda x: typed_python.compiler.python_object_representation.typedPythonTypeToTypeWrapper(x)


class BoundMethodWrapper(Wrapper):
    def __init__(self, t):
        super().__init__(t)

        self.firstArgType = typeWrapper(self.typeRepresentation.FirstArgType)

    @staticmethod
    def Make(wrapperType, attr):
        return BoundMethodWrapper(_types.BoundMethod(wrapperType.typeRepresentation, attr))

    def getNativeLayoutType(self):
        return self.firstArgType.getNativeLayoutType()

    def convert_assign(self, context, target, toStore):
        return self.firstArgType.convert_assign(
            context,
            target.changeType(self.firstArgType),
            toStore.changeType(self.firstArgType)
        )

    def convert_copy_initialize(self, context, target, toStore):
        return self.firstArgType.convert_copy_initialize(
            context,
            target.changeType(self.firstArgType),
            toStore.changeType(self.firstArgType)
        )

    def convert_destroy(self, context, instance):
        return self.firstArgType.convert_destroy(
            context,
            instance.changeType(self.firstArgType)
        )

    def convert_call(self, context, left, args, kwargs):
        return self.firstArgType.convert_method_call(
            context,
            left.changeType(self.firstArgType),
            self.typeRepresentation.FuncName,
            args,
            kwargs
        )

    def convert_to_type_with_target(self, context, instance, targetVal, conversionLevel, mayThrowOnFailure=False):
        if targetVal.expr_type.typeRepresentation is bool:
            targetVal.convert_copy_initialize(context.constant(True))
            return context.constant(True)

        return super().convert_to_type_with_target(context, instance, targetVal, conversionLevel, mayThrowOnFailure)
