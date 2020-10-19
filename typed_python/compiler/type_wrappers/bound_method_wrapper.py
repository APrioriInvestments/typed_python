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
    """Code-generation wrapper for class methods.

    The first argument of a bound method is an instance of the class the method is part of.
    Code generated here, when compiled, is intended to behave the same as interpreted bound methods.
    """
    def __init__(self, t):
        super().__init__(t)

        self.firstArgType = typeWrapper(self.typeRepresentation.FirstArgType)

    @staticmethod
    def Make(wrapperType, attr):
        return BoundMethodWrapper(_types.BoundMethod(wrapperType.typeRepresentation, attr))

    def getNativeLayoutType(self):
        # As data, a bound method is just the class it's a part of.
        # That means we can trivially convert a bound method to its class, with changeType.
        return self.firstArgType.getNativeLayoutType()

    def convert_assign(self, context, target, toStore):
        """Generates code to assign bound method in toStore to target.
        """
        return self.firstArgType.convert_assign(
            context,
            target.changeType(self.firstArgType),
            toStore.changeType(self.firstArgType)
        )

    def convert_copy_initialize(self, context, target, toStore):
        """Generates code to initialize target with bound method in toStore.
        """
        return self.firstArgType.convert_copy_initialize(
            context,
            target.changeType(self.firstArgType),
            toStore.changeType(self.firstArgType)
        )

    def convert_destroy(self, context, instance):
        """Generates code to dispose of a bound method.
        """
        return self.firstArgType.convert_destroy(
            context,
            instance.changeType(self.firstArgType)
        )

    def convert_call(self, context, left, args, kwargs):
        """Generates code to call a bound method.

        Args:
            context: ExpressionConversionContext
            left: TypedExpression of type BoundMethod
            args: arguments of this call
            kwargs: keyword arguments of this call

        Returns:
            TypedExpression of result of this call
        """

        # dispatch this to ClassWrapper
        return self.firstArgType.convert_method_call(
            context,
            left.changeType(self.firstArgType),
            self.typeRepresentation.FuncName,
            args,
            kwargs
        )

    def convert_bool_cast(self, context, e):
        """Generates code to cast bound method to bool.

        By existing, bound methods are True.

        Returns: TypedExpression of True
        """
        return context.constant(True)
