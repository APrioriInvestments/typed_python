#   Copyright 2018 Braxton Mckee
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

from nativepython.type_wrappers.wrapper import Wrapper

import nativepython.native_ast as native_ast
import nativepython

typeWrapper = lambda x: nativepython.python_object_representation.typedPythonTypeToTypeWrapper(x)


class BoundMethodWrapper(Wrapper):
    def __init__(self, t):
        super().__init__(t)

        self.firstArgType = typeWrapper(self.typeRepresentation.FirstArgType)

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
            target.changeType(self.firstArgType),
            toStore.changeType(self.firstArgType)
        )

    def convert_call(self, context, left, args, kwargs):
        clsType = typeWrapper(self.typeRepresentation.FirstArgType)
        funcType = typeWrapper(self.typeRepresentation.Function)

        return funcType.convert_call(
            context,
            context.pushPod(funcType, native_ast.nullExpr),
            (left.changeType(clsType),) + tuple(args),
            kwargs
        )
