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
import typed_python

typeWrapper = lambda t: typed_python.compiler.python_object_representation.typedPythonTypeToTypeWrapper(t)


class MethodDescriptorWrapper(Wrapper):
    is_pod = True
    is_empty = True
    is_pass_by_ref = False

    def __init__(self, f):
        super().__init__(f)

        self.clsType = f.__objclass__
        self.methodName = f.__name__

    def getNativeLayoutType(self):
        return native_ast.Type.Void()

    def convert_call(self, context, left, args, kwargs):
        if not args:
            context.pushException(
                TypeError,
                f"descriptor '{self.methodName}' of '{self.clsType}' object needs an argument"
            )
            return

        if args[0].expr_type != typeWrapper(self.clsType):
            context.pushException(
                TypeError,
                f"descriptor '{self.methodName}' requires a '{self.clsType}' object but "
                f"received a '{args[0].expr_type.typeRepresentation}'"
            )
            return

        return args[0].convert_method_call(self.methodName, args[1:], kwargs)
