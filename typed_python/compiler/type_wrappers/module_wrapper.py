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
from typed_python import Value
import typed_python.compiler.native_ast as native_ast
import typed_python.compiler

typeWrapper = lambda t: typed_python.compiler.python_object_representation.typedPythonTypeToTypeWrapper(t)


class ModuleWrapper(Wrapper):
    is_pod = True
    is_empty = False
    is_pass_by_ref = False

    def __init__(self, f):
        super().__init__(Value(f))

    def getNativeLayoutType(self):
        return native_ast.Type.Void()

    def convert_default_initialize(self, context, target):
        pass

    def convert_assign(self, context, target, toStore):
        pass

    def convert_copy_initialize(self, context, target, toStore):
        pass

    def convert_destroy(self, context, instance):
        pass

    def convert_attribute(self, context, instance, attribute):
        if not hasattr(self.typeRepresentation.Value, attribute):
            return context.pushException(
                AttributeError,
                "module '%s' has no attribute '%s'" % (
                    self.typeRepresentation.Value.__name__,
                    attribute
                )
            )

        return typed_python.compiler.python_object_representation.pythonObjectRepresentation(
            context,
            getattr(self.typeRepresentation.Value, attribute),
        )

    def convert_typeof(self, context, instance):
        return typed_python.compiler.python_object_representation.pythonObjectRepresentation(
            context,
            type(self.typeRepresentation.Value)
        )
