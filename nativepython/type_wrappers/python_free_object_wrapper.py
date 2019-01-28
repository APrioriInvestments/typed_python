#   Copyright 2019 Nativepython Authors
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
from typed_python import NoneType
import nativepython.native_ast as native_ast
import nativepython

class PythonFreeObjectWrapper(Wrapper):
    """Wraps an arbitrary python object we don't know how to convert.

    Practically speaking, this object can't do anything except interact
    in the type layer. But if we access its attributes or call it with other
    type-like objects, we can resolve them."""
    is_pod = True
    is_empty = True
    is_pass_by_ref = False

    def __init__(self, f):
        super().__init__(f)

    def getNativeLayoutType(self):
        return native_ast.Type.Void()

    def convert_call(self, context, left, args):
        if all([isinstance(x.expr_type, PythonFreeObjectWrapper) for x in args]):
            try:
                return nativepython.python_object_representation.pythonObjectRepresentation(
                    context,
                    self.typeRepresentation(*[a.expr_type.typeRepresentation for a in args])
                    )
            except Exception as e:
                context.pushException(type(e), str(e))
                return

        return context.call_py_function(self.typeRepresentation, args)
