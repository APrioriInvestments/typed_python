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

    def convert_call(self, context, left, args, kwargs):
        if all([isinstance(x.expr_type, PythonFreeObjectWrapper) for x in list(args) + list(kwargs.values())]):
            try:
                return nativepython.python_object_representation.pythonObjectRepresentation(
                    context,
                    self.typeRepresentation(
                        *[a.expr_type.typeRepresentation for a in args],
                        **{k: v.expr_type.typeRepresentation for k, v in kwargs.items()}
                    )
                )
            except Exception as e:
                context.pushException(type(e), str(e))
                return

        return context.call_py_function(self.typeRepresentation, args, kwargs)

    def convert_attribute(self, context, instance, attribute):
        try:
            return nativepython.python_object_representation.pythonObjectRepresentation(
                context,
                getattr(self.typeRepresentation, attribute)
            )
        except Exception as e:
            context.pushException(type(e), str(e))
            return
