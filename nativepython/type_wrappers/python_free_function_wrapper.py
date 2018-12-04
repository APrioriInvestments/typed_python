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
from typed_python import NoneType
import nativepython.native_ast as native_ast

class PythonFreeFunctionWrapper(Wrapper):
    is_pod = True

    def __init__(self, f):
        super().__init__(f)

    def lower_as_function_arg(self):
        return self.lower()

    def lower(self):
        return native_ast.Type.Void()

    def unwrap(self, e):
        return e

    def convert_bin_op(self, context, left, op, right):
        raise ConversionException("Not convertible")

    def convert_call(self, context, left, args):
        return context.call_py_function(self.typeRepresentation, args)
