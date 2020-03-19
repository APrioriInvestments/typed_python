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
from typed_python import Int32


def tp_hash_to_py_hash(hVal):
    """Convert a typed-python hash to a regular python hash.

    Python insists that its hash values are never -1, because it uses -1 as an
    indicator that the exception flag is set. TypedPython doesn't have this behavior
    because it uses c++ exception propagation internally. As a result, it's the
    'hash' wrapper that's responsible for mapping -1 to -2.
    """
    if hVal == -1:
        return Int32(-2)
    return hVal


class HashWrapper(Wrapper):
    is_pod = True
    is_empty = False
    is_pass_by_ref = False

    def __init__(self):
        super().__init__(hash)

    def getNativeLayoutType(self):
        return native_ast.Type.Void()

    def convert_call(self, context, expr, args, kwargs):
        if len(args) == 1 and not kwargs:
            hashVal = args[0].convert_hash()
            if hashVal is None:
                return None

            return context.call_py_function(tp_hash_to_py_hash, (hashVal,), {})

        return super().convert_call(context, expr, args, kwargs)
