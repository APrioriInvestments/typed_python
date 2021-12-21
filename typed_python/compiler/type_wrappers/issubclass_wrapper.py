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

from typed_python.compiler import native_ast
from typed_python.compiler.type_wrappers.wrapper import Wrapper


class IssubclassWrapper(Wrapper):
    is_pod = True
    is_empty = False
    is_pass_by_ref = False

    def __init__(self):
        super().__init__(issubclass)

    def getNativeLayoutType(self):
        return native_ast.Type.Void()

    @Wrapper.unwrapOneOfAndValue
    def convert_call(self, context, expr, args, kwargs):
        if len(args) == 2 and not kwargs:
            return args[0].expr_type.convert_issubclass(context, args[0], args[1], True)

        return super().convert_call(context, expr, args, kwargs)
