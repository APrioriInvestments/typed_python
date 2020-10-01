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
#   See the License for the specific language governing permissions and:w
#   limitations under the License.

from typed_python.compiler.type_wrappers.wrapper import Wrapper
import typed_python.compiler.native_ast as native_ast

from math import trunc, floor, ceil


class BuiltinWrapper(Wrapper):
    is_pod = True
    is_empty = False
    is_pass_by_ref = False

    SUPPORTED_FUNCTIONS = (round, trunc, floor, ceil, format, dir, ord, chr)

    def __init__(self, builtin):
        assert builtin in self.SUPPORTED_FUNCTIONS
        super().__init__(builtin)

    def getNativeLayoutType(self):
        return native_ast.Type.Void()

    def convert_call(self, context, expr, args, kwargs):
        builtin = self.typeRepresentation

        if len(args) == 1 and not kwargs:
            return args[0].convert_builtin(builtin)

        # builtins that optionally have a second parameter:
        if builtin in [round, format] and len(args) == 2 and not kwargs:
            return args[0].convert_builtin(builtin, args[1])

        return super().convert_call(context, expr, args, kwargs)
