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
from typed_python.compiler.type_wrappers.runtime_functions import print_string


class PrintWrapper(Wrapper):
    is_pod = True
    is_empty = False
    is_pass_by_ref = False

    def __init__(self):
        super().__init__(print)

    def getNativeLayoutType(self):
        return native_ast.Type.Void()

    def convert_call(self, context, expr, args, kwargs):
        sep = context.constant(" ")
        end = context.constant("\n")

        for kwargName, value in kwargs.items():
            if kwargName == 'sep':
                sep = value.convert_str_cast()
                if sep is None:
                    return
            elif kwargName == 'end':
                end = value.convert_str_cast()
                if end is None:
                    return
            else:
                context.pushException(TypeError, f"'{kwargName}' is an invalid keyword argument for this function")
                return

        res = None

        # it would be better to use join
        for a in args:
            converted = a.convert_str_cast()
            if converted is None:
                return None

            if res is None:
                res = converted
            else:
                res = res + sep + converted

        if res is None:
            res = end
        else:
            res = res + end

        context.pushEffect(print_string.call(res.nonref_expr.cast(native_ast.VoidPtr)))

        return context.pushVoid()
