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

import nativepython.native_ast as native_ast
import nativepython.type_wrappers.runtime_functions as runtime_functions

def generateThrowException(context, exception):
    return (
        #as a short-term hack, use a runtime function to stash this where the callsite can pick it up.
        native_ast.Expression.Call(
           target=runtime_functions.stash_exception_ptr,
           args=(native_ast.const_utf8_cstr(str(exception)),)
           )
        >> native_ast.Expression.Throw(
            expr=native_ast.Expression.Constant(
                val=native_ast.Constant.NullPointer(value_type=native_ast.UInt8.pointer())
                )
            )
        )
