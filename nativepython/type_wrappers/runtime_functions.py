#   Copyright 2017 Braxton Mckee
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

Int64 = native_ast.Int64

def externalCallTarget(name, output, *inputs):
    return native_ast.CallTarget.Named(
        target=native_ast.NamedCallTarget(
            name=name,
            arg_types = inputs,
            output_type = output,
            external=True,
            varargs=False,
            intrinsic=False,
            can_throw=False
            )
        )

nativepython_runtime_mod_int64_int64 = externalCallTarget(
        "nativepython_runtime_mod_int64_int64", 
        Int64, 
        Int64, Int64
        )
nativepython_runtime_pow_int64_int64 = externalCallTarget(
        "nativepython_runtime_pow_int64_int64", 
        Int64, 
        Int64, Int64
        )