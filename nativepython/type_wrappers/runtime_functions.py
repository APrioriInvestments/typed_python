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

UInt8Ptr = native_ast.UInt8Ptr
Int64 = native_ast.Int64
Float64 = native_ast.Float64
Void = native_ast.Void

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


free=externalCallTarget("free", Void, UInt8Ptr)
malloc=externalCallTarget("malloc", UInt8Ptr, Int64)

stash_exception_ptr = externalCallTarget(
        "nativepython_runtime_stash_const_char_ptr_for_exception", 
        Void, 
        UInt8Ptr
        )

mod_int64_int64 = externalCallTarget(
        "nativepython_runtime_mod_int64_int64", 
        Int64, 
        Int64, Int64
        )
mod_float64_float64 = externalCallTarget(
        "nativepython_runtime_mod_float64_float64", 
        Float64, 
        Float64, Float64
        )
pow_int64_int64 = externalCallTarget(
        "nativepython_runtime_pow_int64_int64", 
        Int64, 
        Int64, Int64
        )
pow_float64_float64 = externalCallTarget(
        "nativepython_runtime_pow_float64_float64", 
        Float64, 
        Float64, Float64
        )
incref_pyobj = externalCallTarget(
        "nativepython_runtime_incref_pyobj", 
        Void, 
        Void.pointer()
        )
decref_pyobj = externalCallTarget(
        "nativepython_runtime_decref_pyobj", 
        Void, 
        Void.pointer()
        )
getattr_pyobj = externalCallTarget(
        "nativepython_runtime_getattr_pyobj", 
        Void.pointer(),
        Void.pointer(),
        UInt8Ptr
        )

int_to_pyobj = externalCallTarget(
        "nativepython_runtime_int_to_pyobj", 
        Void.pointer(),
        Int64
        )

pyobj_to_int = externalCallTarget(
        "nativepython_runtime_pyobj_to_int", 
        Int64,
        Void.pointer()
        )
