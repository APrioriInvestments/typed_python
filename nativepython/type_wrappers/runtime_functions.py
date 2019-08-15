#   Coyright 2017-2019 Nativepython Authors
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


Bool = native_ast.Bool
UInt8Ptr = native_ast.UInt8Ptr
Int64 = native_ast.Int64
Int32 = native_ast.Int32
Int16 = native_ast.Int16
Int8 = native_ast.Int8
UInt64 = native_ast.UInt64
UInt32 = native_ast.UInt32
UInt16 = native_ast.UInt16
UInt8 = native_ast.UInt8
Float64 = native_ast.Float64
Float32 = native_ast.Float32
Void = native_ast.Void


def externalCallTarget(name, output, *inputs):
    return native_ast.CallTarget.Named(
        target=native_ast.NamedCallTarget(
            name=name,
            arg_types=inputs,
            output_type=output,
            external=True,
            varargs=False,
            intrinsic=False,
            can_throw=False
        )
    )


free = externalCallTarget("free", Void, UInt8Ptr)
malloc = externalCallTarget("malloc", UInt8Ptr, Int64)
realloc = externalCallTarget("realloc", UInt8Ptr, UInt8Ptr, Int64)
memcpy = externalCallTarget("memcpy", UInt8Ptr, UInt8Ptr, UInt8Ptr, Int64)
memmove = externalCallTarget("memmove", UInt8Ptr, UInt8Ptr, UInt8Ptr, Int64)

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

mod_uint64_uint64 = externalCallTarget(
    "nativepython_runtime_mod_uint64_uint64",
    UInt64,
    UInt64, UInt64
)

mod_float64_float64 = externalCallTarget(
    "nativepython_runtime_mod_float64_float64",
    Float64,
    Float64, Float64
)

pow_int64_int64 = externalCallTarget(
    "nativepython_runtime_pow_int64_int64",
    Float64,
    Int64, Int64
)

pow_uint64_uint64 = externalCallTarget(
    "nativepython_runtime_pow_uint64_uint64",
    Float64,
    UInt64, UInt64
)

pow_float64_float64 = externalCallTarget(
    "nativepython_runtime_pow_float64_float64",
    Float64,
    Float64, Float64
)

lshift_int64_int64 = externalCallTarget(
    "nativepython_runtime_lshift_int64_int64",
    Int64,
    Int64, Int64
)

lshift_uint64_uint64 = externalCallTarget(
    "nativepython_runtime_lshift_int64_int64",
    UInt64,
    UInt64, UInt64
)

rshift_int64_int64 = externalCallTarget(
    "nativepython_runtime_rshift_int64_int64",
    Int64,
    Int64, Int64
)

rshift_uint64_uint64 = externalCallTarget(
    "nativepython_runtime_rshift_uint64_uint64",
    UInt64,
    UInt64, UInt64
)

floordiv_int64_int64 = externalCallTarget(
    "nativepython_runtime_floordiv_int64_int64",
    Int64,
    Int64, Int64
)

floordiv_float64_float64 = externalCallTarget(
    "nativepython_runtime_floordiv_float64_float64",
    Float64,
    Float64, Float64
)

get_pyobj_None = externalCallTarget(
    "nativepython_runtime_get_pyobj_None",
    Void,
    Void.pointer()
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

np_repr = externalCallTarget(
    "nativepython_runtime_repr",
    Void.pointer(),
    Void.pointer(),
    UInt64
)

np_str = externalCallTarget(
    "nativepython_runtime_str",
    Void.pointer(),
    Void.pointer(),
    UInt64
)
getattr_pyobj = externalCallTarget(
    "nativepython_runtime_getattr_pyobj",
    Void.pointer(),
    Void.pointer(),
    UInt8Ptr
)

pyobj_to_typed = externalCallTarget(
    "np_runtime_pyobj_to_typed",
    Void,
    Void.pointer(),
    Void.pointer(),
    UInt64
)

instance_to_bool = externalCallTarget(
    "np_runtime_instance_to_bool",
    Bool,
    Void.pointer(),
    UInt64,
)

to_pyobj = externalCallTarget(
    "np_runtime_to_pyobj",
    Void.pointer(),
    Void.pointer(),
    UInt64,
)

int64_to_pyobj = externalCallTarget(
    "np_runtime_int64_to_pyobj",
    Void.pointer(),
    Int64
)

int32_to_pyobj = externalCallTarget(
    "np_runtime_int32_to_pyobj",
    Void.pointer(),
    Int32
)

int16_to_pyobj = externalCallTarget(
    "np_runtime_int16_to_pyobj",
    Void.pointer(),
    Int16
)

int8_to_pyobj = externalCallTarget(
    "np_runtime_int8_to_pyobj",
    Void.pointer(),
    Int8
)

uint64_to_pyobj = externalCallTarget(
    "np_runtime_uint64_to_pyobj",
    Void.pointer(),
    UInt64
)

uint32_to_pyobj = externalCallTarget(
    "np_runtime_uint32_to_pyobj",
    Void.pointer(),
    UInt32
)

uint16_to_pyobj = externalCallTarget(
    "np_runtime_uint16_to_pyobj",
    Void.pointer(),
    UInt16
)

uint8_to_pyobj = externalCallTarget(
    "np_runtime_uint8_to_pyobj",
    Void.pointer(),
    UInt8
)

float64_to_pyobj = externalCallTarget(
    "np_runtime_float64_to_pyobj",
    Void.pointer(),
    Float64
)

float32_to_pyobj = externalCallTarget(
    "np_runtime_float32_to_pyobj",
    Void.pointer(),
    Float32
)

pyobj_to_int64 = externalCallTarget(
    "np_runtime_pyobj_to_int64",
    Int64,
    Void.pointer()
)

pyobj_to_int32 = externalCallTarget(
    "np_runtime_pyobj_to_int32",
    Int32,
    Void.pointer()
)

pyobj_to_int16 = externalCallTarget(
    "np_runtime_pyobj_to_int16",
    Int16,
    Void.pointer()
)

pyobj_to_int8 = externalCallTarget(
    "np_runtime_pyobj_to_int8",
    Int8,
    Void.pointer()
)

pyobj_to_uint64 = externalCallTarget(
    "np_runtime_pyobj_to_uint64",
    UInt64,
    Void.pointer()
)

pyobj_to_uint32 = externalCallTarget(
    "np_runtime_pyobj_to_uint32",
    UInt32,
    Void.pointer()
)

pyobj_to_uint16 = externalCallTarget(
    "np_runtime_pyobj_to_uint16",
    UInt16,
    Void.pointer()
)

pyobj_to_uint8 = externalCallTarget(
    "np_runtime_pyobj_to_uint8",
    UInt8,
    Void.pointer()
)

pyobj_to_float64 = externalCallTarget(
    "np_runtime_pyobj_to_float64",
    Float64,
    Void.pointer()
)

pyobj_to_float32 = externalCallTarget(
    "np_runtime_pyobj_to_float32",
    Float32,
    Void.pointer()
)

pyobj_to_bool = externalCallTarget(
    "np_runtime_pyobj_to_bool",
    Bool,
    Void.pointer()
)

string_concat = externalCallTarget(
    "nativepython_runtime_string_concat",
    Void.pointer(),
    Void.pointer(), Void.pointer()
)

string_cmp = externalCallTarget(
    "nativepython_runtime_string_cmp",
    Int64,
    Void.pointer(), Void.pointer()
)

string_eq = externalCallTarget(
    "nativepython_runtime_string_eq",
    Bool,
    Void.pointer(), Void.pointer()
)

string_getitem_int64 = externalCallTarget(
    "nativepython_runtime_string_getitem_int64",
    Void.pointer(),
    Void.pointer(), Int64
)

string_from_utf8_and_len = externalCallTarget(
    "nativepython_runtime_string_from_utf8_and_len",
    Void.pointer(),
    UInt8Ptr, Int64
)

string_lower = externalCallTarget(
    "nativepython_runtime_string_lower",
    Void.pointer(),
    Void.pointer()
)

string_upper = externalCallTarget(
    "nativepython_runtime_string_upper",
    Void.pointer(),
    Void.pointer()
)

string_find = externalCallTarget(
    "nativepython_runtime_string_find",
    Int64,
    Void.pointer(), Void.pointer(), Int64, Int64
)

string_find_2 = externalCallTarget(
    "nativepython_runtime_string_find_2",
    Int64,
    Void.pointer(), Void.pointer()
)

string_find_3 = externalCallTarget(
    "nativepython_runtime_string_find_3",
    Int64,
    Void.pointer(), Void.pointer(), Int64
)

string_join = externalCallTarget(
    "nativepython_runtime_string_join",
    Void,
    Void.pointer(), Void.pointer(), Void.pointer()
)

string_split = externalCallTarget(
    "nativepython_runtime_string_split",
    Void,
    Void.pointer(), Void.pointer(), Void.pointer(), Int64
)

string_split_2 = externalCallTarget(
    "nativepython_runtime_string_split_2",
    Void,
    Void.pointer(), Void.pointer()
)

string_split_3 = externalCallTarget(
    "nativepython_runtime_string_split_3",
    Void,
    Void.pointer(), Void.pointer(), Void.pointer()
)

string_split_3max = externalCallTarget(
    "nativepython_runtime_string_split_3max",
    Void,
    Void.pointer(), Void.pointer(), Int64
)

string_isalpha = externalCallTarget(
    "nativepython_runtime_string_isalpha",
    Bool,
    Void.pointer()
)

string_isalnum = externalCallTarget(
    "nativepython_runtime_string_isalnum",
    Bool,
    Void.pointer()
)

string_isdecimal = externalCallTarget(
    "nativepython_runtime_string_isdecimal",
    Bool,
    Void.pointer()
)

string_isdigit = externalCallTarget(
    "nativepython_runtime_string_isdigit",
    Bool,
    Void.pointer()
)

string_islower = externalCallTarget(
    "nativepython_runtime_string_islower",
    Bool,
    Void.pointer()
)

string_isnumeric = externalCallTarget(
    "nativepython_runtime_string_isnumeric",
    Bool,
    Void.pointer()
)

string_isprintable = externalCallTarget(
    "nativepython_runtime_string_isprintable",
    Bool,
    Void.pointer()
)

string_isspace = externalCallTarget(
    "nativepython_runtime_string_isspace",
    Bool,
    Void.pointer()
)

string_istitle = externalCallTarget(
    "nativepython_runtime_string_istitle",
    Bool,
    Void.pointer()
)

string_isupper = externalCallTarget(
    "nativepython_runtime_string_isupper",
    Bool,
    Void.pointer()
)

bytes_cmp = externalCallTarget(
    "nativepython_runtime_bytes_cmp",
    Int64,
    Void.pointer(), Void.pointer()
)

bytes_concat = externalCallTarget(
    "nativepython_runtime_bytes_concat",
    Void.pointer(),
    Void.pointer(), Void.pointer()
)

bytes_from_ptr_and_len = externalCallTarget(
    "nativepython_runtime_bytes_from_ptr_and_len",
    Void.pointer(),
    UInt8Ptr, Int64
)

print_string = externalCallTarget(
    "nativepython_print_string",
    Void,
    Void.pointer()
)

int64_to_string = externalCallTarget(
    "nativepython_int64_to_string",
    Void.pointer(),
    Int64
)

uint64_to_string = externalCallTarget(
    "nativepython_uint64_to_string",
    Void.pointer(),
    Int64
)

float64_to_string = externalCallTarget(
    "nativepython_float64_to_string",
    Void.pointer(),
    Float64
)

float32_to_string = externalCallTarget(
    "nativepython_float32_to_string",
    Void.pointer(),
    Float32
)

dict_create = externalCallTarget(
    "nativepython_dict_create",
    Void.pointer()
)

dict_allocateNewSlot = externalCallTarget(
    "nativepython_dict_allocateNewSlot",
    Int32,
    Void.pointer(), Int64
)

dict_resizeTable = externalCallTarget(
    "nativepython_dict_resizeTable",
    Void,
    Void.pointer()
)

dict_compressItemTable = externalCallTarget(
    "nativepython_dict_compressItemTable",
    Void,
    Void.pointer(), Int64
)

hash_float32 = externalCallTarget(
    "nativepython_hash_float32",
    Int32,
    Float32
)

hash_float64 = externalCallTarget(
    "nativepython_hash_float64",
    Int32,
    Float64
)

hash_int64 = externalCallTarget(
    "nativepython_hash_int64",
    Int32,
    Int64
)

hash_uint64 = externalCallTarget(
    "nativepython_hash_uint64",
    Int32,
    UInt64
)

hash_string = externalCallTarget(
    "nativepython_hash_string",
    Int32,
    Void.pointer()
)

hash_bytes = externalCallTarget(
    "nativepython_hash_bytes",
    Int32,
    Void.pointer()
)

isinf_float32 = externalCallTarget("nativepython_isinf_float32", Bool, Float32)

isnan_float32 = externalCallTarget("nativepython_isnan_float32", Bool, Float32)

isfinite_float32 = externalCallTarget("nativepython_isfinite_float32", Bool, Float32)

isinf_float64 = externalCallTarget("nativepython_isinf_float64", Bool, Float64)

isnan_float64 = externalCallTarget("nativepython_isnan_float64", Bool, Float64)

isfinite_float64 = externalCallTarget("nativepython_isfinite_float64", Bool, Float64)
