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

import typed_python.compiler.native_ast as native_ast
import typed_python.python_ast as python_ast


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


def externalCallTarget(name, output, *inputs, varargs=False):
    return native_ast.CallTarget.Named(
        target=native_ast.NamedCallTarget(
            name=name,
            arg_types=inputs,
            output_type=output,
            external=True,
            varargs=varargs,
            intrinsic=False,
            can_throw=False
        )
    )


def binaryPyobjCallTarget(name):
    return externalCallTarget(
        name,
        Void.pointer(),
        Void.pointer(),
        Void.pointer(),
    )


def unaryPyobjCallTarget(name, retType=Void.pointer()):
    return externalCallTarget(
        name,
        retType,
        Void.pointer()
    )


pyOpToBinaryCallTarget = {
    python_ast.BinaryOp.Add(): binaryPyobjCallTarget("np_pyobj_Add"),
    python_ast.BinaryOp.Sub(): binaryPyobjCallTarget("np_pyobj_Subtract"),
    python_ast.BinaryOp.Mult(): binaryPyobjCallTarget("np_pyobj_Multiply"),
    python_ast.BinaryOp.Pow(): binaryPyobjCallTarget("np_pyobj_Pow"),
    python_ast.BinaryOp.MatMult(): binaryPyobjCallTarget("np_pyobj_MatrixMultiply"),
    python_ast.BinaryOp.Div(): binaryPyobjCallTarget("np_pyobj_TrueDivide"),
    python_ast.BinaryOp.FloorDiv(): binaryPyobjCallTarget("np_pyobj_FloorDivide"),
    python_ast.BinaryOp.Mod(): binaryPyobjCallTarget("np_pyobj_Remainder"),
    python_ast.BinaryOp.LShift(): binaryPyobjCallTarget("np_pyobj_Lshift"),
    python_ast.BinaryOp.RShift(): binaryPyobjCallTarget("np_pyobj_Rshift"),
    python_ast.BinaryOp.BitOr(): binaryPyobjCallTarget("np_pyobj_Or"),
    python_ast.BinaryOp.BitXor(): binaryPyobjCallTarget("np_pyobj_Xor"),
    python_ast.BinaryOp.BitAnd(): binaryPyobjCallTarget("np_pyobj_And"),
}


pyInplaceOpToBinaryCallTarget = {}


pyOpToUnaryCallTarget = {
    python_ast.UnaryOp.Not(): externalCallTarget("np_pyobj_Not", Bool, Void.pointer()),
    python_ast.UnaryOp.Invert(): unaryPyobjCallTarget("np_pyobj_Invert"),
    python_ast.UnaryOp.UAdd(): unaryPyobjCallTarget("np_pyobj_Positive"),
    python_ast.UnaryOp.USub(): unaryPyobjCallTarget("np_pyobj_Negative"),
}


free = externalCallTarget("free", Void, UInt8Ptr)
malloc = externalCallTarget("malloc", UInt8Ptr, Int64)
realloc = externalCallTarget("realloc", UInt8Ptr, UInt8Ptr, Int64)
memcpy = externalCallTarget("memcpy", UInt8Ptr, UInt8Ptr, UInt8Ptr, Int64)
memmove = externalCallTarget("memmove", UInt8Ptr, UInt8Ptr, UInt8Ptr, Int64)

initialize_exception = externalCallTarget(
    "np_initialize_exception",
    Void,
    Void.pointer()
)

builtin_pyobj_by_name = externalCallTarget(
    "np_builtin_pyobj_by_name",
    Void.pointer(),
    UInt8.pointer()
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

call_pyobj = externalCallTarget(
    "nativepython_runtime_call_pyobj",
    Void.pointer(),
    Void.pointer(),
    varargs=True
)

get_pyobj_None = externalCallTarget(
    "nativepython_runtime_get_pyobj_None",
    Void,
    Void.pointer()
)

incref_pyobj = externalCallTarget(
    "nativepython_runtime_incref_pyobj",
    Void.pointer(),
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

np_len = externalCallTarget(
    "nativepython_runtime_len",
    UInt64,
    Void.pointer(),
    UInt64
)

np_contains = externalCallTarget(
    "nativepython_runtime_contains",
    UInt64,
    Void.pointer(),
    UInt64,
    Void.pointer(),
    UInt64
)

pyobj_len = externalCallTarget(
    "nativepython_pyobj_len",
    Int64,
    Void.pointer()
)

getattr_pyobj = externalCallTarget(
    "nativepython_runtime_getattr_pyobj",
    Void.pointer(),
    Void.pointer(),
    UInt8Ptr
)

setattr_pyobj = externalCallTarget(
    "nativepython_runtime_setattr_pyobj",
    Void,
    Void.pointer(),
    UInt8Ptr,
    Void.pointer()
)

delitem_pyobj = externalCallTarget(
    "nativepython_runtime_delitem_pyobj",
    Void,
    Void.pointer(),
    Void.pointer()
)

getitem_pyobj = externalCallTarget(
    "nativepython_runtime_getitem_pyobj",
    Void.pointer(),
    Void.pointer(),
    Void.pointer()
)

setitem_pyobj = externalCallTarget(
    "nativepython_runtime_setitem_pyobj",
    Void,
    Void.pointer(),
    Void.pointer(),
    Void.pointer()
)

pyobj_to_typed = externalCallTarget(
    "np_runtime_pyobj_to_typed",
    Bool,
    Void.pointer(),
    Void.pointer(),
    UInt64,
    Bool
)

instance_to_bool = externalCallTarget(
    "np_runtime_instance_to_bool",
    Bool,
    Void.pointer(),
    UInt64,
)

add_traceback = externalCallTarget(
    "np_add_traceback",
    Void,
    UInt8.pointer(),
    UInt8.pointer(),
    Int64
)

to_pyobj = externalCallTarget(
    "np_runtime_to_pyobj",
    Void.pointer(),
    Void.pointer(),
    UInt64,
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

alternative_cmp = externalCallTarget(
    "np_runtime_alternative_cmp",
    Bool,
    UInt64, Void.pointer(), Void.Pointer(), Int64
)

class_cmp = externalCallTarget(
    "np_runtime_class_cmp",
    Bool,
    UInt64, Void.pointer(), Void.Pointer(), Int64
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

string_strip = externalCallTarget(
    "nativepython_runtime_string_strip",
    Void.pointer(),
    Void.pointer(),
    Bool,
    Bool
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

bool_to_string = externalCallTarget(
    "nativepython_bool_to_string",
    Void.pointer(),
    Bool
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

hash_alternative = externalCallTarget(
    "nativepython_hash_alternative",
    Int32,
    Void.pointer(),
    UInt64
)

hash_class = externalCallTarget(
    "nativepython_hash_class",
    Int32,
    Void.pointer(),
    UInt64
)

isinf_float32 = externalCallTarget("nativepython_isinf_float32", Bool, Float32)

isnan_float32 = externalCallTarget("nativepython_isnan_float32", Bool, Float32)

isfinite_float32 = externalCallTarget("nativepython_isfinite_float32", Bool, Float32)

isinf_float64 = externalCallTarget("nativepython_isinf_float64", Bool, Float64)

isnan_float64 = externalCallTarget("nativepython_isnan_float64", Bool, Float64)

isfinite_float64 = externalCallTarget("nativepython_isfinite_float64", Bool, Float64)

round_float64 = externalCallTarget(
    "nativepython_runtime_round_float64",
    Float64,
    Float64,
    Int64
)

trunc_float64 = externalCallTarget(
    "nativepython_runtime_trunc_float64",
    Float64,
    Float64
)

floor_float64 = externalCallTarget(
    "nativepython_runtime_floor_float64",
    Float64,
    Float64
)

ceil_float64 = externalCallTarget(
    "nativepython_runtime_ceil_float64",
    Float64,
    Float64
)

np_complex = externalCallTarget(
    "nativepython_runtime_complex",
    Void.pointer(),
    Float64,
    Float64
)

np_dir = externalCallTarget(
    "nativepython_runtime_dir",
    Void.pointer(),
    Void.pointer(),
    UInt64
)
