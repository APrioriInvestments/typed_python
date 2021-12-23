#   Copyright 2017-2020 typed_python Authors
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


def externalCallTarget(name, output, *inputs, varargs=False, intrinsic=False, canThrow=False):
    """Create an object that lets us call C functions.

    Note that 'canThrow' really indicates whether we should use llvm 'invoke' instead
    of 'call'. If you set canThrow to false, and you throw from the C function, you'll
    walk up the stack to the exception handler above the call site.
    """
    return native_ast.CallTarget.Named(
        target=native_ast.NamedCallTarget(
            name=name,
            arg_types=inputs,
            output_type=output,
            external=True,
            varargs=varargs,
            intrinsic=intrinsic,
            can_throw=canThrow
        )
    )


def binaryPyobjCallTarget(name):
    return externalCallTarget(
        name,
        Void.pointer(),
        Void.pointer(),
        Void.pointer(),
        canThrow=True
    )


def unaryPyobjCallTarget(name, retType=Void.pointer()):
    return externalCallTarget(
        name,
        retType,
        Void.pointer(),
        canThrow=True
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
    python_ast.ComparisonOp.Eq(): binaryPyobjCallTarget("np_pyobj_EQ"),
    python_ast.ComparisonOp.NotEq(): binaryPyobjCallTarget("np_pyobj_NE"),
    python_ast.ComparisonOp.Lt(): binaryPyobjCallTarget("np_pyobj_LT"),
    python_ast.ComparisonOp.Gt(): binaryPyobjCallTarget("np_pyobj_GT"),
    python_ast.ComparisonOp.LtE(): binaryPyobjCallTarget("np_pyobj_LE"),
    python_ast.ComparisonOp.GtE(): binaryPyobjCallTarget("np_pyobj_GE"),
    python_ast.ComparisonOp.In(): binaryPyobjCallTarget("np_pyobj_In"),
    python_ast.ComparisonOp.NotIn(): binaryPyobjCallTarget("np_pyobj_NotIn"),
}

pyOpToUnaryCallTarget = {
    python_ast.UnaryOp.Not(): externalCallTarget("np_pyobj_Not", Bool, Void.pointer()),
    python_ast.UnaryOp.Invert(): unaryPyobjCallTarget("np_pyobj_Invert"),
    python_ast.UnaryOp.UAdd(): unaryPyobjCallTarget("np_pyobj_Positive"),
    python_ast.UnaryOp.USub(): unaryPyobjCallTarget("np_pyobj_Negative"),
}


throwNullPtr = externalCallTarget("np_throwNullPtr", Void)

typePtrIsSubclass = externalCallTarget("np_typePtrIsSubclass", Bool, Void.pointer(), Void.pointer())
convertTypePtrToTypeObj = externalCallTarget("np_convertTypePtrToTypeObj", Void.pointer(), Void.pointer())
classTypeAsPointer = externalCallTarget("np_classTypeAsPointer", Void.pointer(), Void.pointer())
compileClassDispatch = externalCallTarget("np_compileClassDispatch", Void, Void.pointer(), Int64)
compileClassDestructor = externalCallTarget("np_compileClassDestructor", Void, Void.pointer())
classObjectDowncast = externalCallTarget(
    "np_classObjectDowncast", Bool, Void.pointer(), Void.pointer(), Void.pointer()
)

free = externalCallTarget("tp_free", Void, UInt8Ptr)
malloc = externalCallTarget("tp_malloc", UInt8Ptr, Int64)

realloc = externalCallTarget("tp_realloc", UInt8Ptr, UInt8Ptr, Int64, Int64)
memcpy = externalCallTarget("memcpy", UInt8Ptr, UInt8Ptr, UInt8Ptr, Int64)
memmove = externalCallTarget("memmove", UInt8Ptr, UInt8Ptr, UInt8Ptr, Int64)

computeTypeClassDispatchTable = externalCallTarget(
    "computeTypeClassDispatchTable", Void.pointer(), Void.pointer(), Void.pointer()
)

acos64 = externalCallTarget("np_acos_float64", Float64, Float64)

acosh64 = externalCallTarget("np_acosh_float64", Float64, Float64)

asin64 = externalCallTarget("np_asin_float64", Float64, Float64)

asinh64 = externalCallTarget("np_asinh_float64", Float64, Float64)

atan64 = externalCallTarget("np_atan_float64", Float64, Float64)

atan2_64 = externalCallTarget("np_atan2_float64", Float64, Float64, Float64)

atanh64 = externalCallTarget("np_atanh_float64", Float64, Float64)

ceil64 = externalCallTarget("llvm.ceil.f64", Float64, Float64, intrinsic=True)

copysign64 = externalCallTarget("llvm.copysign.f64", Float64, Float64, Float64, intrinsic=True)

cos64 = externalCallTarget("llvm.cos.f64", Float64, Float64, intrinsic=True)

cosh64 = externalCallTarget("np_cosh_float64", Float64, Float64)

erf64 = externalCallTarget("np_erf_float64", Float64, Float64)

erfc64 = externalCallTarget("np_erfc_float64", Float64, Float64)

exp64 = externalCallTarget("llvm.exp.f64", Float64, Float64, intrinsic=True)

expm1_64 = externalCallTarget("np_expm1_float64", Float64, Float64)

fabs64 = externalCallTarget("llvm.fabs.f64", Float64, Float64, intrinsic=True)

factorial = externalCallTarget("np_factorial", Int64, Int64)
factorial64 = externalCallTarget("np_factorial64", Float64, Float64)

floor64 = externalCallTarget("llvm.floor.f64", Float64, Float64, intrinsic=True)

fmod64 = externalCallTarget("np_fmod_float64", Float64, Float64, Float64)

frexp64 = externalCallTarget("np_frexp_float64", Void, Float64, Void.pointer())

gamma64 = externalCallTarget("np_gamma_float64", Float64, Float64)

gcd = externalCallTarget("np_gcd", Int64, Int64, Int64)

isclose64 = externalCallTarget("np_isclose_float64", Bool, Float64, Float64, Float64, Float64)

ldexp64 = externalCallTarget("np_ldexp_float64", Float64, Float64, Int64)

lgamma64 = externalCallTarget("np_lgamma_float64", Float64, Float64)

log64 = externalCallTarget("llvm.log.f64", Float64, Float64, intrinsic=True)

log1p64 = externalCallTarget("np_log1p_float64", Float64, Float64)

log2_64 = externalCallTarget("llvm.log2.f64", Float64, Float64, intrinsic=True)

log10_64 = externalCallTarget("llvm.log10.f64", Float64, Float64, intrinsic=True)

modf64 = externalCallTarget("np_modf_float64", Void, Float64, Void.pointer())

pow64 = externalCallTarget("llvm.pow.f64", Float64, Float64, Float64, intrinsic=True)

remainder64 = externalCallTarget("llvm.frem.f64", Float64, Float64, Float64, intrinsic=True)

sin64 = externalCallTarget("llvm.sin.f64", Float64, Float64, intrinsic=True)

sinh64 = externalCallTarget("np_sinh_float64", Float64, Float64)

sqrt64 = externalCallTarget("llvm.sqrt.f64", Float64, Float64, intrinsic=True)

tan64 = externalCallTarget("np_tan_float64", Float64, Float64)

tanh64 = externalCallTarget("np_tanh_float64", Float64, Float64)

trunc64 = externalCallTarget("llvm.trunc.f64", Float64, Float64, intrinsic=True)

initialize_exception = externalCallTarget(
    "np_initialize_exception",
    Void,
    Void.pointer()
)

initialize_exception_w_cause = externalCallTarget(
    "np_initialize_exception_w_cause",
    Void,
    Void.pointer(),
    Void.pointer()
)

clear_exception = externalCallTarget(
    "np_clear_exception",
    Void
)

clear_exc_info = externalCallTarget(
    "np_clear_exc_info",
    Void
)

match_exception = externalCallTarget(
    "np_match_exception",
    Bool,
    Void.pointer()
)

match_given_exception = externalCallTarget(
    "np_match_given_exception",
    Bool,
    Void.pointer(),
    Void.pointer()
)

fetch_exception = externalCallTarget(
    "np_fetch_exception",
    Void.pointer()
)

catch_exception = externalCallTarget(
    "np_catch_exception",
    Void
)

fetch_exception_tuple = externalCallTarget(
    "np_fetch_exception_tuple",
    Void,
    Void.pointer()
)

raise_exception_tuple = externalCallTarget(
    "np_raise_exception_tuple",
    Void,
    Void.pointer()
)

builtin_pyobj_by_name = externalCallTarget(
    "np_builtin_pyobj_by_name",
    Void.pointer(),
    UInt8.pointer()
)

pyobj_issubclass = externalCallTarget(
    "np_pyobj_issubclass",
    Bool,
    Void.pointer(),
    Void.pointer()
)

pyobj_typeof = externalCallTarget(
    "np_pyobj_typeof",
    Void.pointer(),
    Void.pointer()
)

pyobj_pynumber_index = externalCallTarget(
    "np_pyobj_pynumber_index",
    Int64,
    Void.pointer()
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
    Float64, Float64,
    canThrow=True
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

create_pyobj = externalCallTarget(
    "nativepython_runtime_create_pyobj",
    Void.pointer(),
    Void.pointer()
)

call_pyobj = externalCallTarget(
    "nativepython_runtime_call_pyobj",
    Void.pointer(),
    Void.pointer(),
    varargs=True,
    canThrow=True
)

call_func_as_pyobj = externalCallTarget(
    "nativepython_runtime_call_func_as_pyobj",
    Void.pointer(),
    Void.pointer(),
    varargs=True,
    canThrow=True
)

get_pyobj_None = externalCallTarget(
    "nativepython_runtime_get_pyobj_None",
    Void.pointer()
)

destroy_pyobj_handle = externalCallTarget(
    "np_destroy_pyobj_handle",
    Void,
    Void.pointer()
)

np_repr = externalCallTarget(
    "nativepython_runtime_repr",
    Void.pointer(),
    Void.pointer(),
    Void.pointer(),
    canThrow=True
)

np_try_pyobj_to_str = externalCallTarget(
    "np_try_pyobj_to_str",
    Bool,
    Void.pointer(),
    Void.pointer(),
    Void.pointer(),
    canThrow=False
)

np_str = externalCallTarget(
    "nativepython_runtime_str",
    Void.pointer(),
    Void.pointer(),
    Void.pointer(),
    canThrow=True
)

pyobj_len = externalCallTarget(
    "nativepython_pyobj_len",
    Int64,
    Void.pointer(),
    canThrow=True
)

getattr_pyobj = externalCallTarget(
    "nativepython_runtime_getattr_pyobj",
    Void.pointer(),
    Void.pointer(),
    UInt8Ptr,
    canThrow=True
)

setattr_pyobj = externalCallTarget(
    "nativepython_runtime_setattr_pyobj",
    Void,
    Void.pointer(),
    UInt8Ptr,
    Void.pointer(),
    canThrow=True
)

delitem_pyobj = externalCallTarget(
    "nativepython_runtime_delitem_pyobj",
    Void,
    Void.pointer(),
    Void.pointer(),
    canThrow=True
)

getitem_pyobj = externalCallTarget(
    "nativepython_runtime_getitem_pyobj",
    Void.pointer(),
    Void.pointer(),
    Void.pointer(),
    canThrow=True
)

setitem_pyobj = externalCallTarget(
    "nativepython_runtime_setitem_pyobj",
    Void,
    Void.pointer(),
    Void.pointer(),
    Void.pointer(),
    canThrow=True
)

pyobj_to_typed = externalCallTarget(
    "np_runtime_pyobj_to_typed",
    Bool,
    Void.pointer(),
    Void.pointer(),
    Void.pointer(),
    Int64,
    Int64,
    canThrow=True
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
    Void.pointer(),
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
    Void.pointer(), Void.pointer(), Void.Pointer(), Int64
)

class_cmp = externalCallTarget(
    "np_runtime_class_cmp",
    Bool,
    UInt64, Void.pointer(), Void.Pointer(), Int64
)

string_chr_int64 = externalCallTarget(
    "nativepython_runtime_string_chr",
    Void.pointer(),
    Int64,
    canThrow=True
)

string_ord = externalCallTarget(
    "nativepython_runtime_string_ord",
    Int64,
    Void.pointer(),
    canThrow=True
)

string_getslice_int64 = externalCallTarget(
    "nativepython_runtime_string_getslice_int64",
    Void.pointer(),
    Void.pointer(), Int64, Int64
)

string_getitem_int64 = externalCallTarget(
    "nativepython_runtime_string_getitem_int64",
    Void.pointer(),
    Void.pointer(), Int64
)

string_mult = externalCallTarget(
    "nativepython_runtime_string_mult",
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

string_capitalize = externalCallTarget(
    "nativepython_runtime_string_capitalize",
    Void.pointer(),
    Void.pointer()
)

string_casefold = externalCallTarget(
    "nativepython_runtime_string_casefold",
    Void.pointer(),
    Void.pointer()
)

string_swapcase = externalCallTarget(
    "nativepython_runtime_string_swapcase",
    Void.pointer(),
    Void.pointer()
)

string_title = externalCallTarget(
    "nativepython_runtime_string_title",
    Void.pointer(),
    Void.pointer()
)

string_find = externalCallTarget(
    "nativepython_runtime_string_find",
    Int64,
    Void.pointer(), Void.pointer(), Int64, Int64
)

string_rfind = externalCallTarget(
    "nativepython_runtime_string_rfind",
    Int64,
    Void.pointer(), Void.pointer(), Int64, Int64
)

string_index = externalCallTarget(
    "nativepython_runtime_string_index",
    Int64,
    Void.pointer(), Void.pointer(), Int64, Int64
)

string_rindex = externalCallTarget(
    "nativepython_runtime_string_rindex",
    Int64,
    Void.pointer(), Void.pointer(), Int64, Int64
)

string_count = externalCallTarget(
    "nativepython_runtime_string_count",
    Int64,
    Void.pointer(), Void.pointer(), Int64, Int64
)

string_join = externalCallTarget(
    "nativepython_runtime_string_join",
    Void,
    Void.pointer(), Void.pointer(), Void.pointer()
)

string_split = externalCallTarget(
    "nativepython_runtime_string_split",
    Void.pointer(),
    Void.pointer(), Void.pointer(), Int64
)

string_rsplit = externalCallTarget(
    "nativepython_runtime_string_rsplit",
    Void.pointer(),
    Void.pointer(), Void.pointer(), Int64
)

string_splitlines = externalCallTarget(
    "nativepython_runtime_string_splitlines",
    Void.pointer(),
    Void.pointer(),
    Bool
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

string_isidentifier = externalCallTarget(
    "nativepython_runtime_string_isidentifier",
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

bytes_getslice_int64 = externalCallTarget(
    "nativepython_runtime_bytes_getslice_int64",
    Void.pointer(),
    Void.pointer(),
    Int64, Int64
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

bytes_join = externalCallTarget(
    "nativepython_runtime_bytes_join",
    Void,
    Void.pointer(), Void.pointer(), Void.pointer()
)

bytes_split = externalCallTarget(
    "nativepython_runtime_bytes_split",
    Void.pointer(),
    Void.pointer(), Void.pointer(), Int64
)

bytes_rsplit = externalCallTarget(
    "nativepython_runtime_bytes_rsplit",
    Void.pointer(),
    Void.pointer(), Void.pointer(), Int64
)

bytes_lower = externalCallTarget(
    "nativepython_runtime_bytes_lower",
    Void.pointer(),
    Void.pointer()
)

bytes_upper = externalCallTarget(
    "nativepython_runtime_bytes_upper",
    Void.pointer(),
    Void.pointer()
)

bytes_capitalize = externalCallTarget(
    "nativepython_runtime_bytes_capitalize",
    Void.pointer(),
    Void.pointer()
)

bytes_swapcase = externalCallTarget(
    "nativepython_runtime_bytes_swapcase",
    Void.pointer(),
    Void.pointer()
)

bytes_title = externalCallTarget(
    "nativepython_runtime_bytes_title",
    Void.pointer(),
    Void.pointer()
)

bytes_splitlines = externalCallTarget(
    "nativepython_runtime_bytes_splitlines",
    Void.pointer(),
    Void.pointer(),
    Bool
)

bytes_strip = externalCallTarget(
    "nativepython_runtime_bytes_strip",
    Void.pointer(),
    Void.pointer(),
    Bool,
    Bool
)

bytes_strip2 = externalCallTarget(
    "nativepython_runtime_bytes_strip2",
    Void.pointer(),
    Void.pointer(),
    Void.pointer(),
    Bool,
    Bool
)

bytes_mult = externalCallTarget(
    "nativepython_runtime_bytes_mult",
    Void.pointer(),
    Void.pointer(),
    Int64
)

bytes_replace = externalCallTarget(
    "nativepython_runtime_bytes_replace",
    Void.pointer(),
    Void.pointer(),
    Void.pointer(),
    Void.pointer(),
    Int64
)

bytes_decode = externalCallTarget(
    "nativepython_runtime_bytes_decode",
    Void.pointer(),
    Void.pointer(),
    Void.pointer(),
    Void.pointer()
)

str_encode = externalCallTarget(
    "nativepython_runtime_str_encode",
    Void.pointer(),
    Void.pointer(),
    Void.pointer(),
    Void.pointer()
)

bytes_translate = externalCallTarget(
    "nativepython_runtime_bytes_translate",
    Void.pointer(),
    Void.pointer(),
    Void.pointer(),
    Void.pointer()
)

bytes_maketrans = externalCallTarget(
    "nativepython_runtime_bytes_maketrans",
    Void.pointer(),
    Void.pointer(),
    Void.pointer()
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

float64_to_int = externalCallTarget(
    "nativepython_float64_to_int",
    Bool,
    Void.pointer(),
    Float64,
    Bool,
    Void.pointer(),
    canThrow=True
)

float32_to_int = externalCallTarget(
    "nativepython_float32_to_int",
    Bool,
    Void.pointer(),
    Float32,
    Bool,
    Void.pointer(),
    canThrow=True
)

bool_to_string = externalCallTarget(
    "nativepython_bool_to_string",
    Void.pointer(),
    Bool
)

table_create = externalCallTarget(
    "nativepython_tableCreate",
    Void.pointer()
)

table_allocate_new_slot = externalCallTarget(
    "nativepython_tableAllocateNewSlot",
    Int32,
    Void.pointer(), Int64
)

table_copy = externalCallTarget(
    "nativepython_tableCopy",
    Void.pointer(),
    Void.pointer(),
    Void.pointer()
)

table_resize = externalCallTarget(
    "nativepython_tableResize",
    Void,
    Void.pointer()
)

table_compress = externalCallTarget(
    "nativepython_tableCompress",
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
    Void.pointer()
)

hash_class = externalCallTarget(
    "nativepython_hash_class",
    Int32,
    Void.pointer(),
    Void.pointer()
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

np_dir = externalCallTarget(
    "nativepython_runtime_dir",
    Void.pointer(),
    Void.pointer(),
    Void.pointer(),
)

pyobj_rlocktype_unlock = externalCallTarget(
    "np_pyobj_rlocktype_unlock",
    Bool,
    Void.pointer()
)

pyobj_rlocktype_lock = externalCallTarget(
    "np_pyobj_rlocktype_lock",
    Bool,
    Void.pointer()
)

pyobj_locktype_unlock = externalCallTarget(
    "np_pyobj_locktype_unlock",
    Bool,
    Void.pointer()
)

pyobj_locktype_lock = externalCallTarget(
    "np_pyobj_locktype_lock",
    Bool,
    Void.pointer()
)

pyobj_iter_next = externalCallTarget(
    "np_pyobj_iter_next",
    Void.pointer(),
    Void.pointer()
)

pyobj_to_bool = externalCallTarget(
    "np_pyobj_to_bool",
    Bool,
    Void.pointer()
)

pyobj_to_int64 = externalCallTarget(
    "np_pyobj_to_int64",
    Int64,
    Void.pointer()
)

pyobj_to_float64 = externalCallTarget(
    "np_pyobj_to_float64",
    Float64,
    Void.pointer()
)

pyobj_ceil = externalCallTarget(
    "np_pyobj_ceil",
    Float64,
    Void.pointer()
)

pyobj_floor = externalCallTarget(
    "np_pyobj_floor",
    Float64,
    Void.pointer()
)

pyobj_trunc = externalCallTarget(
    "np_pyobj_trunc",
    Float64,
    Void.pointer()
)

list_or_tuple_of_to_bytes = externalCallTarget(
    "tp_list_or_tuple_of_to_bytes",
    Void.pointer(),  # returns a BytesType::layout*
    Void.pointer(),  # accepts a TupleOrListOfType::layout_type*
    Void.pointer()   # and a Type*
)

list_or_tuple_of_from_bytes = externalCallTarget(
    "tp_list_or_tuple_of_from_bytes",
    Void.pointer(),  # returns a TupleOrListOfType::layout_type*
    Void.pointer(),  # accepts a BytesType::layout*
    Void.pointer()   # and a Type*
)

str_to_int64 = externalCallTarget(
    "np_str_to_int64",
    Int64,
    Void.pointer(),
    canThrow=True
)

str_to_float64 = externalCallTarget(
    "np_str_to_float64",
    Float64,
    Void.pointer(),
    canThrow=True
)

bytes_to_int64 = externalCallTarget(
    "np_bytes_to_int64",
    Int64,
    Void.pointer(),
    canThrow=True
)

bytes_to_float64 = externalCallTarget(
    "np_bytes_to_float64",
    Float64,
    Void.pointer(),
    canThrow=True
)

# sets the python exception state to an exception of type
raise_exception_fastpath = externalCallTarget(
    "np_raise_exception_fastpath",
    Void,
    UInt8Ptr,
    UInt8Ptr
)

raiseAttributeError = externalCallTarget(
    "np_raiseAttributeErr",
    Void,
    UInt8Ptr
)
