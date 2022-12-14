"""Make AST printing less bad.

Use Numba comparison to do it.

Script will take a function, and print the python ast, our native ast, the llvm ast, and the numba versions for that compiled function.

Comparison will give us an idea how best to display.

NB - NOT POSSIBLE, typed_python needs llvmlite <= 0.38, numba >= 0.39 :(
"""

from multiprocessing.sharedctypes import Value
import os
import pdb
import typed_python
import typed_python.compiler.python_to_native_converter as python_to_native_converter

from numba import jit
from typed_python import Entrypoint, ListOf, Runtime, Function, _types

# from typing import Callable


# def print_asts(input_func: Callable):
#     print("# Normal AST")


def naive_sum(someList, startingInt):
    for x in someList:
        startingInt += x
    return startingInt


compiled = Entrypoint(naive_sum)

# specific_list = ListOf(int)(range(10000))

# # naive_sum(specific_list, 0)
# compiled(specific_list, 0)

# dir(compiled)

# # debug info

runtime = Runtime.singleton()
converter = runtime.converter
compiler = runtime.llvm_compiler
native_converter = compiler.converter

# native_text = Runtime.getNativeIRString(compiled)
# print(native_text)


# llvm_text = Runtime.getLLVMString(compiled)
# print(llvm_text)

# print("="*100)
# print("Double-compiled?")


# specific_list_float = ListOf(float)(range(10000))

# compiled(specific_list_float, 0.0)


# native_text = Runtime.getNativeIRString(compiled)
# print(native_text)


# llvm_text = Runtime.getLLVMString(compiled)
# print(llvm_text)

# print("="*100)
# print("With arg guards")

native_text = Runtime.getNativeIRString(compiled, args=[ListOf(int), int], kwargs=None)
print(native_text)

print("@" * 120)
print("@" * 120)
print("@" * 120)
native_text = Runtime.getNativeIRString(
    compiled, args=[ListOf(float), float], kwargs=None
)
print(native_text)

# llvm_text = Runtime.getLLVMString(compiled, ListOf(int), int)
# print(llvm_text)


def get_full_func_name(compiled_function, argTypes, kwargTypes):
    assert isinstance(compiled_function, typed_python._types.Function)
    typeWrapper = lambda t: python_to_native_converter.typedPythonTypeToTypeWrapper(t)
    # pdb.set_trace()
    compiled_function = _types.prepareArgumentToBePassedToCompiler(compiled_function)

    argTypes = [typeWrapper(a) for a in argTypes]
    kwargTypes = {k: typeWrapper(v) for k, v in kwargTypes.items()}

    overload_index = 0
    overload = compiled_function.overloads[overload_index]

    ExpressionConversionContext = (
        typed_python.compiler.expression_conversion_context.ExpressionConversionContext
    )

    argumentSignature = (
        ExpressionConversionContext.computeFunctionArgumentTypeSignature(
            overload, argTypes, kwargTypes
        )
    )

    if argumentSignature is not None:
        callTarget = runtime.compileFunctionOverload(
            compiled_function, overload_index, argumentSignature, argumentsAreTypes=True
        )
    else:
        raise ValueError("no signature found.")
    return callTarget.name


# pdb.set_trace()
# for key, value in native_converter._functions_by_name.items():
#     print(f'Function {key}' + '_'*20)
#     print(value.body.body)
#     print("_"*80)

# print(get_full_func_name(compiled, [ListOf(float), float], {}))
# print(get_full_func_name(compiled, [ListOf(int), int], {}))


# this needs to be pre-import
# os.environ['TP_COMPILER_VERBOSE'] = "4"

# from typed_python import Entrypoint, ListOf, Runtime
# from typing import Callable

# import pdb


# def print_asts(input_func: Callable):
#     print("# Normal AST")


# def naive_sum(someList, startingInt):
#     for x in someList:
#         startingInt += x
#     return startingInt

# compiled = Entrypoint(naive_sum)

# specific_list = ListOf(int)(range(10000))

# # naive_sum(specific_list, 0)
# compiled(specific_list, 0)


# # debug info

# runtime = Runtime.singleton()
# converter = runtime.converter

# pdb.set_trace()

# numba_compiled = jit(naive_sum, nopython=True)

# numba_compiled(list(specific_list), 0)

# print(numba_compiled.inspect_types())
# x = numba_compiled.inspect_llvm()
# for key, value in x.items():
#     print(key)
#     print(value)
#     print("==========================")
