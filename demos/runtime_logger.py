"""
llvm_assert_failing.py

Small functions are failing the assert statement in line 661 of compiler/native_ast_to_llvm.py.

Attempt to repro by checking plausible reasons.

The statement is

                assert target.name not in self.converter._function_definitions, target.name


"""
# this has to go first for shell/process reasons
import os

os.environ["TP_COMPILER_CACHE"] = "compiler_cache"


import logging

logging.getLogger("TP_compiler").setLevel(logging.INFO)

from typed_python.compiler.compiler_cache import CompilerCache

# os\.environ['TP_COMPILER_VERBOSE'] = "4"

# import typed_python.compiler.compiler_cache


from typed_python import Entrypoint, Runtime

# Complied function implementing str.decref
@Entrypoint
def string_test(x: str) -> str:
    z = x
    return z


@Entrypoint
def string_test_two(x: str) -> str:
    y = x + " hi2"
    y = y + y
    z = string_test(y)
    # z = x
    return z


import sys


def trace_calls(frame, event, arg):
    if event != "call":
        return
    co = frame.f_code
    func_name = co.co_name
    if func_name == "write":
        # Ignore write() calls from print statements
        return
    func_line_no = frame.f_lineno
    func_filename = co.co_filename
    caller = frame.f_back
    print(func_name)
    # if caller:
    #     caller_line_no = caller.f_lineno
    #     caller_filename = caller.f_code.co_filename
    #     print(f'Call to {func_name} on line {func_line_no} of {func_filename} from line {caller_line_no} of {caller_filename} ({caller})')
    # else:
    #     print(f'Call to {func_name} on line {func_line_no} of {func_filename}, no caller')


if __name__ == "__main__":
    # global RDS
    # global runtime
    # global compiler
    # global native_converter
    # global converter
    # global compiler_cache

    sys.settrace(trace_calls)

    RDS = "runtime.decref_str.7272ea99c1f321ede541a5d770ddceac4eb071e9"
    runtime = Runtime.singleton()
    # converter = runtime.converter
    # compiler = runtime.llvm_compiler
    # native_converter = compiler.converter
    # compiler_cache = runtime.compilerCache

    string_test("bla")
    # the below when uncommented will cause the test to pass?!?!
    # compiler_cache = CompilerCache
    # print('compiler cache has RDS: ', compiler_cache._has_symbol(RDS))
    # print('compiler cache has RDS loaded: ', compiler_cache.symbol_is_loaded(RDS))

    # print(compiler_cache.modules_marked_invalid)
    # print(compiler_cache.modules_marked_valid)
    # print(compiler_cache.loaded_modules)
    # print(compiler_cache.name_to_module_hash)
    # import pdb; pdb.set_trace()
    string_test_two(x="hey")

    string_test_two(x="hey")

    # print(native_converter.external_function_references)
