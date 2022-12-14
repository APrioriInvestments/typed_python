"""
llvm_assert_failing.py

Small functions are failing the assert statement in line 661 of compiler/native_ast_to_llvm.py.

Attempt to repro by checking plausible reasons.

The statement is

                assert target.name not in self.converter._function_definitions, target.name


"""
import os
import pdb

# os.environ['TP_COMPILER_VERBOSE'] = "4"

from typed_python import Entrypoint, Runtime

from test_module import string_test, depends_on_string_test_differently


@Entrypoint
def depends_on_string_test(x: str) -> str:
    z = "please " + string_test(x) + depends_on_string_test_differently(x)
    return z


# # Complied function implementing str.decref
# @Entrypoint
# def string_test(x: str) -> str:
#     y = x + ' hi'
#     return y


if __name__ == "__main__":

    # attempt one to repro (doesn't even hit the relevant codepath)

    # print(string_test(x="say"))
    depends_on_string_test(x="say")
    # string_test(x="say")

    # print(depends_on_string_test(x="say"))
    # print(string_test(x="say"))

    # bail out to check the states
    runtime = Runtime.singleton()
    converter = runtime.converter
    compiler = runtime.llvm_compiler
    native_converter = compiler.converter
    # pdb.set_trace()

    for (
        key,
        value,
    ) in native_converter.func_converter.external_function_references.items():
        print(key, ":", value)

    print("#" * 80)
    for key in native_converter._function_definitions.keys():
        print(key)
