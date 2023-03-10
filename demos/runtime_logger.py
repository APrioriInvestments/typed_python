"""
llvm_assert_failing.py

Small functions are failing the assert statement in line 661 of compiler/native_ast_to_llvm.py.

Attempt to repro by checking plausible reasons.

The statement is

                assert target.name not in self.converter._function_definitions, target.name


"""
import os
import pdb
import logging
# os.environ['TP_COMPILER_VERBOSE'] = "4"
os.environ["TP_COMPILER_CACHE"] = 'compiler_cache'

from typed_python import Entrypoint, Runtime, Class

from test_module import string_test, depends_on_string_test_differently



logging.getLogger('TP_compiler').setLevel(logging.WARN)

@Entrypoint
def depends_on_string_test(x: str) -> str:
    z = 'please '#
    z += string_test(x) + depends_on_string_test_differently(x)
    return z


# # Complied function implementing str.decref
# @Entrypoint
# def string_test(x: str) -> str:
#     y = x + ' hi'
#     return y


# Complied function implementing str.decref
@Entrypoint
def string_test_two(x: str) -> str:
    y = x + ' hi2'
    return y

# Complied function implementing str.decref
@Entrypoint
def string_test_three(x: str) -> str:
    y = x + ' hi2' + string_test_two(x) + depends_on_string_test(x)
    return y

# bump this number to force recompile of this function
@Entrypoint
def string_test_four4(x: str) -> str:
    y = x + ' hi2'
    return y

if __name__ == "__main__":

    # attempt one to repro (doesn't even hit the relevant codepath)

    # # print(string_test(x="say"))
    # depends_on_string_test(x="say")
    # pdb.set_trace()
    string_test_two(x="say")
    string_test_three(x="hey")
    string_test_four4(x="hey")

    # print(depends_on_string_test(x="say"))
    # print(string_test(x="say"))


    # bail out to check the states
    runtime = Runtime.singleton()
    converter = runtime.converter
    compiler = runtime.llvm_compiler
    native_converter = compiler.converter
    print(native_converter)
    # pdb.set_trace()

    # print('FunctionConverter external function references')
    # for key, value in native_converter.func_converter.external_function_references.items():
    #      print(key,':',  value)


    # print("#"*80)
    # print('FunctionConverter external function references')
    # for key, value in native_converter.func_converter.external_function_references.items():
    #      print(key,':',  value)

    # print("#"*80)
    # for key in native_converter._function_definitions.keys():
    #      print(key)
#     from typed_python.test_util import evaluateExprInFreshProcess
#     import tempfile
#     # @Entrypoint
#     # def f(x):
#     #     return x + 1

#     # assert f(10) == 11
#     MAIN_MODULE = """
# @Entrypoint
# def f(x):
#     return x + 1

#     """

#     with tempfile.TemporaryDirectory() as compilerCacheDir:
#         # pdb.set_trace()
#         assert evaluateExprInFreshProcess({'x.py': MAIN_MODULE}, 'x.f(10)', compilerCacheDir) == 11
#         print('hit')
    logging.getLogger('TP_compiler').error('test error')

    # pdb.set_trace()
