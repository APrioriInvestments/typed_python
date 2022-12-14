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
os.environ["TP_COMPILER_CACHE"] = "compiler_cache"

from typed_python import Entrypoint, Runtime, Class, Member, ListOf

from test_module import string_test, depends_on_string_test_differently, string_test_two
from pycallgraph3 import PyCallGraph
from pycallgraph3 import Config
from pycallgraph3 import GlobbingFilter
from pycallgraph3.output import GraphvizOutput


logging.getLogger("TP_compiler").setLevel(logging.INFO)


@Entrypoint
def depends_on_string_test(x: str) -> str:
    z = "please "  #
    z += string_test(x) + depends_on_string_test_differently(x)
    return z


# # Complied function implementing str.decref
# @Entrypoint
# def string_test(x: str) -> str:
#     y = x + ' hi'
#     return y


# Complied function implementing str.decref
@Entrypoint
def string_test_three(x: str) -> str:
    y = x + " hi2" + string_test_two(x) + depends_on_string_test(x)
    return y


# bump this number to force recompile of this function
@Entrypoint
def string_test_four(x: str) -> str:
    y = x + " hi2" + string_test_two(x)
    y = y + y
    z = x
    return z

    # bump this number to force recompile of this function


@Entrypoint
def string_test_five(x: str) -> str:
    y = x + " hi2" + string_test_four(x)
    y = y + y
    z = x
    return z


class Bla1(Class):
    x = Member(str)

    # @Entrypoint
    def get_x(self):
        return self.x

    def __str__(self):
        return str(self.x)


class strtest2(Class):
    x = Member(str)
    y = Member(ListOf(str))

    @Entrypoint
    def bla3(self):
        # del x
        z = "t"
        self.y.pop()

    @Entrypoint
    def bla4(self):
        # del x
        z = "t"
        self.y[-1] = z
        # del self.y[-1]

    @Entrypoint
    def __str__(self):
        return str(self.x)


@Entrypoint
def call_x():
    def inner():
        z = "bla"

    inner()

    bla = Bla1(x="$")
    return bla.get_x()


def main():

    # # print(string_test(x="say"))
    # depends_on_string_test(x="say")
    # pdb.set_trace()
    # string_test_two(x="say")
    # string_test_three(x="hey")
    string_test_five(x="hey")
    string_test_four(x="hey")

    call_x()

    # bail out to check the states
    runtime = Runtime.singleton()
    converter = runtime.converter
    # compiler = runtime.llvm_compiler
    # native_converter = compiler.converter
    # print(converter._all_defined_names)
    # for key, value in converter._all_defined_names.items():
    #     print(key, type(value))


if __name__ == "__main__":

    # attempt one to repro (doesn't even hit the relevant codepath)
    main()
