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

from test_module import string_test, depends_on_string_test_differently3
from pycallgraph3 import PyCallGraph
from pycallgraph3 import Config
from pycallgraph3 import GlobbingFilter
from pycallgraph3.output import GraphvizOutput

from typed_python.compiler.runtime import DebuggerVisitor


logging.getLogger("TP_compiler").setLevel(logging.INFO)


@Entrypoint
def depends_on_string_test_2(x: str) -> str:
    z = "please "  #
    z += string_test(x) + depends_on_string_test_differently3(x)
    return z


# # Complied function implementing str.decref
# @Entrypoint
# def string_test(x: str) -> str:
#     y = x + ' hi'
#     return y


# Complied function implementing str.decref
@Entrypoint
def string_test_two(x: str) -> str:
    y = x + " hi2"
    return y


# Complied function implementing str.decref
@Entrypoint
def string_test_three(x: str) -> str:
    y = x + " hi2" + string_test_two(x) + depends_on_string_test_2(x)
    return y


# bump this number to force recompile of this function
@Entrypoint
def string_test_four(x: str) -> str:
    y = x + " hi2"
    return y


class Bla6(Class):
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
    bla = Bla6(x="$")
    return bla.get_x()


def main():

    # # print(string_test(x="say"))
    # depends_on_string_test(x="say")
    # pdb.set_trace()
    string_test_two(x="say")
    string_test_three(x="hey")
    string_test_four(x="hey")

    with DebuggerVisitor():
        call_x()

    str(strtest2(x="5"))

    strtest2(x="5", y=["1", "2"]).bla3()

    strtest2(x="5", y=["1", "2"]).bla4()

    # print(depends_on_string_test(x="say"))
    # print(string_test(x="say"))
    #
    #  print(cal_x())

    # bail out to check the states
    runtime = Runtime.singleton()
    converter = runtime.converter
    compiler = runtime.llvm_compiler
    native_converter = compiler.converter

    # print(native_converter)

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
    #         print('hit')#

    # for row in logging.getLogger('TP_compiler').handlers[0].buffer:
    #     print(f'{row.levelname}-TP Compiler error:{row.getMessage()}')
    logging.getLogger("TP_compiler").error("test error")


def call_graph_filtered(
    function_, output_svg="call_graph.svg", custom_include=None, custom_exclude=None
):
    """Plot the call graph obtained from running <function_>"""
    config = Config()
    config.trace_filter = GlobbingFilter(include=custom_include, exclude=custom_exclude)
    graphviz = GraphvizOutput(output_file=output_svg, output_type="svg")

    with PyCallGraph(output=graphviz, config=config):
        function_()


if __name__ == "__main__":

    # attempt one to repro (doesn't even hit the relevant codepath)

    call_graph_filtered(
        main,
        output_svg="call_graph_no_typewrappers_no_context.svg",
        custom_include=["typed_python.*"],
        custom_exclude=[
            "typed_python.compiler.type_wrappers.*",
            "typed_python.compiler.function_conversion_context.*",
            "typed_python.compiler.expression_conversion_context.*",
        ],
    )
    # pdb.set_trace()
