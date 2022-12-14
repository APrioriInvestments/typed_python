"""
Compiles some modules, to generate a cache.

Then look at what modules, linknames, etc we have.

mark_invalid, mark_valid etc
"""
import networkx as nx
import os
import pdb
import logging


# os.environ['TP_COMPILER_VERBOSE'] = "4"
# this has to be set prior to be TP import
os.environ["TP_COMPILER_CACHE"] = "compiler_cache"

from typed_python import Entrypoint, Runtime, Class, Member, ListOf
from typed_python.compiler.compiler_cache import CacheDependencyGraph

from test_module import string_test, depends_on_string_test_differently, string_test_two


@Entrypoint
def depends_on_string_test(x: str) -> str:
    z = "please "  #
    z += string_test(x) + depends_on_string_test_differently(x)
    return z


# # Complied function implementing str.decref
# @Entrypoint  # delete this function
# def string_test_three(x: str) -> str:
#     y = x + ' hi2' + string_test_two(x) + depends_on_string_test(x)
#     return y

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
    y = y * 2
    z = x
    return z


class Test_Class(Class):
    x = Member(str)

    # @Entrypoint
    def get_x(self):
        return self.x

    def __str__(self):
        return str(self.x)


class Other_Class(Class):
    x = Member(str)
    y = Member(ListOf(str))

    @Entrypoint
    def class_str_method(self):
        z = "t"
        self.y[-1] = z

    @Entrypoint
    def __str__(self):
        return str(self.y)


@Entrypoint
def call_x():
    def inner():
        z = "bla"

    inner()

    bla = Test_Class(x="$")
    return bla.get_x()


def main():
    # depends_on_string_test(x="say")
    # pdb.set_trace()
    string_test_two(x="say")
    # string_test_three(x="hey")  # toggle these two functions
    string_test_five(x="hey")
    string_test_four(x="hey")
    # call_x()
    # print(str(Other_Class(x='bla', y=['foo'])))


if __name__ == "__main__":
    # attempt one to repro (doesn't even hit the relevant codepath)
    main()
    # bail out to check the states
    runtime = Runtime.singleton()
    converter = runtime.converter
    compiler = runtime.llvm_compiler
    native_converter = compiler.converter
    compiler_cache = runtime.compilerCache
    # look at the cache

    # print(compiler_cache.loadedModules)

    # pdb.set_trace()

    # read each module, list all the functions

    for hash, module in compiler_cache.loadedBinarySharedObjects.items():
        print(module, hash)
        for name, hash2 in compiler_cache.nameToModuleHash.items():
            if hash == hash2:
                print("\t", name)
        print()

    dependencies = converter._dependencies

    graph = converter._dependencies._dependencies

    sourceToDest = converter._dependencies._dependencies.sourceToDest

    # cache_graph = CacheDependencyGraph(compiler_cache=compiler_cache)

    # G = cache_graph.get_all_dependencies()

    # print(G)

    # import matplotlib.pyplot as plt
    # pos = nx.spring_layout(G, k=10)
    # nx.draw_networkx_nodes(G, pos)
    # nx.draw_networkx_labels(G, pos, font_size=6)
    # nx.draw_networkx_edges(G, pos)
    # plt.show()

    # import json
    # print(json.dumps(sourceToDest, indent=2))

    # for key, value in sourceToDest.items():
    #     print(converter._link_name_for_identity[key])
    #     print('\tSource2Dest:')
    #     for val in value:
    #         print('\t\t', converter._link_name_for_identity[val])
    #     print()
    #     print('\tGraph Keys:')
    #     print('\n'.join(f'\t\t{converter._link_name_for_identity[val]}' for val in graph.outgoing(key)))
    #     print()
    #     print()

    # pdb.set_trace()
    # print(compiler_cache)
