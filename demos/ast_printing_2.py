"""Make AST printing less bad.

Use Numba comparison to do it.

Script will take a function, and print the python ast, our native ast, the llvm ast, and the numba versions for that compiled function.

Comparison will give us an idea how best to display.

NB - NOT POSSIBLE, typed_python needs llvmlite <= 0.38, numba >= 0.39 :(
"""

# this needs to be pre-import
import os

os.environ["TP_COMPILER_VERBOSE"] = "4"

from typed_python import Entrypoint, ListOf, Runtime
from typing import Callable

import pdb


def naive_sum(someList, startingInt):
    for x in someList:
        startingInt += x
    return startingInt


# load some vars
runtime = Runtime.singleton()
converter = runtime.converter
compiler = runtime.llvm_compiler
native_converter = compiler.converter


if __name__ == "__main__":

    compiled = Entrypoint(naive_sum)
    specific_list = ListOf(int)(range(10000))

    compiled(specific_list, 0)


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
