# test whether a cached function of sufficiently low complexity can be inlined

import os

# this has to be set prior to be TP import
os.environ["TP_COMPILER_CACHE"] = "compiler_cache"

from typed_python import Entrypoint
from inline_cached_functions_2 import f

@Entrypoint
def multi_f_2():
    x = 0
    for _ in range(10_000_000):
        x = f(x)
    return x


multi_f_2()
