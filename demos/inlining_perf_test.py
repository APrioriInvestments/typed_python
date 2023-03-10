"""
compare performance when swapping the compilation order of two things.
"""
# import networkx as nx
import os
import tempfile
import time
# this has to be set prior to be TP import
# os.environ["TP_COMPILER_CACHE"] = "compiler_cache"

from typed_python import (
    Entrypoint,
    Runtime,
    Class,
    Member,
    ListOf,
    SerializationContext,
)

from typed_python.test_util import evaluateExprInFreshProcess


"""
create a function that uses  something inlinable (e.g. f(x) x+1). run it 1000 times. ensure that there's inlining
create a second function uses f(x), run it 1000 times. compare the performance.
"""

xmodule = "\n".join([
    "@Entrypoint",
    "def f(x):",
    "    return x+1",
    "@Entrypoint",
    "def multi_f_1():",
    "    x = 0",
    "    for _ in range(10_000_000):",
    "        x = f(x)", 
    "    return x",
])


ymodule = "\n".join([
    "from x import f",
    "@Entrypoint",
    "def multi_f_2():",
    "    x = 0",
    "    for _ in range(1000_000):",
    "        x = f(x)", 
    "    return x",
])



if __name__ == '__main__':

    VERSION1 = {'x.py': xmodule}
    VERSION2 = {'x.py': xmodule, 'y.py': ymodule}

    with tempfile.TemporaryDirectory() as compilerCacheDir:
        t0 = time.time()
        res =  evaluateExprInFreshProcess(VERSION1, 'x.multi_f_1()', compilerCacheDir)
        t1 = time.time()

        elapsed_inmodule = t1 - t0
        print(elapsed_inmodule)

        t0 = time.time()
        res =  evaluateExprInFreshProcess(VERSION2, 'y.multi_f_2()', compilerCacheDir)
        t1 = time.time()

        elapsed_inmodule = t1 - t0
        print(elapsed_inmodule)


    @flaky(max_runs=3, min_passes=2)
    def test_cached_functions_inlineable(self):
        xmodule = "\n".join([
            "@Entrypoint",
            "def f(x):",
            "    return x+1",
            "@Entrypoint",
            "def multi_f_1():",
            "    x = 0",
            "    for _ in range(10_000_000):",
            "        x = f(x)", 
            "    return x",
        ])


        ymodule = "\n".join([
            "from x import f",
            "@Entrypoint",
            "def multi_f_2():",
            "    x = 0",
            "    for _ in range(1000_000):",
            "        x = f(x)", 
            "    return x",
        ])

        VERSION1 = {'x.py': xmodule}
        VERSION2 = {'x.py': xmodule, 'y.py': ymodule}

        with tempfile.TemporaryDirectory() as compilerCacheDir:
            t0 = time.time()
            res =  evaluateExprInFreshProcess(VERSION1, 'x.multi_f_1()', compilerCacheDir)
            t1 = time.time()

            elapsed_inmodule = t1 - t0
            print(elapsed_inmodule)

            t0 = time.time()
            res =  evaluateExprInFreshProcess(VERSION2, 'y.multi_f_2()', compilerCacheDir)
            t1 = time.time()

            elapsed_outofmodule = t1 - t0
            assert elapsed_outofmodule < elapsed_inmodule
