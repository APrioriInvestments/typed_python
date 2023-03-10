import os

# this has to be set prior to be TP import
os.environ["TP_COMPILER_CACHE"] = "compiler_cache"

from typed_python import Dict

G = Dict(int, int)({1: 3})
def f(x):
    return G[x]
