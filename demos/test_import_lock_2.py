import os
# this has to be set prior to be TP import
os.environ["TP_COMPILER_CACHE"] = "compiler_cache"
from typed_python import Entrypoint

@Entrypoint
def f(x):
    return x+1

f(1)
