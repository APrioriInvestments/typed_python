import os

# this has to be set prior to be TP import
os.environ["TP_COMPILER_CACHE"] = "compiler_cache"

from typed_python import Entrypoint
from debug_sigsegv_module_1 import f

@Entrypoint
def g(x):
    return f(x)


if __name__ == '__main__':
    # import pdb; pdb.set_trace()
    g(1)
