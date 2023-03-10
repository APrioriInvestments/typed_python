import os

# this has to be set prior to be TP import
os.environ["TP_COMPILER_CACHE"] = "compiler_cache"

from typed_python import Entrypoint
from compiler_understands_changes import f

@Entrypoint
def g(x):
    return f(x)


if __name__ == "__main__":
    print(g(1))