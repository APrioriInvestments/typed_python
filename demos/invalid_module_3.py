"""used in invalid_module.py."""
from typed_python import Entrypoint


@Entrypoint
def f(x):
    return x


# @Entrypoint
# def g2(x):
#     return f(x)


aList = []


@Entrypoint
def g1(x):
    return len(aList) + f(x)
