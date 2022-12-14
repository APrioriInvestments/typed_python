from typed_python import Entrypoint

"""used in invalid_module.py."""


@Entrypoint
def f(x):
    return x


@Entrypoint
def g2(x):
    return f(x)


aList = []
bList = []


@Entrypoint
def g1(x):
    return len(aList) + f(x)


"""
Ok, so we need to:
1. track the global vars in a module (?)
2. change how validateGlobalVariables works
 we dump


currently we mark invalid when validateGlobalVariables returns False

and we linkGlobalVariables when we loadFromPath
"""
