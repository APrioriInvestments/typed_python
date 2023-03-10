from typed_python import Dict

G = Dict(int, float)({1: 2.5})
def f(x):
    return G[x]