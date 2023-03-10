import os
import tempfile

from typed_python.test_util import evaluateExprInFreshProcess



def test_compiler_cache_understands_type_changes():
    xmodule = "\n".join([
        "G = Dict(int, int)({1: 2})",
        "def f(x):",
        "    return G[x]",
    ])
    ymodule = "\n".join([
        "from x import f",
        "@Entrypoint",
        "def g(x):",
        "    return f(x)",
    ])

    VERSION1 = {'x.py': xmodule, 'y.py': ymodule}
    VERSION2 = {'x.py': xmodule.replace("1: 2", "1: 3"), 'y.py': ymodule}
    VERSION3 = {'x.py': xmodule.replace("int, int", "int, float").replace('1: 2', '1: 2.5'), 'y.py': ymodule}

    assert '1: 3' in VERSION2['x.py']

    with tempfile.TemporaryDirectory() as compilerCacheDir:
        assert evaluateExprInFreshProcess(VERSION1, 'y.g(1)', compilerCacheDir) == 2
        assert len(os.listdir(compilerCacheDir)) == 1

        # no recompilation necessary
        assert evaluateExprInFreshProcess(VERSION2, 'y.g(1)', compilerCacheDir) == 3
        assert len(os.listdir(compilerCacheDir)) == 1

        # this forces a recompile
        assert evaluateExprInFreshProcess(VERSION3, 'y.g(1)', compilerCacheDir) == 2.5
        assert len(os.listdir(compilerCacheDir)) == 2


def test_compiler_cache_handles_changed_types():
    xmodule1 = "\n".join([
        "@Entrypoint",
        "def f(x):",
        "    return x",
        "aList=[]",
        "@Entrypoint",
        "def g1(x):",
        "    return len(aList) + f(x)",
    ])

    xmodule2 = "\n".join([
        "@Entrypoint",
        "def f(x):",
        "    return x",
        "@Entrypoint",
        "def g2(x):",
        "    return f(x)",
    ])

    VERSION1 = {'x.py': xmodule1}
    VERSION2 = {'x.py': xmodule2}

    with tempfile.TemporaryDirectory() as compilerCacheDir:
        assert evaluateExprInFreshProcess(VERSION1, 'x.g1(1)', compilerCacheDir) == 1
        assert len(os.listdir(compilerCacheDir)) == 1

        # if we try to use 'f', it should work even though we no longer have
        # a defniition for 'g2'
        assert evaluateExprInFreshProcess(VERSION2, 'x.f(1)', compilerCacheDir) == 1
        assert len(os.listdir(compilerCacheDir)) == 2

        badCt = 0
        for subdir in os.listdir(compilerCacheDir):
            if 'marked_invalid' in os.listdir(os.path.join(compilerCacheDir, subdir)):
                badCt += 1

        assert badCt == 1

    
test_compiler_cache_handles_changed_types()
test_compiler_cache_understands_type_changes()