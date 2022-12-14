from typed_python.test_util import evaluateExprInFreshProcess

import tempfile
import os

xmodule1 = "\n".join(
    [
        "@Entrypoint",
        "def f(x):",
        "    return x",
        "aList=[]",
        "@Entrypoint",
        "def g1(x):",
        "    return len(aList) + f(x)",
    ]
)

xmodule2 = "\n".join(
    [
        "@Entrypoint",
        "def f(x):",
        "    return x",
        "@Entrypoint",
        "def g2(x):",
        "    return f(x)",
    ]
)

VERSION1 = {"x.py": xmodule1}
VERSION2 = {"x.py": xmodule2}

with tempfile.TemporaryDirectory() as compilerCacheDir:
    result, output = evaluateExprInFreshProcess(VERSION1, "x.g1(1)", compilerCacheDir)
    print(output.decode("UTF8"))
    print(bytes(output.decode("UTF8")))

    assert evaluateExprInFreshProcess(VERSION1, "x.g1(1)", compilerCacheDir) == 1
    assert len(os.listdir(compilerCacheDir)) == 1

    # if we try to use 'f', it should work even though we no longer have
    # a defniition for 'g2'
    assert evaluateExprInFreshProcess(VERSION2, "x.f(1)", compilerCacheDir) == 1
    print(evaluateExprInFreshProcess(VERSION2, "x.f(1)", compilerCacheDir))
    print("ah")
    assert len(os.listdir(compilerCacheDir)) == 2

    badCt = 0
    for subdir in os.listdir(compilerCacheDir):
        if "marked_invalid" in os.listdir(os.path.join(compilerCacheDir, subdir)):
            badCt += 1

    assert badCt == 1
