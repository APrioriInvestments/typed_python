"""
Benchmarks:
organised into suites
can have setup and teardown methods
tests start with time, mem, track, or peakmem.
Four classes of timing test:
- Cached/Uncached  - i.e is the compiler cache turned on.
- Absolute/Relative - i.e do we measure absolute time taken (with time_*)
    or the relative time taken (with track_*)


TODO:
    - Extend suite to better track real usage
    - Tests to compare the performance of already-cached code (rather than cold start)

"""
import tempfile

from typed_python.test_util import evaluateExprInFreshProcess


class TimeSuiteCached:
    def time_cache_handles_changed_types(self):
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
            assert evaluateExprInFreshProcess(VERSION2, 'x.f(1)', compilerCacheDir) == 1
            assert evaluateExprInFreshProcess(VERSION1, 'x.g1(1)', compilerCacheDir) == 1
