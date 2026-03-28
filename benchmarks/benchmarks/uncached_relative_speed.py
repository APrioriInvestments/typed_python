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
import threading
import time

from typed_python import Entrypoint, Dict


class TimeSuiteRelativeUncached:
    unit = 'ratio'

    def track_inlining_ratio(self):
        def f1(x):
            return f2(x)

        def f2(x):
            return f3(x)

        def f3(x):
            return f4(x)

        def f4(x: int):
            return x

        @Entrypoint
        def callsF1(times: int):
            res = 0.0
            for i in range(times):
                res += f1(i)
            return res

        @Entrypoint
        def callsF4(times: int):
            res = 0.0
            for i in range(times):
                res += f4(i)
            return res

        # prime the compilation
        callsF4(1)
        callsF1(1)

        t0 = time.time()
        callsF1(10000000)
        t1 = time.time()
        callsF4(10000000)
        t2 = time.time()

        callsDeeply = t1 - t0
        callsShallowly = t2 - t1
        return callsDeeply / callsShallowly

    def track_star_kwarg_intermediate_ratio(self):
        def f(x, y):
            return x + y

        def g(**kwargs):
            return f(**kwargs)

        @Entrypoint
        def sumUsingG(a: int):
            res = 0.0
            for i in range(a):
                res += g(x=2, y=i)
            return res

        @Entrypoint
        def sumUsingF(a: int):
            res = 0.0
            for i in range(a):
                res += f(x=2, y=i)
            return res

        sumUsingF(10)
        sumUsingG(10)

        t0 = time.time()
        sumUsingG(1000000)
        elapsedG = time.time() - t0

        t0 = time.time()
        sumUsingF(1000000)
        elapsedF = time.time() - t0

        return elapsedF / elapsedG

    def track_dict_read_write_multicore_ratio(self):
        @Entrypoint
        def dict_setmany(d, count, passes):
            for _ in range(passes):
                for i in range(count):
                    if i in d:
                        d[i] += i
                    else:
                        d[i] = i

        # make sure we compile this immediately
        aDictToForceCompilation = Dict(int, int)()
        dict_setmany(aDictToForceCompilation, 1, 1)

        # test it with one core
        t0 = time.time()
        aDict = Dict(int, int)()
        dict_setmany(aDict, 10000, 100)
        t1 = time.time()

        # test it with 2 cores
        threads = [threading.Thread(target=dict_setmany, args=(Dict(int, int)(), 10000, 100)) for _ in range(2)]
        t2 = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        t3 = time.time()

        slowdownRatio = (t3 - t2) / (t1 - t0)

        return slowdownRatio
