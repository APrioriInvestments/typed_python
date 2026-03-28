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
import typed_python.compiler.python_ast_analysis as python_ast_analysis
import typed_python.python_ast as python_ast

from typed_python import Entrypoint, Class, Member, TupleOf, Function, ListOf
from typed_python.test_util import CodeEvaluator


class AClass(Class):
    x = Member(int)
    y = Member(float)
    z = Member(TupleOf(int))

    def f(self) -> float:
        return self.x + self.y

    def f(self, arg) -> float:  # noqa
        return self.x + self.y + arg

    def g(self) -> float:
        return 100

    def add(self, x) -> float:
        return 100 + x

    def loop(self, count: int) -> float:
        i = 0
        res = self.y
        while i < count:
            res = res + self.y
            i = i + 1

        return res


class AChildClass(AClass):
    def g(self) -> float:
        return 1234

    def add(self, x) -> float:
        if isinstance(x, int):
            return 0.2

        return 1234 + x


class TimeSuiteAbsoluteUncached:
    """The TP perf tests, recast as benchmarks."""

    def time_variables_read(self):
        count = 200
        evaluator = CodeEvaluator()

        def makeF(cCount):
            CODE = (
                "def f():\n"
                "    class C:\n"
            ) + (
                "        class B:\n"
                "            pass\n"
            ) * cCount
            moduleDict = {}
            evaluator.evaluateInto(CODE, moduleDict)
            return moduleDict.get('f')

        pyast = python_ast.convertFunctionToAlgebraicPyAst(makeF(count))
        python_ast_analysis.computeVariablesReadByClosures(pyast.body)

    def time_bytes_add(self):
        @Entrypoint
        def bytesAdd(x: bytes):
            i = 0
            res = 0
            while i < len(x):
                j = 0
                while j < len(x):
                    res = res + x[i] + x[j]
                    j = j + 1
                i = i + 1
            return res
        bytesAdd(b" " * 1)  # once to compile
        bytesAdd(b" " * 2000)

    def time_bytes_split(self):
        @Entrypoint
        def splitAndCount(s: bytes, sep: bytes, times: int):
            res = 0
            for i in range(times):
                res += len(s.split(sep))
            return res
        splitAndCount(b"a,"*100, b",", 10)  # once to compile
        splitAndCount(b"a,"*100, b",", 10_000)

    def time_call_method_dispatch(self):
        @Entrypoint
        def addCaller(c: AClass, count: int):
            res = c.add(1) + c.add(2.5)

            for i in range(count - 1):
                res += c.add(1) + c.add(2.5)

            return res
        c = AClass()
        c2 = AChildClass()
        addCaller(c, 200 * 1)
        addCaller(c, 200 * 10000)
        addCaller(c2, 200 * 10000)

    def time_call_closure(self):
        ct = 1000000
        aList1 = ListOf(int)([])

        def makeAppender(l):
            @Function
            def append(y):
                l.append(y)
            return append

        @Entrypoint
        def callManyTimes(c1, ct):
            for i in range(ct):
                c1(i)

        callManyTimes(makeAppender(aList1), 1)
        aList1.clear()
        callManyTimes(makeAppender(aList1), ct)

    def time_assign_functions_with_closure(self):
        @Entrypoint
        def callIt(x):
            y = 10.0
            if x % 2:
                def f(a):
                    return a + y + 1.0
            else:
                def f(a):
                    return a + y + 2.0
            res = 0.0
            for i in range(x):
                x = x + 1
                res += f(i)
            return res

        callIt(1)
        callIt(1000000)

    def time_mtrgs(self):
        def q(x):
            return x-1

        def z(x):
            return q(x)+1

        def f(x):
            return z(g(x - 1)) + z(g(x - 2)) + z(x)

        @Entrypoint
        def g(x):
            if x > 0:
                return z(f(x-1)) * z(2) + f(x-2)
            return 1

        for input in [18, 18.0]:
            for _ in range(1000):
                g(input)

    def time_inlining(self):
        """This one makes no sense in the absolute suite."""
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

        callsF1(10000000)
        callsF4(10000000)
