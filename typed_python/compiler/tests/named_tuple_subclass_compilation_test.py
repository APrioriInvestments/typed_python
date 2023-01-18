import time

from typed_python import Entrypoint, NamedTuple, Function
from flaky import flaky


class T(NamedTuple(x=int, y=float)):
    def __add__(self, other):
        return T(x=self.x + other, y=self.y)


def test_can_construct():
    @Entrypoint
    def getOne(x):
        return T(x=x)

    assert getOne(10).x == 10


def test_can_hit_operators():
    @Entrypoint
    def getOne(x):
        return T() + x

    assert getOne(10).x == 10


def test_subclass_of_named_tuple_compilation():
    NT = NamedTuple(a=int, b=str)

    class X(NT):
        def aMethod(self, x):
            return self.a + x

        @property
        def aProperty(self):
            return self.a + 100

        @staticmethod
        def aStaticMethod(x):
            return x

        @Function
        def aMethodWithType(self, x) -> float:
            return self.a + x

    assert X().aProperty == 100

    @Entrypoint
    def makeAnX(a, b):
        return X(a=a, b=b)

    assert makeAnX(a=1, b="hi").a == 1
    assert makeAnX(a=1, b="hi").b == 'hi'

    @Entrypoint
    def callAnXMethod(x: X, y):
        return x.aMethod(y)

    assert callAnXMethod(X(a=1), 10) == 11

    @Entrypoint
    def callAnXMethodWithType(x: X, y):
        return x.aMethodWithType(y)

    assert callAnXMethod(X(a=1), 10.5) == 11.5

    @Entrypoint
    def callAStaticMethodOnInstance(x: X, y):
        return x.aStaticMethod(y)

    callAStaticMethodOnInstance(X(a=1), 12) == 12

    @Entrypoint
    def callAStaticMethodOnClass(y):
        return X.aStaticMethod(y)

    assert callAStaticMethodOnClass(13) == 13

    @Entrypoint
    def getAProperty(x: X):
        return x.aProperty

    assert getAProperty(X(a=12)) == 112


@flaky(max_runs=3, min_passes=1)
def test_subclass_of_named_tuple_compilation_perf():
    NT = NamedTuple(a=float, b=str)

    class X(NT):
        def aMethod(self, x):
            return self.a + x

        @Entrypoint
        def loop(self, x, times):
            res = 0.0

            for i in range(times):
                res += self.aMethod(x)

            return res

    X(a=123).loop(1.2, 100)

    t0 = time.time()
    X(a=123).loop(1.2, 1000000)
    print(time.time() - t0, " to do 1mm")

    # I get about .001 seconds for this.
    assert time.time() - t0 < .01


def test_bound_method_on_named_tuple():
    class NT(NamedTuple(x=str)):
        @Function
        def f(self, x):
            return x + self.x

    @Entrypoint
    def callIt(n):
        method = n.f

        return method("asdf")

    assert callIt(NT(x="hi")) == "asdfhi"
