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
from typed_python import Class, Member, Held, Final


@Held
class H(Class, Final):
    x = Member(int, nonempty=True)
    y = Member(float, nonempty=True)

    def f(self):
        return self.x + self.y

    def addToX(self, y):
        return self.x + y

    def increment(self):
        self.x += 1
        self.y += 1


class MemSuite:
    def peakmem_held_class_on_heap(self):
        for _ in range(100000):
            H()
