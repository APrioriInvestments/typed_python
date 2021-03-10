import unittest
import math
import sys
import time
from flaky import flaky

from typed_python import Class, Final, ListOf, Held, Member, PointerTo, pointerTo
from typed_python.test_util import currentMemUsageMb


@Held
class H(Class, Final):
    x = Member(int)
    y = Member(float)

    def f(self):
        return self.x + self.y

    def increment(self):
        self.x += 1
        self.y += 1


class TestHeldClassInterpreterSemantics(unittest.TestCase):
    def test_list_of_held_class_item_type(self):
        assert sys.gettrace() is None

        aList = ListOf(H)()
        aList.resize(1)

        h = aList[0]

        # h is a copy of aList[0], so writing to it shouldn't affect the list
        h.x = 10
        assert aList[0].x == 0

        # we are able to write directly into the list
        aList[0].x = 20
        assert aList[0].x == 20

        # we can then copy out of the list and it worked
        h = aList[0]
        assert h.x == 20

        assert sys.gettrace() is None

    def test_construct_held_class_on_heap_doesnt_leak(self):
        mb = currentMemUsageMb()
        for _ in range(100000):
            H()
        assert currentMemUsageMb() - mb < .1

    def test_can_call_held_class_in_interpreter(self):
        @Held
        class H(Class, Final):
            def __call__(self, a):
                return (type(self), a)

        assert H()(10) == (H, 10)

    def test_can_construct_held_class_on_heap(self):
        h = H(x=10, y=20)
        assert type(h) is H
        assert h.x == 10

        h.x = 20

        assert h.x == 20

    def test_can_construct_held_class_on_heap_with_init(self):
        @Held
        class H(Class, Final):
            x = Member(int)

            def __init__(self):
                self.x = 100

        h = H()
        assert h.x == 100

    def test_held_class_methods_get_passed_refs(self):
        @Held
        class H(Class, Final):
            def f(self):
                return type(self)

        assert H().f() is H

    def test_multiple_refs_simultaneously(self):
        aList = ListOf(H)()
        aList.resize(2)

        assert aList[0].x + aList[1].x == 0
        x, y = aList[0], aList[1]

        assert isinstance(x, H)
        assert isinstance(y, H)

    def test_refs_passed_to_functions(self):
        aList = ListOf(H)()
        aList.resize(1)

        def f(h, y):
            h.x = y
            assert type(h) is H

        f(aList[0], 20)

        assert aList[0].x == 20

        h = aList[0]
        f(h, 22)
        assert h.x == 22
        assert aList[0].x == 20

        p = pointerTo(aList[0])

        assert type(p) is PointerTo(H)

        p.x.set(10)
        assert aList[0].x == 10

        def inner(h):
            return h

        def outer(h, val):
            y = inner(h)

            y.x = val

        outer(aList[0], 30)
        assert aList[0].x == 30

    @flaky(max_runs=3, min_passes=1)
    def test_nested_ref_call_cost(self):
        # verify that the python interpreter is not
        # slowed down too much while HeldClass references are in play.
        # in particular, we shouldn't be slower if there are multiple
        # HeldClass references outstanding at once.
        t0 = time.time()
        for i in range(100000):
            pass
        normalTime = time.time() - t0

        def f(l, h, depth):
            t0 = time.time()
            for i in range(100000):
                pass
            elapsed = time.time() - t0

            assert elapsed < normalTime * 20

            if depth < 100:
                f(l, l[0], depth+1)

        aList = ListOf(H)()
        aList.resize(1)

        f(aList, aList[0], 0)

    def test_access_held_class_functions(self):
        aList = ListOf(H)()
        aList.resize(1)

        aList[0].x = 10
        assert aList[0].x == 10

        aList[0].increment()

        assert aList[0].x == 11

    def test_held_class_held_classes(self):
        @Held
        class H2(Class, Final):
            h = Member(H)

        aList = ListOf(H2)()
        aList.resize(1)

        aList[0].h.x += 1
        assert aList[0].h.x == 1

        h = aList[0].h
        h.x = 5
        assert aList[0].h.x == 1

        aList[0].h = h
        assert aList[0].h.x == 5

    def test_held_class_ref_conversion_with_existing_trace_function(self):
        # HeldClass uses 'sys.settrace' internally to know when the temporary ref
        # needs to get converted to a proper instance. We need to check that this doesn't
        # conflict with an existing reference
        traced = []

        def traceFun(*args):
            traced.append(args[1])
            return traceFun

        aList = ListOf(H)()
        aList.resize(5)

        def tryIt(l):
            objects = []
            for i in range(len(l)):
                objects.append(l[i])
            return objects

        try:
            sys.settrace(traceFun)
            objects1 = tryIt(aList)
            sys.settrace(None)
            traced1 = traced
            traced = []

            sys.settrace(traceFun)
            tryIt([1] * 5)
            sys.settrace(None)
            traced2 = traced
            traced = []

        finally:
            sys.settrace(None)

        assert traced1 == traced2
        assert type(objects1[0]) is H

    def test_held_class_instance_conversion_doesnt_leak(self):
        aList = ListOf(H)()
        aList.resize(1)
        x = aList[0]

        m0 = currentMemUsageMb()

        for _ in range(100000):
            x = aList[0]

        assert isinstance(x, H)

        leakSize = currentMemUsageMb() - m0

        assert leakSize < .1

    def test_pointer_to_held_class(self):
        h = H()

        p = pointerTo(h)

        h.x = 10
        assert p.x.get() == 10

    def test_contains_operator(self):
        @Held
        class H(Class, Final):
            x = Member(int)

            def __contains__(self, other):
                return other == self.x

        assert 10 in H(x=10)
        assert 11 not in H(x=10)

    def test_len_operator(self):
        @Held
        class H(Class, Final):
            x = Member(int)

            def __len__(self):
                return self.x

        assert len(H(x=10)) == 10

    def test_getitem(self):
        @Held
        class H(Class, Final):
            x = Member(int)

            def __getitem__(self, y):
                return self.x + y

        assert H(x=10)[20] == 30

    def test_setitem(self):
        @Held
        class H(Class, Final):
            x = Member(int)

            def __setitem__(self, x, y):
                if x == 0:
                    self.x = y

        h = H(x=10)
        h[0] = 20
        assert h.x == 20
        h[1] = 30
        assert h.x == 20

    def test_inquiry(self):
        @Held
        class H(Class, Final):
            x = Member(int)

        assert H()

        @Held
        class H2(Class, Final):
            x = Member(int)

            def __bool__(self):
                return self.x

        assert H2(x=1)
        assert not H2(x=0)

    def test_ceil(self):
        @Held
        class H(Class, Final):
            def __ceil__(self):
                return 123

        assert math.ceil(H()) == 123

    def test_add_operator(self):
        @Held
        class H(Class, Final):
            x = Member(int)

            def __add__(self, other):
                return self.x + other

        assert H(x=12) + 1 == 13
