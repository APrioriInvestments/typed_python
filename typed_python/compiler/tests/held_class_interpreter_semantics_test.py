import unittest
import sys

from typed_python import Class, Final, ListOf, Held, Member, refTo, RefTo
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

        def f(x):
            assert type(x) is H

        f(aList[0])

        # refTo is a 'C' function, so we can pass it
        r = refTo(aList[0])

        assert type(r) is RefTo(H)

        r.x = 10
        assert aList[0].x == 10

    def test_access_held_class_functions(self):
        aList = ListOf(H)()
        aList.resize(1)

        aList[0].increment()

        assert aList[0].x == 1

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
