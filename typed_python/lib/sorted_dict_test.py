import numpy
import pytest
import time
from typed_python import Entrypoint, ListOf, Dict
from typed_python.lib.sorted_dict import SortedDict


def test_sorted_dict_basic():
    d = SortedDict(int, int)()

    assert str(d) == '{}'
    assert len(d) == 0
    assert 10 not in d
    assert d.first() is None
    assert d.last() is None

    with pytest.raises(KeyError):
        d[10]

    d[10] = 20

    assert d[10] == 20
    assert len(d) == 1
    assert d.first() == 10
    assert d.last() == 10

    del d[10]
    assert len(d) == 0
    assert d.first() is None
    assert d.last() is None

    d[10] = 20
    assert d.pop(10) == 20
    assert len(d) == 0
    assert 10 not in d

    assert d.pop(10, 20) == 20
    assert 10 not in d

    assert d.setdefault(10) == 0
    assert d.setdefault(10, 20) == 0
    assert d.setdefault(11, 20) == 20


def test_sorted_dict_insert():
    d = SortedDict(int, int)()

    d[1] = 1
    d._checkInvariants()

    d[2] = 2
    d._checkInvariants()

    d[3] = 3
    d._checkInvariants()

    del d[3]
    d._checkInvariants()

    del d[2]
    d._checkInvariants()

    del d[1]
    d._checkInvariants()


def test_sorted_dict_size_on_repeat_set():
    d = SortedDict(int, int)()
    for i in range(10):
        for j in range(10):
            d[j] = 10
    assert len(d) == 10


def test_sorted_dict_invariants():
    numpy.random.seed(42)

    for count in [5, 10, 20, 100, 200, 1000, 10000]:
        items = numpy.random.choice(count, size=100000)

        t0 = time.time()
        d = {}
        for x in items:
            if x in d:
                del d[x]
            else:
                d[x] = x
        pythonTime = time.time() - t0

        d = SortedDict(int, int)()

        @Entrypoint
        def checkIt(d, items: ListOf(int)):
            for x in items:
                if x in d:
                    del d[x]
                else:
                    d[x] = x

        t0 = time.time()
        checkIt(d, items)
        sortedDictTime = time.time() - t0

        t0 = time.time()
        checkIt(Dict(int, int)(), items)
        typedDictTime = time.time() - t0

        print(
            "100k took ", sortedDictTime, " with ",
            count, " items and a final size of ", len(d),
            "height=", d.height(),
            "k=", pow(len(d), 1 / d.height()),
            "pytime was ", pythonTime,
            "typed dict time was ", typedDictTime
        )

        d._checkInvariants()


def test_sorted_dict_comparator():
    d = SortedDict(int, int, lambda x, y: y < x)()

    for i in range(10):
        d[i] = i

    d._checkInvariants()

    assert list(d) == list(reversed(range(10)))


def test_sorted_dict_compiled_iter():
    d = SortedDict(int, int)()

    for i in range(10):
        d[i] = i

    def addItUp(d):
        res = 0
        for k in d:
            res += k
        return res

    assert addItUp(d) == 45
    assert Entrypoint(addItUp)(d) == 45
