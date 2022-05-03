#   Copyright 2017-2021 typed_python Authors
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from typed_python import (
    TupleOf, ListOf, Dict, Class, Member, NamedTuple, ConstDict, Tuple, Set,
    Alternative, Forward, OneOf, deepcopy, deepcopyContiguous, totalBytesAllocatedInSlabs,
    deepBytecountAndSlabs, refcount, totalBytesAllocatedOnFreeStore
)
from typed_python.test_util import currentMemUsageMb
import time
import numpy


def checkDeepcopySimple(obj, requiresSlab, objectIsSlabRoot=False):
    initSlabBytes = totalBytesAllocatedInSlabs()

    obj2 = deepcopyContiguous(obj, trackInternalTypes=True)

    assert obj2 == obj

    if requiresSlab:
        assert totalBytesAllocatedInSlabs() > initSlabBytes

        slabs = deepBytecountAndSlabs(obj2)[1]
        assert len(slabs) == 1
        if objectIsSlabRoot:
            assert slabs[0].allocCount() >= 1
            assert slabs[0].allocIsAlive(0)
            assert slabs[0].extractObject(0) == obj2
        slabs = None

        # check that deepcopying gets rid of the slab reference
        temp = deepcopy(obj2)
        assert not deepBytecountAndSlabs(temp)[1]
        temp = None
    else:
        assert totalBytesAllocatedInSlabs() == initSlabBytes
        assert len(deepBytecountAndSlabs(obj2)[1]) == 0

    obj2 = None

    assert totalBytesAllocatedInSlabs() == initSlabBytes

    # check that deepcopyContiguous doesn't leak
    t0 = time.time()
    m0 = currentMemUsageMb()

    while time.time() - t0 < .1:
        obj2 = deepcopyContiguous(obj)

        if requiresSlab:
            assert totalBytesAllocatedInSlabs() > initSlabBytes
        else:
            assert totalBytesAllocatedInSlabs() == initSlabBytes

        obj2 = None
        assert totalBytesAllocatedInSlabs() == initSlabBytes

    assert currentMemUsageMb() - m0 < 1.0

    # check that deepcopy doesn't leak
    t0 = time.time()
    m0 = currentMemUsageMb()

    while time.time() - t0 < .1:
        assert totalBytesAllocatedInSlabs() == initSlabBytes
        obj2 = deepcopy(obj)
        assert totalBytesAllocatedInSlabs() == initSlabBytes

    assert currentMemUsageMb() - m0 < 1.0


def test_deepcopy_pod_tuple():
    checkDeepcopySimple(Tuple(int, int)(), False)


def test_deepcopy_named_tuple():
    checkDeepcopySimple(NamedTuple(x=int, y=str)(x=10, y="hi"), True)


def test_deepcopy_listof():
    checkDeepcopySimple(ListOf(int)(range(1000)), True, True)


def test_deepcopy_setof():
    checkDeepcopySimple(Set(int)(range(1000)), True, True)


def test_deepcopy_tupleof():
    checkDeepcopySimple(TupleOf(int)(range(1000)), True, True)


def test_deepcopy_oneof():
    checkDeepcopySimple(
        TupleOf(OneOf(None, "hi", int))(list(range(1000)) + ["hi", None, "hi", 3]),
        True
    )


def test_deepcopy_pystring():
    checkDeepcopySimple("hi", False, True)


def test_deepcopy_tupleof_str():
    checkDeepcopySimple(TupleOf(str)(["hi"]), True, True)


def test_deepcopy_empty_tup():
    checkDeepcopySimple((), False)


def test_deepcopy_pytuple():
    checkDeepcopySimple((1, 2, 3), False)


def test_deepcopy_pylist():
    checkDeepcopySimple([1, 2, 3], False)


def test_deepcopy_pydict():
    checkDeepcopySimple({1: 2, 3: 4}, False)


def test_deepcopy_pyset():
    checkDeepcopySimple({1, 2, 3, 4}, False)


def test_untyped_holding_typed():
    checkDeepcopySimple([ListOf(int)(range(100))], True)


def test_deepcopy_Class():
    class C(Class):
        x = Member(ListOf(int))

        def __eq__(self, other):
            if not isinstance(other, C):
                return False

            return self.x == other.x

    checkDeepcopySimple(C(x=ListOf(int)(range(1000))), True, True)


def test_deepcopy_PySubClass():
    class C(NamedTuple(x=int)):
        pass

    checkDeepcopySimple(ListOf(C)([C(x=10)]), True, True)


def test_deepcopy_PySubClassWithSlabStuff():
    class C(NamedTuple(x=ListOf(int))):
        pass

    checkDeepcopySimple(C(x=ListOf(int)([10])), True, False)


def test_deepcopy_class():
    class C:
        def __init__(self, x):
            self.x = x

        def __eq__(self, other):
            return self.x == other.x

    checkDeepcopySimple(C(10), False, True)


def test_holding_interior_item_keeps_slab_alive():
    initSlabBytes = totalBytesAllocatedInSlabs()

    listOfLists = ListOf(ListOf(int))([[1, 2, 3], [4, 5, 6]])

    listOfLists2 = deepcopyContiguous(listOfLists)

    assert totalBytesAllocatedInSlabs() > 0

    lst = listOfLists2[0]
    listOfLists2 = None

    assert totalBytesAllocatedInSlabs() > 0

    lst = None  # noqa

    assert totalBytesAllocatedInSlabs() == initSlabBytes


def test_deepcopy_with_external_pyobj_doesnt_leak():
    mem = currentMemUsageMb()

    for passIx in range(100):
        deepcopyContiguous(" " * 1024 * 1024 + "_" * (passIx+1))

    assert currentMemUsageMb() - mem < 10


def test_deepcopy_with_tuple_doesnt_leak():
    mem = currentMemUsageMb()

    for passIx in range(10000):
        deepcopyContiguous(("_" * (passIx+1),))

    assert currentMemUsageMb() - mem < 10


def test_deepcopy_with_mutually_recursive_objects_doesnt_leak():
    mem = currentMemUsageMb()

    l1 = [1, 2, 3, 4]
    l2 = [1, 2, 3, 4, l1]
    d1 = {1: 2, 3: l2}
    l1.append(d1)

    for passIx in range(10000):
        deepcopyContiguous(l1)

    assert currentMemUsageMb() - mem < 10


def test_deepcopy_with_Class():
    initSlabBytes = totalBytesAllocatedInSlabs()

    class C(Class):
        x = Member(int)

    class D(C):
        y = Member(int)

    cCopy = deepcopyContiguous(C(x=23))

    assert cCopy.x == 23
    assert type(cCopy) is C

    dCopy = deepcopyContiguous(NamedTuple(c=C)(c=D(x=23, y=24))).c

    assert type(dCopy) is D
    assert dCopy.x == 23
    assert dCopy.y == 24

    cCopy = None
    dCopy = None

    assert totalBytesAllocatedInSlabs() == initSlabBytes


def test_deepcopy_with_Dict():
    initSlabBytes = totalBytesAllocatedInSlabs()
    mem = currentMemUsageMb()

    v = Dict(str, str)()

    deepcopyContiguous(v)

    v["hi"] = "bye"
    for i in range(500):
        v[str(i)] = str(i) + "out"
        v2 = deepcopyContiguous(v)
        assert v2 == v

    v = None
    v2 = None

    assert currentMemUsageMb() - mem < 10
    assert totalBytesAllocatedInSlabs() == initSlabBytes


def test_deepcopy_with_Set():
    initSlabBytes = totalBytesAllocatedInSlabs()
    mem = currentMemUsageMb()

    v = Set(str)()

    for i in range(500):
        v.add(str(i))
        v2 = deepcopyContiguous(v)
        assert v2 == v

    v = None
    v2 = None

    assert currentMemUsageMb() - mem < 10
    assert totalBytesAllocatedInSlabs() == initSlabBytes


def test_deepcopy_with_ConstDict():
    initSlabBytes = totalBytesAllocatedInSlabs()
    mem = currentMemUsageMb()

    v = ConstDict(str, str)()

    for i in range(500):
        v = v + {str(i): str(i)}
        v2 = deepcopyContiguous(v)
        assert v2 == v

    v = None
    v2 = None

    assert currentMemUsageMb() - mem < 10
    assert totalBytesAllocatedInSlabs() == initSlabBytes


def test_deepcopy_with_Alternative():
    initSlabBytes = totalBytesAllocatedInSlabs()
    mem = currentMemUsageMb()

    A = Forward("A")
    A = A.define(Alternative("A", T=dict(x=A, y=A), E=dict(x=int)))

    a = A.E(x=10)

    for i in range(500):
        a = A.T(x=a, y=a)
        aCheck = deepcopyContiguous(a)
        assert totalBytesAllocatedInSlabs() < 500 * (i+1) + 1024

        for j in range(i):
            assert aCheck.matches.T
            aCheck = aCheck.x
        aCheck = None

    assert currentMemUsageMb() - mem < 10
    assert totalBytesAllocatedInSlabs() == initSlabBytes


def test_see_slabs():
    list1 = deepcopyContiguous(ListOf(int)(range(1000)), trackInternalTypes=True)
    list2 = deepcopyContiguous(ListOf(int)(range(1000)), trackInternalTypes=True)

    bytecount, slabs = deepBytecountAndSlabs([list1, list2])
    assert bytecount < 1000
    assert len(slabs) == 2

    for slab in slabs:
        assert slab.allocCount()


def test_deepcopy_class_with_dual_references():
    class C(Class):
        x = Member(Dict(int, int))
        y = Member(Dict(int, int))

    l = Dict(int, int)()
    c = C(x=l, y=l)
    l = None

    assert refcount(c.x) == 3

    c2 = deepcopy(c)

    assert refcount(c2.x) == 3


def test_bytes_on_free_store_basic():
    bytecount0 = totalBytesAllocatedOnFreeStore()

    x = ListOf(int)(range(1000))

    bytecount1 = totalBytesAllocatedOnFreeStore()

    x.resize(2000)

    bytecount2 = totalBytesAllocatedOnFreeStore()

    x.resize(500)
    x.reserve(500)

    bytecount3 = totalBytesAllocatedOnFreeStore()

    x = None

    assert bytecount1 > bytecount0
    assert bytecount2 > bytecount1
    assert bytecount1 > bytecount3 > bytecount0
    assert bytecount0 == totalBytesAllocatedOnFreeStore()


def test_deepcopy_perf():
    x = ListOf(str)()

    for ix in range(1000000):
        x.append(str(ix))

    from typed_python import SerializationContext

    t0 = time.time()
    deepcopyContiguous(x, trackInternalTypes=False)
    t1 = time.time()
    deepcopy(x)
    t2 = time.time()

    sc = SerializationContext()
    sc.deserialize(sc.serialize(x))
    t3 = time.time()

    print("serialization", t3 - t2)
    print("deepcopy", t2 - t1)
    print("deepcopyContiguous", t1 - t0)


def test_deepcopy_typeMap():
    aTup = ([1], [2])

    def mapper(aLst):
        return aLst + aLst

    res = deepcopy(aTup, typeMap={list: mapper})

    assert res[0] == [1, 1]
    assert res[1] == [2, 2]


def test_deepcopy_typeMap_typed():
    class C(Class):
        x = Member(int)

    def mapper(i):
        return i + 1

    res = deepcopy(C(x=10), typeMap={int: mapper})

    assert res.x == 11


def test_deepcopy_typeMap_baseclass():
    class C(object):
        def __init__(self, x):
            self.x = x

    class D(C):
        pass

    def mapper(c):
        return type(c)(x=c.x + 1)

    res = deepcopy(D(x=10), typeMap={C: mapper})

    assert res.x == 11


def test_deepcopy_empty_alternatives():
    X = Alternative("X", A=dict(), B=dict())
    Y = Alternative("Y", C=dict(x=X))
    y = Y.C(x=X.B())
    print(deepcopy(y))


def test_deepcopy_class_subclass():
    class Base(Class):
        pass

    class Child(Base):
        child = Member(Tuple(int, int, int, int))

    for i in range(10):
        x = ListOf(Base)([Child() for _ in range(i)])
        deepcopyContiguous(x)


def test_deepcopy_tuple_of_strings():
    for i in range(100):
        x = ListOf(OneOf(str, float))(['h'] * i)
        deepcopyContiguous(x)


def test_deepcopy_numpy_array():
    x = numpy.array([1, 2, 3])

    y = deepcopy(x)

    assert x is not y
    assert x.tolist() == y.tolist()

    x[1] = 100
    assert y[1] == 2


class ClassWithReduce:
    def __init__(self, state):
        self.state = state

    def __reduce__(self):
        return (ClassWithReduce, (None,), (self.state,))

    def __setstate__(self, state):
        self.state = state[0]


def test_deepcopy_class_with_custom_reduce():
    c = ClassWithReduce("state")

    c2 = deepcopy(c)

    assert c is not c2


class ClassWithCustomSetState:
    def __init__(self, state):
        self.state = state

    def __reduce__(self):
        return (ClassWithReduce, (None,), (self.state,), None, None, ClassWithCustomSetState.setState)

    def setState(self, state):
        self.state = state[0]


def test_deepcopy_class_with_custom_setState():
    c = ClassWithCustomSetState("state")

    c2 = deepcopy(c)

    assert c is not c2


class ClassWithIterators:
    def __init__(self):
        self.items = []
        self.kvs = {}

    def __reduce__(self):
        return (ClassWithIterators, (), None, ['1', '2'], [('a', 'b')])

    def extend(self, x):
        self.items.extend(x)

    def __setitem__(self, x, y):
        self.kvs[x] = y


def test_deepcopy_class_with_custom_reduce_iterators():
    c = ClassWithIterators()

    c2 = deepcopy(c)

    assert c is not c2

    assert c2.items == ['1', '2']
    assert c2.kvs == {'a': 'b'}
