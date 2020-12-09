#   Copyright 2017-2019 typed_python Authors
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

import numpy
import psutil
import time
import typed_python._types as _types
import unittest

from flaky import flaky
from typed_python import ListOf, Function, TupleOf, OneOf, Compiled, Entrypoint
from typed_python import UInt8, UInt16, UInt32, Class, Final, Member


class TestListOfCompilation(unittest.TestCase):
    def checkFunction(self, f, argsToCheck):
        f_fast = Compiled(f)

        t_py = 0.0
        t_fast = 0.0
        for a in argsToCheck:
            t0 = time.time()
            fastval = f_fast(*a)
            t1 = time.time()
            slowval = f(*a)
            t2 = time.time()

            t_py += t2-t1
            t_fast += t1-t0

            self.assertEqual(fastval, slowval)
        return t_py, t_fast

    def test_list_of_list_refcounts(self):
        @Compiled
        def f(x: ListOf(ListOf(int)), z: bool):
            if z:
                y = x[0]  # noqa

                return x

            return 10

        aList = ListOf(ListOf(int))()
        aList.resize(1)

        interiorList = aList[0]

        rc = _types.refcount(interiorList)

        f(aList, True)

        self.assertEqual(rc, _types.refcount(interiorList))

    def test_list_of_float(self):
        def f(x: ListOf(float), y: ListOf(float)) -> float:
            j = 0
            res = 0.0
            i = 0

            while j < len(y):
                i = 0
                while i < len(x):
                    res = res + x[i] * y[j]
                    i = i + 1
                j = j + 1

            return res

        aListOfFloat = ListOf(float)(list(range(1000)))
        aListOfFloat2 = ListOf(float)(list(range(1000)))

        self.assertEqual(_types.refcount(aListOfFloat), 1)

        t_py, t_fast = self.checkFunction(f, [(aListOfFloat, aListOfFloat2)])

        self.assertEqual(_types.refcount(aListOfFloat), 1)

        # I get around 150x
        self.assertTrue(t_py / t_fast > 50.0)

        print(t_py / t_fast, " speedup")

    def test_list_negative_indexing(self):
        @Compiled
        def getitem(x: ListOf(int), y: int):
            return x[y]

        for listToCheck in [[], [1], [1, 2], [1, 2, 3]]:
            for ix in [-2, -1, 0, 1, 2]:
                try:
                    val = listToCheck[ix]
                except IndexError:
                    val = None

                if val is not None:
                    self.assertEqual(val, getitem(listToCheck, ix))
                else:
                    with self.assertRaises(Exception):
                        getitem(listToCheck, ix)

    def test_list_passing(self):
        @Compiled
        def f(x: ListOf(int)) -> int:
            return 0

        self.assertEqual(f((1, 2, 3)), 0)

    def test_list_len(self):
        @Compiled
        def f(x: ListOf(int)) -> int:
            return len(x)

        self.assertEqual(f((1, 2, 3)), 3)

    def test_list_assign(self):
        @Compiled
        def f(x: ListOf(int)) -> ListOf(int):
            y = x
            return y

        t = ListOf(int)((1, 2, 3))

        self.assertEqual(f(t), t)

        self.assertEqual(_types.refcount(t), 1)

    def test_list_indexing(self):
        @Compiled
        def f(x: ListOf(int), y: int) -> int:
            return x[y]

        self.assertEqual(f((1, 2, 3), 1), 2)

        with self.assertRaises(Exception):
            f((1, 2, 3), 1000000000)

    def test_list_refcounting(self):
        @Function
        def f(x: ListOf(int), y: ListOf(int)) -> ListOf(int):
            return x

        for compileIt in [False, True]:
            if compileIt:
                f = Compiled(f)

            intTup = ListOf(int)(list(range(1000)))

            self.assertEqual(_types.refcount(intTup), 1)

            res = f(intTup, intTup)

            self.assertEqual(_types.refcount(intTup), 2)

            res = None  # noqa: F841

            self.assertEqual(_types.refcount(intTup), 1)

    def test_list_of_adding(self):
        T = ListOf(int)

        @Compiled
        def f(x: T, y: T) -> T:
            return x + y

        t1 = T((1, 2, 3))
        t2 = T((3, 4))

        res = f(t1, t2)

        self.assertEqual(_types.refcount(res), 1)
        self.assertEqual(_types.refcount(t1), 1)
        self.assertEqual(_types.refcount(t2), 1)

        self.assertEqual(res, t1+t2)

    def test_list_of_list_refcounting(self):
        T = ListOf(int)
        TT = ListOf(T)

        @Compiled
        def f(x: TT) -> TT:
            return x + x + x

        t1 = T((1, 2, 3))
        t2 = T((4, 5, 5))

        aTT = TT((t1, t2))

        fRes = f(aTT)

        self.assertEqual(fRes, aTT+aTT+aTT)
        self.assertEqual(_types.refcount(aTT), 1)
        self.assertEqual(_types.refcount(fRes), 1)

        fRes = None
        aTT = None
        self.assertEqual(_types.refcount(t1), 1)

    def test_list_creation_doesnt_leak(self):
        T = ListOf(int)

        @Compiled
        def f(x: T, y: T) -> T:
            return x + y

        t1 = T(tuple(range(10000)))

        initMem = psutil.Process().memory_info().rss / 1024 ** 2

        for i in range(10000):
            f(t1, t1)

        finalMem = psutil.Process().memory_info().rss / 1024 ** 2

        self.assertTrue(finalMem < initMem + 5)

    def test_list_reserve(self):
        T = ListOf(TupleOf(int))

        @Compiled
        def f(x: T):
            x.reserve(x.reserved() + 10)

        aList = T()
        aList.resize(10)

        oldReserved = aList.reserved()
        f(aList)
        self.assertEqual(oldReserved+10, aList.reserved())

    def test_list_resize(self):
        T = ListOf(TupleOf(int))

        aTup = TupleOf(int)((1, 2, 3))

        @Compiled
        def f(x: T, y: TupleOf(int)):
            x.resize(len(x) + 10, y)

        aList = T()
        aList.resize(10)

        f(aList, aTup)

        self.assertEqual(_types.refcount(aTup), 11)

    def test_list_extend(self):
        T = ListOf(int)

        @Compiled
        def f(x: T, y: T):
            x.extend(y)

        t = T([1, 2, 3])
        t2 = T([1, 2, 3])

        f(t, t2)

        self.assertEqual(t, [1, 2, 3, 1, 2, 3])

        t = T([1, 2, 3])
        t2 = T([1, 2, 3])

        for _ in range(5):
            print(len(t))

            f(t, t)
            t2.extend(t2)

            self.assertEqual(t, t2)

    def test_list_append(self):
        T = ListOf(int)

        @Compiled
        def f(x: T):
            i = 0
            ct = len(x)
            while i < ct:
                x.append(x[i])
                i = i + 1

        aList = T([1, 2, 3, 4])

        f(aList)

        self.assertEqual(aList, [1, 2, 3, 4, 1, 2, 3, 4])

    def test_list_pop(self):
        T = ListOf(int)

        @Compiled
        def f(x: T):
            i = 0
            while i < len(x):
                x.pop(i)
                i = i + 1

        aList = T([1, 2, 3, 4])

        f(aList)

        self.assertEqual(aList, [2, 4])

    def test_list_of_oneOf(self):
        T = ListOf(OneOf(None, float))

        @Compiled
        def f():
            x = T()
            x.resize(2)
            x[0] = 10.0
            x[1] = None
            x.append(10.0)
            x.append(None)
            return x

        self.assertEqual(f(), [10.0, None, 10.0, None])

    @flaky(max_runs=3, min_passes=1)
    def test_lists_add_perf(self):
        T = ListOf(int)

        @Compiled
        def range(x: int):
            out = T()
            out.resize(x)

            i = 0
            while i < x:
                out[i] = i
                i = i + 1

            return out

        @Compiled
        def addSafe(x: T, y: T):
            out = T()

            i = 0
            while i < len(x):
                out.append(x[i] + y[i])
                i = i + 1

            return out

        @Compiled
        def addUnsafe(x: T, y: T):
            out = T()
            out.reserve(len(x))

            dest_ptr = out.pointerUnsafe(0)
            max_ptr = out.pointerUnsafe(len(x))
            x_ptr = x.pointerUnsafe(0)
            y_ptr = y.pointerUnsafe(0)

            while dest_ptr < max_ptr:
                dest_ptr.initialize(x_ptr.get() + y_ptr.get())
                dest_ptr += 1
                x_ptr += 1
                y_ptr += 1

            out.setSizeUnsafe(len(x))

            return out

        def timingComparison(addFun):
            ct = 10000000
            x = range(ct)
            y = range(ct)

            xnumpy = numpy.arange(ct)
            ynumpy = numpy.arange(ct)

            t0 = time.time()
            for _ in range(10):
                y = addFun(x, y)
            t1 = time.time()
            for _ in range(10):
                ynumpy = xnumpy+ynumpy
            t2 = time.time()

            slowerThanNumpyRatio = (t1 - t0) / (t2 - t1)

            self.assertEqual(y[10], x[10] * 11)

            print("Performance of ", addFun, "vs numpy is", slowerThanNumpyRatio, "times slower")

            return slowerThanNumpyRatio

        # numpy sped up between 1.16 and 1.17 somehow. I suspect it's using a more native set of
        # SIMD instructions that we're not persuading LLVM to emit.
        self.assertLess(timingComparison(addSafe), 10)  # 2.0 for me with numpy 1.16, but 4.0 with numpy 1.17?
        self.assertLess(timingComparison(addUnsafe), 4.0)  # 1.07 for me against numpy 1.16 but 4.0 against numpy 1.17

    def test_list_duplicate_operation(self):
        @Compiled
        def dupList(x: ListOf(int)):
            return ListOf(int)(x)

        x = ListOf(int)([1, 2, 3])
        y = dupList(x)
        x[0] = 100
        self.assertEqual(y[0], 1)

    def test_list_short_circuit_if(self):
        @Compiled
        def chkList(x: ListOf(int)):
            return len(x) > 0 and x[0]

        x = ListOf(int)([])
        self.assertFalse(chkList(x))

    def test_not_list(self):
        @Compiled
        def chkListInt(x: ListOf(int)):
            return not x

        x = ListOf(int)([])
        self.assertTrue(chkListInt(x))
        x.append(0)
        self.assertFalse(chkListInt(x))

        @Compiled
        def chkListListInt(x: ListOf(ListOf(int))):
            return not x

        x = ListOf(ListOf(int))()
        self.assertTrue(chkListListInt(x))
        x.append(ListOf(int)())
        self.assertFalse(chkListListInt(x))

    def test_convert_tuple_of_to_list(self):
        @Entrypoint
        def convertTo(x, T):
            return T(x)

        self.assertEqual(
            convertTo(TupleOf(int)([1, 2, 3]), ListOf(int)),
            [1, 2, 3]
        )

        self.assertEqual(
            convertTo(TupleOf(float)([1.5, 2.5, 3.5]), ListOf(int)),
            [1, 2, 3]
        )

        self.assertEqual(
            ListOf(int)(TupleOf(float)([1.5, 2.5, 3.5])),
            [1, 2, 3]
        )

        self.assertEqual(
            ListOf(OneOf(int, str))(TupleOf(float)([1.5, 2.5, 3.5])),
            [1, 2, 3]
        )

        self.assertEqual(
            convertTo(TupleOf(float)([1.5, 2.5, 3.5]), ListOf(OneOf(int, str))),
            [1, 2, 3]
        )

        # in a loop, so we can see we're not violating any refcounts
        for _ in range(100):
            with self.assertRaisesRegex(
                TypeError,
                "Cannot construct a new float from an instance of str"
            ):
                convertTo(TupleOf(float)([1.5, 2.5, "3.5"]), ListOf(OneOf(int, str)))

    def test_pop_behind_if(self):
        @Entrypoint
        def f(aList):
            if aList[-1] < 0:
                aList.pop()

            return aList

        self.assertEqual(f(ListOf(int)((0,))), [0])
        self.assertEqual(f(ListOf(int)((-1,))), [])

    def test_list_slice(self):
        def slice(aList: ListOf(int), a, b, c):
            return aList[a:b:c]

        sliceCompiled = Entrypoint(slice)

        self.assertEqual(sliceCompiled([], None, None, 1), [])

        for l in [
            ListOf(int)([]),
            ListOf(int)([1]),
            ListOf(int)([1, 2]),
            ListOf(int)([1, 2, 3]),
            ListOf(int)([1, 2, 3, 4])
        ]:
            for a in [None] + list(range(-5, 5)):
                for b in [None] + list(range(-5, 5)):
                    for c in [None] + list(range(-5, 5)):
                        if c != 0:
                            l0 = list(list(l)[a:b:c])
                            l1 = slice(l, a, b, c)
                            l2 = sliceCompiled(l, a, b, c)

                            self.assertTrue(len(l1) >= 0, len(l1))
                            self.assertTrue(len(l2) >= 0, len(l2))

                            self.assertEqual(l0, l1, (l, a, b, c))
                            self.assertEqual(l0, l2, (l, a, b, c))

    def test_list_pop_refcounts(self):
        lst = ListOf(ListOf(int))()
        aTup = ListOf(int)()

        for i in range(100):
            lst.append(aTup)

        @Entrypoint
        def popTen(lst):
            for _ in range(10):
                lst.pop()

        rc1 = _types.refcount(aTup)
        popTen(lst)
        rc2 = _types.refcount(aTup)

        self.assertEqual(rc2, rc1 - 10)

    def test_list_resize_recompile(self):
        lst = ListOf(int)()

        @Entrypoint
        def r1(lst):
            lst.resize(10)

        @Entrypoint
        def r2(lst):
            lst.resize(10, 10)

        r1(lst)
        r2(lst)

    def test_listof_aliasing(self):
        T = ListOf(int)

        t1 = T()
        t2 = T(t1)

        t1.append(1)

        assert len(t2) == 0

    def test_listof_aliasing_compiled(self):
        T = ListOf(int)

        @Entrypoint
        def dup(x):
            return T(x)

        t1 = T()
        t2 = dup(t1)

        t1.append(1)

        assert len(t2) == 0

    def test_list_add_coersion(self):
        aLst = ListOf(UInt16)()
        aLst.resize(20)

        @Entrypoint
        def addTwo(l):
            l[10] += 2

        addTwo(aLst)
        assert aLst[10] == 2

        aLst[10] = UInt8(2)
        assert aLst[10] == 2

        aLst[10] = UInt16(2)
        assert aLst[10] == 2

        aLst[10] += 2
        assert aLst[10] == 4

    def test_list_index_with_uints(self):
        aLst = ListOf(UInt16)()
        aLst.resize(20)

        @Entrypoint
        def addTwo(l):
            l[UInt8(10)] += 2

        addTwo(aLst)
        assert aLst[UInt8(10)] == 2

        aLst[UInt8(10)] = UInt8(2)
        assert aLst[UInt8(10)] == 2

        aLst[UInt8(10)] = UInt16(2)
        assert aLst[UInt8(10)] == 2

        aLst[UInt8(10)] += 2
        assert aLst[UInt8(10)] == 4

    def test_list_index_with_floats_fails(self):
        aLst = ListOf(UInt16)()
        aLst.resize(20)

        @Entrypoint
        def addTwo(l):
            l[10.5] += 2

        @Entrypoint
        def lookup(l):
            return l[10.5]

        @Entrypoint
        def assign(l):
            l[10.5] = 0

        with self.assertRaisesRegex(TypeError, "Can't take.*integer index"):
            addTwo(aLst)

        with self.assertRaisesRegex(TypeError, "Can't take.*integer index"):
            lookup(aLst)

        with self.assertRaisesRegex(TypeError, "Can't take.*integer index"):
            assign(aLst)

        with self.assertRaises(TypeError):
            aLst[1.2]

        with self.assertRaises(TypeError):
            aLst[1.2] = 1

        with self.assertRaises(TypeError):
            aLst[1.2] += 1

    def test_list_index_with_class_with_index(self):
        aLst = ListOf(UInt16)()
        aLst.resize(20)

        class AnIndex(Class, Final):
            ix = Member(int)

            def __index__(self):
                return self.ix

        anIndex = AnIndex(ix=10)

        @Entrypoint
        def addTwo(l):
            l[anIndex] += 2

        addTwo(aLst)
        assert aLst[anIndex] == 2

        @Entrypoint
        def addTwoObjectstyle(l, index: object):
            l[index] += 2

        addTwoObjectstyle(aLst, anIndex)
        assert aLst[anIndex] == 4

        aLst[anIndex] = UInt8(2)
        assert aLst[anIndex] == 2

        aLst[anIndex] = UInt16(2)
        assert aLst[anIndex] == 2

        aLst[anIndex] += 2
        assert aLst[anIndex] == 4

    def test_list_index_with_class_with_bad_index(self):
        aLst = ListOf(UInt32)()
        aLst.resize(20)

        class AnIndex(Class, Final):
            ix = Member(int)

            def __index__(self):
                return "invalid index"

        anIndex = AnIndex(ix=10)

        @Entrypoint
        def getIndex(l, index):
            return l[index]

        with self.assertRaisesRegex(TypeError, "returned non-int"):
            getIndex(aLst, anIndex)

        with self.assertRaisesRegex(TypeError, "returned non-int"):
            aLst[anIndex]

        @Entrypoint
        def getIndexAsObject(l, index: object):
            return l[index]

        with self.assertRaisesRegex(TypeError, "returned non-int"):
            getIndexAsObject(aLst, anIndex)

        class AUintIndex(Class, Final):
            ix = Member(int)

            def __index__(self):
                return UInt8(10)

        with self.assertRaisesRegex(TypeError, "returned non-int"):
            assert getIndexAsObject(aLst, AUintIndex()) == aLst[10]

        with self.assertRaisesRegex(TypeError, "returned non-int"):
            assert getIndex(aLst, AUintIndex()) == aLst[10]

        with self.assertRaisesRegex(TypeError, "returned non-int"):
            assert aLst[AUintIndex()] == aLst[10]

    def test_list_to_and_from_bytes(self):
        aList = ListOf(int)([1, 2, 3])

        assert len(aList.toBytes()) == 24
        assert ListOf(int).fromBytes(aList.toBytes()) == aList

        with self.assertRaisesRegex(TypeError, "POD"):
            ListOf(str)(["hi"]).toBytes()

        with self.assertRaisesRegex(TypeError, "POD"):
            ListOf(str).fromBytes(b"")

        @Entrypoint
        def toBytes(l):
            return l.toBytes()

        @Entrypoint
        def fromBytes(T, b):
            return T.fromBytes(b)

        assert aList.toBytes() == toBytes(aList)
        assert fromBytes(ListOf(int), aList.toBytes()) == aList

        assert fromBytes(ListOf(UInt8), aList.toBytes()) == ListOf(UInt8).fromBytes(aList.toBytes())

    def test_assign_list_conversion(self):
        L = ListOf(ListOf(int))

        aList = L()

        # this casts the list
        aList.append([1, 2])

        # this also casts the list
        aList[0] = [1, 2, 3]

        @Entrypoint
        def appendIt(l, x):
            l.append(x)

        @Entrypoint
        def assignIt(l, x):
            l[0] = x

        appendIt(aList, [1, 2])
        assignIt(aList, [1, 2, 3])

        with self.assertRaises(TypeError):
            appendIt(aList, ["hi"])

        with self.assertRaises(TypeError):
            assignIt(aList, ["hi"])

        with self.assertRaises(TypeError):
            aList.append("hi")

        with self.assertRaises(TypeError):
            aList[0] = "hi"
