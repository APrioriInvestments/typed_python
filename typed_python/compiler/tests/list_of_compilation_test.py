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

from typed_python import ListOf, Function, TupleOf, OneOf, Compiled, Entrypoint, Tuple
import typed_python._types as _types
import unittest
import time
import numpy
import psutil
import pytest


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
            with self.assertRaisesRegex(TypeError, "not str"):
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

    def test_list_comprehensions(self):
        # def f(i):
        #     if random.random() < 0.999:
        #         res = ListOf(float)()
        #     else:
        #         res = list()
        #     for x in i:
        #         if random.random() < 0.01:
        #             e] = float(x)
        #         else:
        #             e = object(float(x))
        #         res.append(e)
        #     return res
        #
        # i = range(9)
        # r1 = f(i)
        # r2 = Entrypoint(f)(i)
        # print(r1,r2)
        # return
        # def f(i:ListOf(int)):
        #     result = list()
        #     for x in i:
        #         result.append(float(x))
        #     return result
        #
        # r = convertFunctionToAlgebraicPyAst(f)
        # f([1,2,3])
        # @Compiled
        # def internal(s:Tuple(int,int,str)):
        #     r = 0.123
        #     for x in s:
        #         if random.random() < 0.1:
        #             r = x
        #     return r
        #
        # #r1 = internal(Tuple(int,int,str)((1,3, "a")))
        # r2 = internal(ListOf(int)([1,2,3]),lambda x: 2*x)
        # return
        # def find_terminator(s, pos, terms):
        #     ret = len(s)
        #     for t in terms:
        #         p = s.find(t, pos)
        #         if p == -1:
        #             p = len(s)
        #         ret = min(ret, p)
        #     return ret
        #
        # def alpha(s, pos=0):
        #     x = find_terminator(s, pos, "(),")
        #     if x == len(s) or s[x] != '(':
        #         return ("Name" + s[pos:x] +"\r", x)
        #
        #     t = s[pos:x].strip()
        #     pos = x + 1
        #     arg_asts = []
        #     while True:
        #         (arg_ast, pos) = alpha(s, pos)
        #         arg_asts.append(arg_ast)
        #         if s[pos].isspace():
        #             pos += 1
        #             continue
        #         if s[pos] == ')':
        #             pos += 1
        #             break
        #         if s[pos] == ',':
        #             pos += 1
        #             continue
        #         assert False
        #         assert pos < len(s)
        #     return ("Call\rfunc=Name\rid=" + t + "\rargs=(" + "\r".join(arg_asts) + "\r)", pos)
        #
        # #r1 = alpha("Set")
        # #r2 = alpha("Set(int)")
        # #r3 = alpha("Tuple(int, float, str)")
        # r4 = alpha("Tuple(Set(int), float, OneOf(int,float), OneOf(ListOf(str),TupleOf(str)))")
        # return

        def f1(i):
            return [x for x in i]

        def f2(i, k):
            return [x * 2 + 1 for x in i if x % k]

        def f3(i, k1):
            return [0.01 * x for x in i if x % k1 if x % 5]

        def f4(i, k1):
            return [(x, y) for x in i for y in range(7) if x % k1 if y % (x + 2)]

        def f5(i, k1):
            return [(x, y) for x in i if x % k1 for y in range(7) if y % (x + 2)]

        def f6(i, k1):
            return [Tuple(int, int)((x, y)) for x in i if x % k1 for y in range(x) if y % k1]

        # def f7(i, k):
        #     return [Tuple(float,int)((float(x+0.1),x)) for x in i]
        #
        # k = 2
        # i= range(5, 9)
        # f = f6
        # r1 = f(i, 2)
        # r2 = Entrypoint(f)(i,2)
        # print(r1, r2)
        # self.assertEqual(r1, r2)
        # return

        for i in [range(5), ListOf(int)(range(10)), range(100), []]:
            r1 = f1(i)
            r2 = Entrypoint(f1)(i)
            self.assertEqual(r1, r2)

            for f in [f2, f3, f4, f5, f6]:
                for k in [1, 2, 3]:
                    r1 = f(i, k)
                    r2 = Entrypoint(f)(i, k)
                    self.assertEqual(r1, r2)

    @pytest.mark.skip(reason="fails")
    def test_list_comprehensions_nested(self):
        def f1(i1, i2):
            return [[x+y for x in i1 * y] for y in i2]

        for i1 in [[1, 2, 3], [10, 20, 30, 40, 50]]:
            for i2 in [[1, 2], [3, 2, 1]]:
                r1 = f1(i1, i2)
                r2 = Entrypoint(f1)(i1, i2)
                self.assertEqual(r1, r2)
