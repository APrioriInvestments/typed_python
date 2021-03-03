#   Copyright 2017-2020 typed_python Authors
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

import unittest
import time
import gc
import pytest
from flaky import flaky

from typed_python import Entrypoint, ListOf, TupleOf, Class, Member, Final, Generator, OneOf
from typed_python.test_util import currentMemUsageMb


def timeIt(f):
    t0 = time.time()
    f()
    return time.time() - t0


class TestGeneratorsAndComprehensions(unittest.TestCase):
    def test_list_comp(self):
        @Entrypoint
        def listComp(x):
            return [a + 1 for a in range(x)]

        lst = listComp(10)

        assert isinstance(lst, list)
        assert lst == [a + 1 for a in range(10)]

    def test_set_comprehension(self):
        @Entrypoint
        def setComp(x):
            return {a + 1 for a in range(x)}

        st = setComp(10)

        assert isinstance(st, set)
        assert st == {a + 1 for a in range(10)}

    def test_tuple_comprehension(self):
        @Entrypoint
        def tupComp(x):
            return tuple(a + 1 for a in range(x))

        tup = tupComp(10)

        assert isinstance(tup, tuple)
        assert tup == tuple(a + 1 for a in range(10))

    def test_dict_comprehension(self):
        @Entrypoint
        def dictComp(x):
            return {k: k + 1 for k in range(10)}

        d = dictComp(10)

        assert isinstance(d, dict)
        assert d == {a: a + 1 for a in range(10)}

    def test_dict_comprehension_multiple_types(self):
        def dictComp(x):
            return {k if (k%3) else "Boo": k + 1 if (k % 2) else "hi" for k in range(10)}

        dictCompCompiled = Entrypoint(dictComp)

        d = dictCompCompiled(10)
        assert isinstance(d, dict)

        assert d == dictComp(10)

    def test_list_from_listcomp(self):
        @Entrypoint
        def listComp(x):
            return ListOf(int)([a + 1 for a in range(x)])

        lst = listComp(10)

        assert isinstance(lst, ListOf(int))
        assert lst == [a + 1 for a in range(10)]

    def test_generator_exp_is_generator(self):
        @Entrypoint
        def generatorComp(x):
            return (a + 1 for a in range(x))

        g = generatorComp(10)

        assert isinstance(g, Generator(int))

        assert list(g) == [x + 1 for x in range(10)]

    @flaky(max_runs=3, min_passes=1)
    def test_list_from_listcomp_perf(self):
        def sum(iterable):
            res = 0
            for s in iterable:
                res += s
            return res

        @Entrypoint
        def listCompSumConverted(x):
            return sum(ListOf(int)([a + 1 for a in range(x)]))

        @Entrypoint
        def listCompSumGenerator(x):
            return sum(ListOf(int)(a + 1 for a in range(x)))

        @Entrypoint
        def tupleCompSumConverted(x):
            return sum(TupleOf(int)([a + 1 for a in range(x)]))

        @Entrypoint
        def tupleCompSumGenerator(x):
            return sum(TupleOf(int)(a + 1 for a in range(x)))

        @Entrypoint
        def listCompSumMasquerade(x):
            return sum([a + 1 for a in range(x)])

        def listCompSumUntyped(x):
            return sum([a + 1 for a in range(x)])

        listCompSumConverted(1000)
        listCompSumGenerator(1000)
        tupleCompSumConverted(1000)
        tupleCompSumGenerator(1000)
        listCompSumMasquerade(1000)

        compiledTimes = [
            timeIt(lambda: listCompSumConverted(1000000)),
            timeIt(lambda: listCompSumGenerator(1000000)),
            timeIt(lambda: tupleCompSumConverted(1000000)),
            timeIt(lambda: tupleCompSumGenerator(1000000)),
            timeIt(lambda: listCompSumMasquerade(1000000)),
        ]
        untypedTime = timeIt(lambda: listCompSumUntyped(1000000))

        print(compiledTimes)

        avgCompiledTime = sum(compiledTimes) / len(compiledTimes)

        # they should be about the same
        for timeElapsed in compiledTimes:
            assert .5 <= timeElapsed / avgCompiledTime <= 2.0

        # but python is much slower. I get about 30 x.
        assert untypedTime / avgCompiledTime > 10

    @flaky(max_runs=3, min_passes=1)
    def test_untyped_tuple_from_listcomp_perf(self):
        def sum(iterable):
            res = 0
            for s in iterable:
                res += s
            return res

        @Entrypoint
        def tupleCompSumConverted(x):
            return sum(tuple([a + 1 for a in range(x)]))

        @Entrypoint
        def tupleCompSumConvertedGenerator(x):
            return sum(tuple(a + 1 for a in range(x)))

        @Entrypoint
        def listCompSumConvertedGenerator(x):
            return sum(list(a + 1 for a in range(x)))

        def listCompSumUntyped(x):
            return sum(tuple(a + 1 for a in range(x)))

        tupleCompSumConverted(1000)
        tupleCompSumConvertedGenerator(1000)
        listCompSumConvertedGenerator(1000)

        tupleCompiled = timeIt(lambda: tupleCompSumConverted(10000000))
        tupleCompiledGenerator = timeIt(lambda: tupleCompSumConvertedGenerator(10000000))
        listCompiledGenerator = timeIt(lambda: listCompSumConvertedGenerator(10000000))
        untypedTime = timeIt(lambda: listCompSumUntyped(10000000))

        print("tupleCompiled = ", tupleCompiled)
        print("tupleCompiledGenerator = ", tupleCompiledGenerator)
        print("listCompiledGenerator = ", listCompiledGenerator)
        print("untypedTime = ", untypedTime)

        # they should be about the same
        assert .75 <= tupleCompiled / tupleCompiledGenerator <= 1.25
        assert .75 <= listCompiledGenerator / tupleCompiledGenerator <= 1.25

        # but python is much slower. I get about 30 x.
        assert untypedTime / listCompiledGenerator > 10

    def executeInLoop(self, f, duration=.25, threshold=1.0):
        gc.collect()
        memUsage = currentMemUsageMb()

        t0 = time.time()

        count = 0
        while time.time() - t0 < duration:
            f()
            count += 1

        gc.collect()
        print("count=", count, "allocated=", currentMemUsageMb() - memUsage)
        self.assertLess(currentMemUsageMb() - memUsage, threshold)

    @pytest.mark.skipif('sys.platform=="darwin"')
    def test_listcomp_doesnt_leak(self):
        @Entrypoint
        def listComp(x):
            return [a + 1 for a in range(x)]

        @Entrypoint
        def sumListComp(x):
            l = listComp(x)
            res = 0
            for val in l:
                res += val
            return res

        # burn it in
        sumListComp(1000000)
        self.executeInLoop(lambda: sumListComp(1000000), duration=.1, threshold=20.0)
        self.executeInLoop(lambda: sumListComp(1000000), duration=.25, threshold=1.0)

    def test_call_generator(self):
        @Entrypoint
        def generateInts(ct):
            yield 1
            yield 2

        assert list(generateInts(100)) == [1, 2]

    def test_call_generator_with_branch(self):
        @Entrypoint
        def generateInts(ct):
            yield 1

            if ct > 0:
                yield 2
            else:
                yield 3

            yield 4

        assert list(generateInts(1)) == [1, 2, 4]
        assert list(generateInts(-1)) == [1, 3, 4]

    def test_call_generator_with_loop(self):
        @Entrypoint
        def generateInts(ct):
            x = 0
            while x < ct:
                yield x
                x = x + 1
            else:
                yield -1
            yield -2

        assert list(generateInts(10)) == list(range(10)) + [-1, -2]
        assert list(generateInts(0)) == list(range(0)) + [-1, -2]

    def test_call_generator_with_closure_var(self):
        xInClosure = 100

        @Entrypoint
        def generateInts(ct):
            yield 1
            yield xInClosure
            yield ct
            yield 2

        assert list(generateInts(10)) == [1, 100, 10, 2]

    def test_call_generator_with_arg_assign(self):
        @Entrypoint
        def generateInts(ct):
            yield ct
            ct = 2
            yield ct

        assert list(generateInts(10)) == [10, 2]

    def test_call_generator_with_aug_assign(self):
        @Entrypoint
        def generateInts(ct):
            yield ct
            ct += 1
            yield ct
            ct += 2
            yield ct
            ct += 3
            yield ct

        assert list(generateInts(0)) == [0, 1, 3, 6]

    def test_call_generator_with_ann_assign(self):
        @Entrypoint
        def generateInts(ct):
            yield ct
            ct: int = 1
            yield ct

        assert list(generateInts(0)) == [0, 1]

    def test_call_generator_with_pass(self):
        @Entrypoint
        def generateInts(ct):
            if ct > 10:
                pass

            yield ct

        assert list(generateInts(100)) == [100]

    def test_call_generator_with_continue(self):
        @Entrypoint
        def generateInts(ct):
            x = 0

            while x < ct:
                yield x
                x = x + 1

                if x % 2 == 0:
                    continue

                yield x

        assert list(generateInts(5)) == [0, 1, 1, 2, 3, 3, 4, 5]

    def test_call_generator_with_break(self):
        @Entrypoint
        def generateInts(ct):
            x = 0

            while x < ct:
                yield x
                x = x + 1

                if x % 2 == 0:
                    break

                yield x

        assert list(generateInts(5)) == [0, 1, 1]

    def test_call_generator_with_closure_var_cant_assign(self):
        xInClosure = 100

        @Entrypoint
        def generateInts(ct):
            yield 1
            yield xInClosure
            xInClosure = xInClosure + 1  # noqa
            yield ct
            yield 2

        with self.assertRaises(NameError):
            assert list(generateInts(10)) == [1, 100, 10, 2]

    def test_call_generator_with_assert(self):
        @Entrypoint
        def generateInts(ct):
            y = ct

            assert y > 10
            yield y
            y -= 10
            yield y

        assert list(generateInts(15)) == [15, 5]

    def test_call_generator_with_try_finally(self):
        @Entrypoint
        def generateInts():
            try:
                yield 1

                yield 2
            finally:
                yield 3

                yield 4

            yield 5

        assert list(generateInts()) == [1, 2, 3, 4, 5]

    def test_call_generator_with_try(self):
        @Entrypoint
        def generateInts():
            hasCaught = False

            try:
                yield 1

                yield 2

                raise Exception("catch me")

                yield 300
            except Exception:
                assert not hasCaught
                hasCaught = True
                yield 3
                assert hasCaught
                yield 4
            finally:
                yield 5

            yield 6

        assert list(generateInts()) == [1, 2, 3, 4, 5, 6]

    def test_generator_produces_stop_iteration_when_done(self):
        @Entrypoint
        def generateInts():
            yield 1

            yield 2

        g = generateInts()

        assert g.__next__() == 1
        assert g.__next__() == 2

        with pytest.raises(StopIteration):
            g.__next__()

        with pytest.raises(StopIteration):
            g.__next__()

        with pytest.raises(StopIteration):
            g.__next__()

    def test_raise_in_generator_stops_iteration(self):
        @Entrypoint
        def generateInts():
            yield 1

            yield 2

            raise Exception("catch me")

        g = generateInts()

        assert g.__next__() == 1
        assert g.__next__() == 2

        with pytest.raises(Exception, match="catch me"):
            g.__next__()

        with pytest.raises(StopIteration):
            g.__next__()

        with pytest.raises(StopIteration):
            g.__next__()

    @pytest.mark.skip(reason='not implemented yet')
    def test_reraise_in_generator_after_yield(self):
        @Entrypoint
        def generateInts():
            try:
                yield 1

                yield 2

                raise Exception("catch me")
            except Exception:
                yield 3

                raise

        g = generateInts()

        assert g.__next__() == 1
        assert g.__next__() == 2
        assert g.__next__() == 3

        with pytest.raises(Exception, match="catch me"):
            g.__next__()

    def test_return_in_generator(self):
        @Entrypoint
        def generateInts():
            yield 1

            return 20

        g = generateInts()

        assert g.__next__() == 1

        try:
            g.__next__()
        except StopIteration as i:
            assert i.args == (20,)

        try:
            g.__next__()
        except StopIteration as i:
            assert i.args == ()

    def test_argless_return_in_generator(self):
        @Entrypoint
        def generateInts():
            yield 1

            return

        g = generateInts()

        assert g.__next__() == 1

        try:
            g.__next__()
        except StopIteration as i:
            assert i.args == ()

        try:
            g.__next__()
        except StopIteration as i:
            assert i.args == ()

    def test_with_in_generator(self):
        class ContextManager(Class, Final):
            entered = Member(int)

            def __init__(self):
                self.entered = 0

            def __enter__(self):
                self.entered += 1

            def __exit__(self, a, b, c):
                self.entered -= 1

        cm = ContextManager()

        @Entrypoint
        def generateInts(x):
            yield 1

            with x:
                yield 2

                with x:
                    yield 3

                    yield 4

                yield 5

            yield 6

        g = generateInts(cm)

        assert cm.entered == 0

        assert g.__next__() == 1
        assert cm.entered == 0

        assert g.__next__() == 2
        assert cm.entered == 1

        assert g.__next__() == 3
        assert cm.entered == 2

        assert g.__next__() == 4
        assert cm.entered == 2

        assert g.__next__() == 5
        assert cm.entered == 1

        assert g.__next__() == 6
        assert cm.entered == 0

    def test_for_in_generator(self):
        @Entrypoint
        def generateInts(x):
            yield -1

            for val in range(x):
                yield val
            else:
                yield -2

            yield -3

        g = generateInts(5)

        assert g.__next__() == -1
        assert g.__next__() == 0
        assert g.__next__() == 1
        assert g.__next__() == 2
        assert g.__next__() == 3
        assert g.__next__() == 4
        assert g.__next__() == -2
        assert g.__next__() == -3

        with self.assertRaises(StopIteration):
            assert g.__next__() == -3

    def test_for_in_generator_over_various_builtin_types(self):
        @Entrypoint
        def generate(l):
            for val in l:
                yield val

        for toIterate in [
            ListOf(int)([1, 2, 3]),
            TupleOf(int)([1, 2, 3]),
            "hi",
            b"someBytes"
        ]:
            assert list(generate(toIterate)) == list(toIterate)

    def test_can_iterate_class(self):
        class C(Class, Final):
            x = Member(int)

            def __iter__(self):
                for val in range(self.x):
                    yield val * 2

        @Entrypoint
        def iterate(l):
            res = None
            for val in l:
                if res is None:
                    res = ListOf(type(val))()
                res.append(val)
            return res

        c = C(x=4)

        print(list(c))

    @flaky(max_runs=3, min_passes=1)
    def test_can_iterate_class_perf(self):
        class C(Class, Final):
            x = Member(int)

            def __iter__(self):
                val = 0
                while val < self.x:
                    yield val * 2
                    val += 1

        @Entrypoint
        def add(l):
            res = 0

            for val in l:
                res += val

            return res

        add(C(x=4))

        CT = 10 * 1000000

        t0 = time.time()
        add(C(x=CT))
        elapsedTyped = time.time() - t0

        def untypedGenerator(x):
            val = 0
            while val < x:
                yield val * 2
                val += 1

        def addUntyped(iterable):
            res = 0
            for val in iterable:
                res += val
            return res

        t0 = time.time()
        addUntyped(untypedGenerator(CT))
        elapsedUntyped = time.time() - t0

        print(elapsedUntyped, f" to iterate {CT//1000000}mm untyped")
        print(elapsedTyped, f" to iterate {CT//1000000}mm typed")
        print("speedup is ", elapsedUntyped / elapsedTyped)

        # I get about 12.
        assert elapsedUntyped / elapsedTyped > 4

    def test_can_use_lambdas_in_generators(self):
        @Entrypoint
        def iterate(x):
            f = lambda y: y * 2

            yield f(1)
            yield f(x)

        assert list(iterate(10)) == [2, 20]

    def test_can_use_lambdas_in_generators_with_renamed_variables(self):
        @Entrypoint
        def iterate(x):
            f = lambda x: x * 2

            yield f(1)
            yield f(x)

        assert list(iterate(10)) == [2, 20]

    def test_can_make_closures_in_generator(self):
        @Entrypoint
        def iterate(x):
            def f(y):
                return y * 2

            yield f(1)
            yield f(x)

        assert list(iterate(10)) == [2, 20]

    def test_generator_interior_closures_masked_correctly(self):
        @Entrypoint
        def iterate(x):
            def f(y):
                x = y
                return x * 2

            yield f(1)
            yield f(x)

        assert list(iterate(10)) == [2, 20]

    def test_generator_interior_closures_capture(self):
        @Entrypoint
        def iterate(x):
            def f(y):
                return y + x

            yield f(1)

        assert list(iterate(10)) == [11]

    def test_generator_doubly_interior_closures_masked_correctly(self):
        @Entrypoint
        def iterate(x):
            def f(y):
                def g():
                    x = y
                    return x * 2
                return g()

            yield f(1)
            yield f(x)

        assert list(iterate(10)) == [2, 20]

    def test_generator_in_generator(self):
        @Entrypoint
        def iterate(x):
            def f(y):
                yield y + 2
                yield y + 4

            for val in f(1):
                yield val

            for val in f(x):
                yield val

        assert list(iterate(10)) == [3, 5, 12, 14]

    def test_generator_in_generator_reading_parent(self):
        @Entrypoint
        def iterate(x):
            def f(y):
                yield y + x

            yield f(1).__iter__().__next__()

        assert list(iterate(10)) == [11]

    def test_list_comp_in_generator(self):
        @Entrypoint
        def iterate(x):
            yield [z for z in range(x)]

        assert list(iterate(10)) == [list(range(10))]

    def test_set_comp_in_generator(self):
        @Entrypoint
        def iterate(x):
            yield {z for z in range(x)}

        assert list(iterate(10)) == [set(range(10))]

    def test_tuple_comp_in_generator(self):
        @Entrypoint
        def iterate(x):
            yield tuple(z for z in range(x))

        assert list(iterate(10)) == [tuple(range(10))]

    def test_dict_comp_in_generator(self):
        @Entrypoint
        def iterate(x):
            yield {z: z + 1 for z in range(x)}

        assert list(iterate(10)) == [{z: z + 1 for z in range(10)}]

    @pytest.mark.skip("not implemented yet")
    def test_list_comp_masking(self):
        # check that the masking behavior of nested variables in list comps is right
        # technically, each successive listcomp is its own scope
        # which we do not obey yet.

        @Entrypoint
        def iterate(x):
            return [x+1 for x in range(x) for x in range(x - 3)]

        x = 10
        assert iterate(10) == [x+1 for x in range(x) for x in range(x - 3)]

    def test_type_annotations_on_generator(self):
        @Entrypoint
        def iterate(x) -> Generator(OneOf(None, int)):
            yield 1

        assert isinstance(iterate(10), Generator(OneOf(None, int)))
