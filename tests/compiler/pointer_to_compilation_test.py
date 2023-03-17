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

from typed_python import ListOf, Tuple, Compiled, Entrypoint, Class, Member, Final, NamedTuple, PointerTo, Set
import typed_python._types as _types
import unittest


class TestPointerToCompilation(unittest.TestCase):
    def test_pointer_operations(self):
        T = ListOf(int)

        def testfun(x: T):
            pointer = x.pointerUnsafe(0)

            pointer.set(20)
            (pointer+1).set(20)
            (pointer+2).set((pointer+1).get()+1)
            (pointer+3).initialize((pointer+2).get())

            (pointer+4).cast(float).set(1.0)
            return pointer[3]

        compiledFun = Compiled(testfun)

        l1 = T(list(range(10)))
        l2 = T(list(range(10)))

        self.assertEqual(testfun(l1), l1[3])
        self.assertEqual(compiledFun(l2), l2[3])

        self.assertEqual(l1, l2)

        self.assertEqual(l1[0], 20)
        self.assertEqual(l1[1], 20)
        self.assertEqual(l1[2], 21)
        self.assertEqual(l1[3], 21)
        self.assertEqual(l1[4], 0x3ff0000000000000)  # hex representation of 64 bit float 1.0

    def test_bytecount(self):
        def testfun(x):
            return _types.bytecount(type(x))

        self.assertEqual(testfun(0), 8)

        def check(x):
            self.assertEqual(
                testfun(x),
                Entrypoint(testfun)(x)
            )

        check(False)
        check(0)
        check(0.0)
        check(ListOf(int)([10]))
        check(Tuple(int, int, int)((10, 10, 10)))

    def test_pointer_subtraction(self):
        T = ListOf(int)

        def testfun(x: T):
            pointer = x.pointerUnsafe(0)

            return (pointer + 1) - pointer

        compiledFun = Compiled(testfun)

        self.assertEqual(testfun(T()), 1)
        self.assertEqual(compiledFun(T()), 1)

    def test_pointer_bool(self):
        T = ListOf(int)

        def testfun(x: T):
            pointer = x.pointerUnsafe(0)

            return bool(pointer)

        compiledFun = Compiled(testfun)

        self.assertEqual(testfun(T([1])), True)
        self.assertEqual(compiledFun(T([1])), True)

    def test_pointer_to_addition(self):
        aList = ListOf(int)()
        aList.resize(10)

        p = aList.pointerUnsafe(0)

        def add(x, y):
            return x + y

        self.assertEqual(add(p, 1), Entrypoint(add)(p, 1))

        def iadd(x, y):
            x += y
            return x

        self.assertEqual(iadd(p, 1), Entrypoint(iadd)(p, 1))

    def test_pointer_setitem_works(self):
        def f():
            x = ListOf(int)([1, 2, 3])
            p = x.pointerUnsafe(0)
            p[1] += p[0]

            return x[1]

        self.assertEqual(f(), 3)
        self.assertEqual(Entrypoint(f)(), 3)

    def test_initialize_with_unlike_type(self):
        def f(T, v):
            x = ListOf(T)()
            x.reserve(1)
            x.pointerUnsafe(0).initialize(v)
            x.setSizeUnsafe(1)
            return x

        self.assertEqual(f(int, 2.5), Entrypoint(f)(int, 2.5))

    def test_initialize_with_class_without_default_init(self):
        def f(T, v):
            x = ListOf(T)()
            x.reserve(1)
            x.pointerUnsafe(0).initialize(v)
            x.setSizeUnsafe(1)
            return x

        class C(Class, Final):
            x = Member(int)

            def __init__(self, x):
                self.x = x

            def __eq__(self, other):
                if type(self) != type(other):
                    return False
                return self.x == other.x

        self.assertEqual(f(C, C(2.5)), Entrypoint(f)(C, C(2.5)))

        # you can't copy-initialize a C with an int. you have to actively call it
        with self.assertRaisesRegex(Exception, "Couldn't initialize type C from int"):
            Entrypoint(f)(C, 2)

        with self.assertRaisesRegex(Exception, "Cannot implicitly convert an object of type int to an instance of C"):
            f(C, 2)

    def test_pointer_destroy_in_interpreter(self):
        aList = ListOf(str)(["a", "b"])

        p = aList.pointerUnsafe(0)

        p.destroy()
        p.initialize("B")

        assert aList[0] == "B"

        p.destroy()
        p.initialize()

        assert aList[0] == ""

    def test_pointer_to_named_tuple(self):
        NT = NamedTuple(x=int, y=float)
        Lst = ListOf(NT)

        lst = Lst([NT(x=10, y=20)])

        @Entrypoint
        def ptrToX(p):
            return p.x

        assert type(ptrToX(lst.pointerUnsafe(0))) == PointerTo(int)
        assert type(lst.pointerUnsafe(0).x) == PointerTo(int)

        assert lst.pointerUnsafe(0).x.get() == 10
        assert ptrToX(lst.pointerUnsafe(0)).get() == 10

    def test_type_of_element_of_pointer_to_set(self):
        P = PointerTo(Set(int))

        assert P.ElementType is Set(int)
        assert P().ElementType is Set(int)

        @Entrypoint
        def check():
            assert P.ElementType is Set(int), P.ElementType
            assert P().ElementType is Set(int), P().ElementType

        check()
