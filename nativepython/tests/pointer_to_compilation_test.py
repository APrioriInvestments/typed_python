#   Copyright 2018 Braxton Mckee
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

from typed_python import *
import typed_python._types as _types
from nativepython.runtime import Runtime
import unittest


def Compiled(f):
    f = Function(f)
    return Runtime.singleton().compile(f)


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
                Runtime.singleton().compile(testfun, {'x': type(x)})(x)
            )

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
