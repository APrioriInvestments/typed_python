#   Coyright 2017-2019 Nativepython Authors
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
# from typed_python.internals import forwardToName
from typed_python import TupleOf, OneOf, Alternative, Class, Member, Forward


class NativeForwardTypesTests(unittest.TestCase):

    # def test_forwardToName(self):
    #     X = 10
    #     self.assertEqual(forwardToName(lambda: X), "X")
    #     self.assertEqual(forwardToName(lambda: X+X), "UnknownForward")

    def test_recursive_alternative(self):
        List = Forward("List*")
        List = List.define(Alternative(
            "List",
            Node={'head': int, 'tail': List },
            Leaf={},
            unpack=lambda self: () if self.matches.Leaf else (self.head,) + self.tail.unpack()
        ))

        # ensure recursive implementation actually works
        lst = List.Leaf()

        for i in range(100):
            lst = List.Node(head=i, tail=lst)

        self.assertEqual(list(lst.unpack()), list(reversed(range(100))))

    # TODO: make this test meaningful?
    # def test_instantiating_invalid_forward(self):
    #     X = Alternative("X", A={'x': lambda: this_does_not_Exist })  # noqa:F821
    #
    #     with self.assertRaises(TypeError):
    #         X.A()
    #
    #     this_does_not_exist = int
    #
    #     # fixing it doesn't help
    #     with self.assertRaises(TypeError):
    #         X.A()
    #
    #     # but a new type is OK.
    #     X = Alternative("X", A={'x': lambda: this_does_not_exist })
    #
    #     X.A()

    def test_mutually_recursive_classes(self):
        B0 = Forward("B*")

        class A(Class):
            bvals = Member(TupleOf(B0))

        class B(Class):
            avals = Member(TupleOf(A))

        B0 = B0.define(B)
        a = A()
        b = B()

        a.bvals = (b,)
        b.avals = (a,)

        self.assertTrue(a.bvals[0].avals[0] == a)

    def DISABLEDtest_recursives_held_infinitely_throws(self):
        # not implemented yet but should throw
        class X(Class):
            impossible = Member(OneOf(None, lambda: X))

        with self.assertRaises(TypeError):
            X()

    def test_tuple_of_one_of(self):
        X = Forward("X*")
        Y = OneOf(None, X)
        X = X.define(TupleOf(Y))

        str(X)

        anX = X( (None,) )

        self.assertTrue("Unresolved" not in str(X))

        anotherX = X( (anX, anX) )

        self.assertEqual(anotherX[0], anX)

    def test_deep_forwards_work(self):
        X = Forward("X*")
        X = X.define(TupleOf(TupleOf(TupleOf(TupleOf(OneOf(None, X))))))

        str(X)

        anX = X( ((((None,),),),) )

        anotherX = X( ((((anX,),),),) )

        self.assertEqual(anotherX[0][0][0][0], anX)
