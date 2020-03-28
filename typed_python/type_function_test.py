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

from typed_python import (
    TypeFunction, Class, Alternative, Member, SerializationContext,
    Forward, ListOf, Final, isCompiled, Entrypoint
)

import unittest


class TypeFunctionTest(unittest.TestCase):
    def test_basic(self):
        @TypeFunction
        def List(T):
            return Alternative(
                "List",
                Node={"head": T, "tail": List(T)},
                Empty={}
            )

        self.assertIs(List(int), List(int))
        self.assertIsNot(List(float), List(int))

        l_i = List(int).Empty()
        l_f = List(float).Empty()

        List(int).Node(head=10, tail=l_i)
        List(float).Node(head=10.5, tail=l_f)

        with self.assertRaises(TypeError):
            List(int).Node(head=10, tail=l_f)

    def test_mutually_recursive(self):
        @TypeFunction
        def X(T):
            class X(Class):
                i = Member(T)
                y = Member(Y(T))
            return X

        @TypeFunction
        def Y(T):
            class Y(Class):
                i = Member(T)
                x = Member(X(T))
            return Y

        self.assertIs(Y(int), Y(int))
        self.assertIs(Y(int).x.type, X(int))
        self.assertIs(Y(int).x.type.y.type, Y(int))

        anX = X(int)()
        anX.y = Y(int)()
        anX.y.x = anX

        with self.assertRaises(TypeError):
            anX.y = Y(float)()

    def test_can_serialize_type_functions(self):
        @TypeFunction
        def List(T):
            ListT = Forward("ListT")
            return ListT.define(Alternative(
                "List",
                Node={"head": T, "tail": ListT},
                Empty={}
            ))

        context = SerializationContext({'List': List})

        self.assertIs(
            context.deserialize(context.serialize(List(int))),
            List(int)
        )
        self.assertIsInstance(
            context.deserialize(context.serialize(List(int).Empty())),
            List(int)
        )

        list_of_int = List(int)
        list_of_list = List(list_of_int)

        l0 = list_of_int.Empty()
        l_l = list_of_list.Node(head=l0, tail=list_of_list.Empty())

        self.assertEqual(
            context.deserialize(context.serialize(l_l)),
            l_l
        )

    def test_type_function_on_typed_lists(self):
        @TypeFunction
        def SumThem(x):
            summedValues = sum(x)

            class C:
                X = summedValues

            return C

        self.assertIs(
            SumThem([1, 2, 3]),
            SumThem(ListOf(int)([1, 2, 3]))
        )

        self.assertEqual(SumThem(ListOf(int)([1, 2, 3])).X, 6)

    def test_type_functions_with_recursive_annotations(self):
        @TypeFunction
        def Boo(T):
            class Boo_(Class, Final):
                def f(self) -> Boo(T):
                    return self
            return Boo_

        boo = Boo(int)()

        self.assertEqual(boo, boo.f())

    def test_compile_typefunc_staticmethod(self):
        @TypeFunction
        def A(T):
            class MyClass(Class):
                @Entrypoint
                @staticmethod
                def aFunc(x: ListOf(T)) -> ListOf(T):
                    # accessing 'T' means that this is now a
                    # closure, holding 'T' in a 'global cell'
                    T

                    assert isCompiled()

                    return x

            return MyClass

        @Entrypoint
        def callAFunc():
            return A(float).aFunc(ListOf(float)([1, 2]))

        # we should be able to dispatch to this from the compiler
        self.assertEqual(callAFunc(), [1, 2])

        # we should also know that this is an entrypoint
        self.assertTrue(A(float).aFunc.isEntrypoint)

        # we should be able to dispatch to this from the interpreter and not
        # trip the assertion on isCompiled
        self.assertEqual(A(float).aFunc(ListOf(float)([1, 2])), [1, 2])
