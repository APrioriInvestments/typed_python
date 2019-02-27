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

from typed_python import TypeFunction, Class, Alternative, Member, SerializationContext
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

        # TODO: re-enable after we fix `applyNameChangesToType`
        # self.assertIn("List(int)", str(List(int)))

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

    def test_nonsensical_self_recursion_throws(self):
        @TypeFunction
        def crazy(T):
            return crazy(crazy(T))

        with self.assertRaisesRegex(TypeError, "has not resolved yet"):
            crazy(int)

    def test_can_serialize_type_functions(self):
        @TypeFunction
        def List(T):
            return Alternative(
                "List",
                Node={"head": T, "tail": List(T)},
                Empty={}
            )

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
