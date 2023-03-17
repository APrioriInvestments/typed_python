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
import unittest
from typed_python import (
    ListOf, Set, Dict, TupleOf, OneOf, Tuple, NamedTuple,
    ConstDict, Alternative, Forward, Class, PointerTo, Type
)


class TypesMetadataTest(unittest.TestCase):
    def test_type_relationships(self):
        assert issubclass(ListOf, Type)

        assert issubclass(ListOf(int), ListOf)
        assert issubclass(TupleOf(int), TupleOf)
        assert issubclass(Set(int), Set)
        assert issubclass(OneOf(int, float), OneOf)
        assert issubclass(NamedTuple(x=int), NamedTuple)
        assert issubclass(Tuple(int), Tuple)
        assert issubclass(Alternative("A", X={}), Alternative)
        assert issubclass(Dict(int, int), Dict)
        assert issubclass(ConstDict(int, int), ConstDict)
        assert issubclass(PointerTo(int), PointerTo)

        class A(Class):
            pass

        assert issubclass(A, Class)

    def test_tupleOf(self):
        self.assertEqual(TupleOf(int), TupleOf(int))
        self.assertEqual(TupleOf(int).ElementType, int)

        self.assertEqual(TupleOf(float).ElementType, float)
        self.assertEqual(TupleOf(OneOf(10, 20)).ElementType, OneOf(10, 20))

        self.assertEqual(TupleOf(object).ElementType, object)
        self.assertEqual(TupleOf(10).ElementType.__typed_python_category__, "Value")

    def test_tuple(self):
        self.assertEqual(Tuple(int, int, OneOf(10, 20)).ElementTypes, (int, int, OneOf(10, 20)))

    def test_named_tuple(self):
        self.assertEqual(NamedTuple(x=int, y=int, z=OneOf(10, 20)).ElementTypes, (int, int, OneOf(10, 20)))
        self.assertEqual(NamedTuple(x=int, y=int, z=OneOf(10, 20)).ElementNames, ('x', 'y', 'z'))

    def test_const_dict(self):
        self.assertEqual(ConstDict(str, int).KeyType, str)
        self.assertEqual(ConstDict(str, int).ValueType, int)

    def test_alternatives(self):
        X = Forward("X")
        X = X.define(Alternative(
            "X",
            Left={'x': int, 'y': str},
            Right={'x': X, 'val': int}
        ))

        self.assertEqual(len(X.__typed_python_alternatives__), 2)

        Left, Right = X.__typed_python_alternatives__

        self.assertEqual(Left.Index, 0)
        self.assertEqual(Right.Index, 1)

        self.assertEqual(Left.ElementType.ElementNames, ("x", "y"))
        self.assertEqual(Left.ElementType.ElementTypes, (int, str))
        self.assertEqual(Right.ElementType.ElementNames, ('x', 'val'))
        self.assertEqual(Right.ElementType.ElementTypes, (X, int))

    def test_oneof(self):
        someInts = TupleOf(int)((1, 2))

        T = OneOf(1, "2", someInts)

        self.assertEqual(T.Types[0].Value, 1)
        self.assertEqual(T.Types[1].Value, "2")
        self.assertEqual(T.Types[2].Value, someInts)
