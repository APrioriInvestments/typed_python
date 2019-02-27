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
import unittest
from typed_python import TupleOf, OneOf, Tuple, NamedTuple, Int64, Float64, String, \
    ConstDict, Alternative, _types


class TypesMetadataTest(unittest.TestCase):
    def test_tupleOf(self):
        self.assertEqual(TupleOf(int), TupleOf(Int64))
        self.assertEqual(TupleOf(int).ElementType, Int64)

        self.assertEqual(TupleOf(float).ElementType, Float64)
        self.assertEqual(TupleOf(OneOf(10, 20)).ElementType, OneOf(10, 20))

        self.assertEqual(TupleOf(object).ElementType.__typed_python_category__, "PythonObjectOfType")
        self.assertEqual(TupleOf(10).ElementType.__typed_python_category__, "Value")

    def test_tuple(self):
        self.assertEqual(Tuple(int, int, OneOf(10, 20)).ElementTypes, (Int64, Int64, OneOf(10, 20)))

    def test_named_tuple(self):
        self.assertEqual(NamedTuple(x=int, y=int, z=OneOf(10, 20)).ElementTypes, (Int64, Int64, OneOf(10, 20)))
        self.assertEqual(NamedTuple(x=int, y=int, z=OneOf(10, 20)).ElementNames, ('x', 'y', 'z'))

    def test_const_dict(self):
        self.assertEqual(ConstDict(str, int).KeyType, String)
        self.assertEqual(ConstDict(str, int).ValueType, Int64)

    def test_alternatives(self):
        X = Alternative(
            "X",
            Left={'x': int, 'y': str},
            Right={'x': lambda: X, 'val': int}
        )

        _types.resolveForwards(X)

        self.assertEqual(len(X.__typed_python_alternatives__), 2)

        Left, Right = X.__typed_python_alternatives__

        self.assertEqual(Left.Index, 0)
        self.assertEqual(Right.Index, 1)

        self.assertEqual(Left.ElementType.ElementNames, ("x", "y"))
        self.assertEqual(Left.ElementType.ElementTypes, (Int64, String))
        self.assertEqual(Right.ElementType.ElementNames, ('x', 'val'))
        self.assertEqual(Right.ElementType.ElementTypes, (X, Int64))
