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

from typed_python import TypeFunction, Entrypoint, ListOf, Value, TupleOf
import unittest


class CompileTypeFunctionTest(unittest.TestCase):
    def test_basic(self):
        @TypeFunction
        def List(T):
            return ListOf(T)

        @Entrypoint
        def canInferType(T):
            return List(T)

        self.assertEqual(
            canInferType.resultTypeFor(Value(int)).typeRepresentation.Value,
            ListOf(int)
        )

    def test_pass_type_function_as_value(self):
        @TypeFunction
        def List(T):
            return ListOf(T)

        @Entrypoint
        def inferType(TF, T):
            return TF(T)

        self.assertEqual(
            inferType.resultTypeFor(Value(List), Value(int)).typeRepresentation.Value,
            ListOf(int)
        )

        self.assertEqual(inferType(List, int), ListOf(int))

    def test_instantiating_type_function_value(self):
        @TypeFunction
        def List(T):
            return ListOf(T)

        @Entrypoint
        def convertToList(aContainer, TF):
            T = type(aContainer).ElementType
            MyList = TF(T)
            assert MyList == ListOf(T)
            return MyList()

        self.assertEqual(
            type(convertToList(TupleOf(int)(), List)),
            ListOf(int)
        )

        self.assertEqual(
            convertToList.resultTypeFor(TupleOf(int), Value(List)).typeRepresentation,
            ListOf(int)
        )
