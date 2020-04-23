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
    Function, OneOf, Int64, Float64, Alternative,
    Value, String, ListOf, NotCompiled
)
from typed_python.compiler.runtime import Entrypoint
import unittest


A = Alternative(
    "A",
    X={'x': int},
    Y={'x': str}
)


class TestTypeInference(unittest.TestCase):
    def test_basic_inference(self):
        @Function
        def f(x, y):
            return x + y

        self.assertEqual(f.resultTypeFor(int, int).typeRepresentation, Int64)
        self.assertEqual(f.resultTypeFor(int, float).typeRepresentation, Float64)
        self.assertEqual(f.resultTypeFor(int, str), None)

    def test_sequential_assignment(self):
        @Function
        def f(x):
            y = "hi"
            y = x
            return y

        self.assertEqual(f.resultTypeFor(int).typeRepresentation, Int64)

    def test_if_short_circuit(self):
        @Function
        def f(x):
            if True:
                y = x
            else:
                y = "hi"
            return y

        self.assertEqual(f.resultTypeFor(int).typeRepresentation, Int64)

    def test_if_merging(self):
        @Function
        def f(x):
            if x % 2 == 0:
                y = x
            else:
                y = "hi"
            return y

        self.assertEqual(set(f.resultTypeFor(int).typeRepresentation.Types), set([Int64, String]))

    def test_isinstance_propagates(self):
        @Function
        def f(x):
            if isinstance(x, int):
                return x
            else:
                return 0

        self.assertEqual(f.resultTypeFor(OneOf(str, int)).typeRepresentation, Int64)

    def test_alternative_inference(self):
        @Function
        def f(anA):
            return anA.x

        self.assertEqual(set(f.resultTypeFor(A).typeRepresentation.Types), set([Int64, String]))

    def test_alternative_inference_with_branch(self):
        @Function
        def f(anA):
            if anA.matches.X:
                return anA.x
            else:
                return 0

        self.assertEqual(f.resultTypeFor(A).typeRepresentation, Int64)

    def test_alternative_inference_with_nonsense_branch(self):
        @Function
        def f(anA):
            if anA.matches.NotReal:
                return "compiler realizes we can't get here"
            else:
                return 0

        self.assertEqual(f.resultTypeFor(A).typeRepresentation, Int64)

    def test_no_result_from_while_true(self):
        @Function
        def f(x):
            while True:
                x = x + 1
                if x > 10000:
                    return x

        self.assertEqual(f.resultTypeFor(int).typeRepresentation, Int64)
        self.assertEqual(Entrypoint(f)(10), 10001)

    def test_infer_type_object(self):
        @Function
        def f(x):
            return type(x)

        self.assertEqual(f.resultTypeFor(str).typeRepresentation, Value(str))
        self.assertEqual(
            set(f.resultTypeFor(OneOf(int, str)).typeRepresentation.Types),
            set([Value(int), Value(str)])
        )

    def test_infer_list_item(self):
        @Function
        def f(a: ListOf(str), x: int):
            return a[x]

        @Function
        def g(b, x: int):
            return b[x]

        self.assertEqual(f.resultTypeFor(ListOf(str), int).typeRepresentation, String)
        self.assertEqual(g.resultTypeFor(object, int).typeRepresentation.PyType, object)

    def test_infer_conditional_eval_exception(self):
        @Function
        def exc():
            raise Exception('error')

        @Function
        def and1(x, y):
            return exc() and x and y

        @Function
        def and2(x, y):
            return x and exc() and y

        @Function
        def and3(x, y):
            return x and y and exc()

        @Function
        def or1(x, y):
            return exc() or x or y

        @Function
        def or2(x, y):
            return x or exc() or y

        @Function
        def or3(x, y):
            return x or y or exc()

        self.assertEqual(exc.resultTypeFor(), None)
        self.assertEqual(and1.resultTypeFor(int, float), None)
        self.assertEqual(and2.resultTypeFor(int, float).typeRepresentation, Int64)
        self.assertEqual(and2.resultTypeFor(float, int).typeRepresentation, Float64)
        self.assertEqual(set(and3.resultTypeFor(int, float).typeRepresentation.Types), set([Int64, Float64]))
        self.assertEqual(set(and3.resultTypeFor(float, int).typeRepresentation.Types), set([Int64, Float64]))
        self.assertEqual(or1.resultTypeFor(int, float), None)
        self.assertEqual(or2.resultTypeFor(int, float).typeRepresentation, Int64)
        self.assertEqual(or2.resultTypeFor(float, int).typeRepresentation, Float64)
        self.assertEqual(set(or3.resultTypeFor(int, float).typeRepresentation.Types), set([Int64, Float64]))
        self.assertEqual(set(or3.resultTypeFor(float, int).typeRepresentation.Types), set([Int64, Float64]))

    def test_infer_type_of_assignment_with_guard(self):
        @Function
        def f(x):
            if isinstance(x, int):
                y = x
            else:
                y = 0

            return y

        self.assertEqual(f.resultTypeFor(OneOf(None, int)).typeRepresentation, Int64)
        self.assertEqual(f.resultTypeFor(OneOf(None, int, float)).typeRepresentation, Int64)
        self.assertEqual(f.resultTypeFor(int).typeRepresentation, Int64)
        self.assertEqual(f.resultTypeFor(None).typeRepresentation, Int64)
        self.assertEqual(f.resultTypeFor(float).typeRepresentation, Int64)

    def test_infer_type_of_nocompile(self):
        @NotCompiled
        def f() -> int:
            # currently, we can't compile this
            return [x for x in range(10)][0]

        self.assertEqual(f.resultTypeFor().typeRepresentation, Int64)

        @NotCompiled
        def f2():
            # currently, we can't compile this
            return [x for x in range(10)][0]

        self.assertEqual(f2.resultTypeFor().typeRepresentation.PyType, object)
