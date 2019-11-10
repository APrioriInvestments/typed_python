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

from typed_python import Function, OneOf, Int64, Float64, Alternative, Value, String, ListOf
from typed_python.compiler.runtime import Runtime, Entrypoint
import unittest
import pytest


A = Alternative(
    "A",
    X={'x': int},
    Y={'x': str}
)


def Compiled(f):
    f = Function(f)
    return Runtime.singleton().compile(f)


def resultType(f, **kwargs):
    return Runtime.singleton().resultTypes(f, kwargs)


class TestTypeInference(unittest.TestCase):
    def test_basic_inference(self):
        def f(x, y):
            return x + y

        self.assertEqual(resultType(f, x=int, y=int), Int64)
        self.assertEqual(resultType(f, x=int, y=float), Float64)
        self.assertEqual(resultType(f, x=int, y=str), None)

    def test_sequential_assignment(self):
        def f(x):
            y = "hi"
            y = x
            return y

        self.assertEqual(resultType(f, x=int), Int64)

    def test_if_short_circuit(self):
        def f(x):
            if True:
                y = x
            else:
                y = "hi"
            return y

        self.assertEqual(resultType(f, x=int), Int64)

    def test_if_merging(self):
        def f(x):
            if x % 2 == 0:
                y = x
            else:
                y = "hi"
            return y

        self.assertEqual(set(resultType(f, x=int).Types), set([Int64, String]))

    def test_isinstance_propagates(self):
        def f(x):
            if isinstance(x, int):
                return x
            else:
                return 0

        self.assertEqual(resultType(f, x=OneOf(str, int)), Int64)

    def test_alternative_inference(self):
        def f(anA):
            return anA.x

        self.assertEqual(set(resultType(f, anA=A).Types), set([Int64, String]))

    def test_alternative_inference_with_branch(self):
        def f(anA):
            if anA.matches.X:
                return anA.x
            else:
                return 0

        self.assertEqual(resultType(f, anA=A), Int64)

    def test_alternative_inference_with_nonsense_branch(self):
        def f(anA):
            if anA.matches.NotReal:
                return "compiler realizes we can't get here"
            else:
                return 0

        self.assertEqual(resultType(f, anA=A), Int64)

    def test_no_result_from_while_true(self):
        def f(x):
            while True:
                x = x + 1
                if x > 10000:
                    return x

        self.assertEqual(resultType(f, x=int), Int64)
        self.assertEqual(Entrypoint(f)(10), 10001)

    def test_infer_type_object(self):
        def f(x):
            return type(x)

        self.assertEqual(resultType(f, x=str), Value(str))
        self.assertEqual(set(resultType(f, x=OneOf(int, str)).Types), set([Value(int), Value(str)]))

    @pytest.mark.skip(reason="to be addressed")
    def test_infer_list_item(self):
        a = ListOf(str)(("a", "b", "c"))
        b = [1, 2.5, "x"]

        def f(x: int):
            return a[x]

        def g(x: int):
            return b[x]

        # Currently: resultType is None
        self.assertEqual(resultType(f, x=int), String)
        # Currently: raises NotImplementedError
        self.assertEqual(resultType(g, x=int), object)  # what should the resultType be?

    def test_infer_conditional_eval_exception(self):
        def exc():
            raise Exception('error')

        def and1(x, y):
            return exc() and x and y

        def and2(x, y):
            return x and exc() and y

        def and3(x, y):
            return x and y and exc()

        def or1(x, y):
            return exc() or x or y

        def or2(x, y):
            return x or exc() or y

        def or3(x, y):
            return x or y or exc()

        self.assertEqual(resultType(and1, x=int, y=float), None)
        self.assertEqual(resultType(and2, x=int, y=float), Int64)
        self.assertEqual(resultType(and2, x=float, y=int), Float64)
        self.assertEqual(resultType(and3, x=int, y=float), OneOf(Float64, Int64))
        self.assertEqual(resultType(and3, x=float, y=int), OneOf(Float64, Int64))
        self.assertEqual(resultType(or1, x=int, y=float), None)
        self.assertEqual(resultType(or2, x=int, y=float), Int64)
        self.assertEqual(resultType(or2, x=float, y=int), Float64)
        self.assertEqual(resultType(or3, x=int, y=float), OneOf(Float64, Int64))
        self.assertEqual(resultType(or3, x=float, y=int), OneOf(Float64, Int64))
