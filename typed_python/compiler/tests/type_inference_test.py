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
    Function, OneOf, Alternative,
    Value, ListOf, NotCompiled, TupleOf, Tuple, UInt8,
    typeKnownToCompiler,
    localVariableTypesKnownToCompiler
)
from typed_python.compiler.runtime import Entrypoint
from typed_python.compiler.runtime import Runtime
import unittest
import time

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

        self.assertEqual(f.resultTypeFor(int, int).typeRepresentation, int)
        self.assertEqual(f.resultTypeFor(int, float).typeRepresentation, float)
        self.assertEqual(f.resultTypeFor(int, str), None)

    def test_sequential_assignment(self):
        @Function
        def f(x):
            y = "hi"
            y = x
            return y

        self.assertEqual(f.resultTypeFor(int).typeRepresentation, int)

    def test_if_short_circuit(self):
        @Function
        def f(x):
            if True:
                y = x
            else:
                y = "hi"
            return y

        self.assertEqual(f.resultTypeFor(int).typeRepresentation, int)

    def test_if_merging(self):
        @Function
        def f(x):
            if x % 2 == 0:
                y = x
            else:
                y = "hi"
            return y

        self.assertEqual(set(f.resultTypeFor(int).typeRepresentation.Types), set([int, str]))

    def test_isinstance_propagates(self):
        @Function
        def f(x):
            if isinstance(x, int):
                return x
            else:
                return 0

        self.assertEqual(f.resultTypeFor(OneOf(str, int)).typeRepresentation, int)

    def test_alternative_inference(self):
        @Function
        def f(anA):
            return anA.x

        self.assertEqual(set(f.resultTypeFor(A).typeRepresentation.Types), set([int, str]))

    def test_alternative_inference_with_branch(self):
        @Function
        def f(anA):
            if anA.matches.X:
                return anA.x
            else:
                return 0

        self.assertEqual(f.resultTypeFor(A).typeRepresentation, int)

    def test_alternative_inference_with_nonsense_branch(self):
        @Function
        def f(anA):
            if anA.matches.NotReal:
                return "compiler realizes we can't get here"
            else:
                return 0

        self.assertEqual(f.resultTypeFor(A).typeRepresentation, int)

    def test_no_result_from_while_true(self):
        @Function
        def f(x):
            while True:
                x = x + 1
                if x > 10000:
                    return x

        self.assertEqual(f.resultTypeFor(int).typeRepresentation, int)
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

    def test_infer_result_of_uint8(self):
        @Entrypoint
        def f(x):
            return UInt8(x)

        self.assertEqual(f.resultTypeFor(float).typeRepresentation, UInt8)

    def test_infer_result_of_uint8_constant(self):
        @Entrypoint
        def f():
            return UInt8(10)

        self.assertEqual(f.resultTypeFor().typeRepresentation, UInt8)

    def test_infer_list_item(self):
        @Function
        def f(a: ListOf(str), x: int):
            return a[x]

        @Function
        def g(b, x: int):
            return b[x]

        self.assertEqual(f.resultTypeFor(ListOf(str), int).typeRepresentation, str)
        self.assertEqual(g.resultTypeFor(object, int).typeRepresentation, object)

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
        self.assertEqual(and2.resultTypeFor(int, float).typeRepresentation, int)
        self.assertEqual(and2.resultTypeFor(float, int).typeRepresentation, float)
        self.assertEqual(set(and3.resultTypeFor(int, float).typeRepresentation.Types), set([int, float]))
        self.assertEqual(set(and3.resultTypeFor(float, int).typeRepresentation.Types), set([int, float]))
        self.assertEqual(or1.resultTypeFor(int, float), None)
        self.assertEqual(or2.resultTypeFor(int, float).typeRepresentation, int)
        self.assertEqual(or2.resultTypeFor(float, int).typeRepresentation, float)
        self.assertEqual(set(or3.resultTypeFor(int, float).typeRepresentation.Types), set([int, float]))
        self.assertEqual(set(or3.resultTypeFor(float, int).typeRepresentation.Types), set([int, float]))

    def test_infer_type_of_assignment_with_guard(self):
        @Function
        def f(x):
            if isinstance(x, int):
                y = x
            else:
                y = 0

            return y

        self.assertEqual(f.resultTypeFor(OneOf(None, int)).typeRepresentation, int)
        self.assertEqual(f.resultTypeFor(OneOf(None, int, float)).typeRepresentation, int)
        self.assertEqual(f.resultTypeFor(int).typeRepresentation, int)
        self.assertEqual(f.resultTypeFor(None).typeRepresentation, int)
        self.assertEqual(f.resultTypeFor(float).typeRepresentation, int)

    def test_infer_type_of_nocompile(self):
        @NotCompiled
        def f() -> int:
            # currently, we can't compile this
            return [x for x in range(10)][0]

        self.assertEqual(f.resultTypeFor().typeRepresentation, int)

        @NotCompiled
        def f2():
            # currently, we can't compile this
            return [x for x in range(10)][0]

        self.assertEqual(f2.resultTypeFor().typeRepresentation, object)

    def test_compiler_can_see_through_explicitly_constructed_typed_tuples(self):
        @Entrypoint
        def returnTupElts(x):
            return TupleOf(int)((1, 2, 3 + x))[2]

        assert returnTupElts.resultTypeFor(int).typeRepresentation == int

    def test_compiler_can_see_through_untyped_tuples(self):
        @Entrypoint
        def returnTupElts(x):
            return (1, 2.2, 3 + x)[2]

        assert returnTupElts.resultTypeFor(int).typeRepresentation == int

    def test_compiler_can_merge_like_untyped_tuples(self):
        @Entrypoint
        def returnTupElts(x):
            if x > 10:
                aTup = (1, 2.2, 3 + x)
            else:
                aTup = (1, 2.2, 3 + x + x)

            return aTup[2]

        assert returnTupElts.resultTypeFor(int).typeRepresentation == int

    def test_compiler_can_converts_unlike_untyped_tuples_to_object(self):
        @Entrypoint
        def returnTupElts(x):
            if x > 10:
                aTup = (1, 2.2, 3 + x)
            else:
                aTup = (1, "2.2", 3 + x + x)

            return aTup

        assert returnTupElts.resultTypeFor(int).typeRepresentation is tuple

    def test_tuple_as_variable_traces_properly(self):
        @Entrypoint
        def returnTupElts(x):
            aTup = (1, 2, x)

            return Tuple(int, int, float)(aTup)

        assert returnTupElts.resultTypeFor(float).typeRepresentation is Tuple(int, int, float)

    def test_compiler_knows_type_of_arguments(self):
        @Entrypoint
        def returnTypeOfArgument(x):
            return typeKnownToCompiler(x)

        assert returnTypeOfArgument(10) is int
        assert returnTypeOfArgument(10.5) is float
        assert returnTypeOfArgument(float) is Value(float)

    def test_compiler_knows_it_has_oneofs(self):
        @Entrypoint
        def returnTypeOfArgument(x):
            if x > 0:
                y = 0
            else:
                y = 1.0

            return typeKnownToCompiler(y)

        assert returnTypeOfArgument(10) is OneOf(float, int)

    def test_compiler_knows_that_isinstance_constrains_types(self):
        @Entrypoint
        def returnTypeOfArgument(x):
            if x > 0:
                y = 0
            else:
                y = 1.0

            if isinstance(y, float):
                return typeKnownToCompiler(y)
            elif isinstance(y, int):
                return typeKnownToCompiler(y)

        assert returnTypeOfArgument(10) is int
        assert returnTypeOfArgument(0) is float

    def test_compiler_knows_local_variable_types(self):
        @Entrypoint
        def localVariableTypes(x):
            if x > 0:
                y = 0 # noqa
            else:
                y = 1.0 # noqa

            return localVariableTypesKnownToCompiler()

        res = localVariableTypes(10)
        assert res == dict(x=int, y=OneOf(float, int))

    def test_type_inference_perf(self):
        t0 = time.time()
        f = Function(lambda o: o + 1)

        Runtime.singleton().resultTypeForCall(f, [int], {})

        for _ in range(10000):
            Runtime.singleton().resultTypeForCall(f, [int], {})

        # I get around 0.03 on my desktop with the caching, and .7 without it
        assert time.time() - t0 < .4

    def test_type_inference_with_exceptions(self):
        # check that if we catch an arbitrary exception, then we can't really
        # know what the type of a variable coming out of the exception block.
        # this is a little crude because obviously we can tell below because we
        # know we are not throwing exceptions, but we want to ensure that we don't
        # mistakenly assume it's 'str'.
        @Entrypoint
        def check():
            try:
                y = 10
                y = "hi"  # noqa
            except Exception:
                pass

            return localVariableTypesKnownToCompiler()

        assert check() == dict(y=OneOf(int, str))

        # but if we return from the handler directly, then the only way to get to the
        # code after the except is if everything passed.
        @Entrypoint
        def check2():
            try:
                y = 10
                y = "hi"  # noqa
            except Exception:
                return

            return localVariableTypesKnownToCompiler()

        assert check2()['y'] == str

        # check that the 'finally' block knows it could be the merge
        @Entrypoint
        def check3():
            try:
                y = 10
                y = "hi"  # noqa
            except Exception:
                y = 12.5  # noqa
            finally:
                return localVariableTypesKnownToCompiler()

        assert check3()['y'] == OneOf(float, str)
