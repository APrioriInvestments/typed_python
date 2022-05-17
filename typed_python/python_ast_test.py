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

import typed_python.python_ast as python_ast
import tempfile
import os
import unittest

ownName = os.path.abspath(__file__)

# needed as a module-level variable by a test below
someVarname = 10


class TestPythonAst(unittest.TestCase):
    def test_basic_parsing(self):
        pyast = python_ast.convertFunctionToAlgebraicPyAst(lambda: X)  # noqa: F821

        self.assertTrue(pyast.matches.Lambda)
        self.assertTrue(pyast.body.matches.Name)
        self.assertEqual(pyast.body.id, "X")

    def reverseParseCheck(self, f):
        pyast = python_ast.convertFunctionToAlgebraicPyAst(f)
        native_ast = python_ast.convertAlgebraicToPyAst(pyast)
        pyast2 = python_ast.convertPyAstToAlgebraic(native_ast, pyast.filename)
        f2 = python_ast.evaluateFunctionPyAst(pyast2)
        pyast3 = python_ast.convertFunctionToAlgebraicPyAst(f2)

        self.assertEqual(pyast, pyast2)
        self.assertEqual(pyast, pyast3)

        return f2

    def test_reverse_parse(self):
        self.reverseParseCheck(lambda: X)  # noqa: F821
        self.reverseParseCheck(lambda x: X)  # noqa: F821
        self.reverseParseCheck(lambda x: X+1)  # noqa: F821
        self.reverseParseCheck(lambda x: 1.2)
        self.reverseParseCheck(lambda x: "hi")
        self.reverseParseCheck(lambda x: (x, True))
        self.reverseParseCheck(lambda x: (x, None))
        self.reverseParseCheck(lambda x: x.asdf)
        self.reverseParseCheck(lambda x: x(10))
        self.reverseParseCheck(lambda x: x[:10])

        def f(x):
            try:
                A()  # noqa: F821
            except Exception:
                pass
            else:
                print("hihi")

        self.reverseParseCheck(f)

    def test_ast_for_function_with_decorator(self):
        # check that the ast for a function with a decorator doesn't actually
        # include the decorator, as the decorator isn't part of the function
        # itself.
        def identity(x):
            return x

        @identity
        def f(x):
            return x

        pyast = python_ast.convertFunctionToAlgebraicPyAst(f)

        assert pyast.matches.FunctionDef
        assert not pyast.decorator_list

    def test_reverse_parse_classdef(self):
        def f():
            class A:
                z = 20

                @staticmethod
                def y(self, z: int):
                    pass

                @otherDecordator  # noqa: F821
                def z(self, z: int, *args: list, **kwargs: dict) -> float:
                    while a < b < c:  # noqa: F821
                        pass
                    return 12

        self.reverseParseCheck(f)

    def test_reverse_parse_functions_with_keywords(self):
        def f():
            def g(x=10, y=20, *args, q=30):
                return (x, y, args, q)
            g(x=20, y=30)

        self.reverseParseCheck(f)

    def test_reverse_parse_comprehensions(self):
        def f():
            [x for x in y]  # noqa: F821
            [x for x in y for z in q]  # noqa: F821
            {k: v for k in v}  # noqa: F821

        self.reverseParseCheck(f)

    def reverseParseAndEvalCheck(self, f, args):
        try:
            iter(args)
        except TypeError:
            args = tuple([args])

        pyast = python_ast.convertFunctionToAlgebraicPyAst(f)

        f_2 = python_ast.evaluateFunctionPyAst(pyast)

        self.assertEqual(f(*args), f_2(*args))

    def test_reverse_parse_eval(self):
        def f(x):
            return x+x
        self.reverseParseAndEvalCheck(f, 10)
        self.reverseParseAndEvalCheck(lambda x: x+x, 10)

    def test_reverse_parse_eval_withblock(self):
        def f(x, filename):
            with open(filename, 'r'):
                pass
            return x+x

        self.reverseParseAndEvalCheck(f, (10, ownName))

    def test_reverse_parse_eval_import(self):
        def f(x):
            from numpy import float64
            return float(float64(x))
        self.reverseParseAndEvalCheck(f, 10)

    def test_parsing_fstring(self):
        """
        This code generates:
            - ast.JoinedStr
            - ast.FormattedValue
        """
        self.reverseParseCheck(lambda x: f" - {x} - ")

    def test_parsing_assert(self):
        """This code generates ast.Assert node."""
        def f(x):
            assert x == 1
        self.reverseParseCheck(f)

    def test_parsing_yield_from(self):
        """This code generates ast.YieldFrom."""
        def f(x):
            yield from x
        self.reverseParseCheck(f)

    def test_parsing_async(self):
        """This code generates:
            - ast.AsyncFunctionDef
            - ast.Await
            - ast.AsyncWith
        """
        import asyncio

        async def f(x, y):
            print("Compute %s + %s ..." % (x, y))
            await asyncio.sleep(1.0)
            async for i in x:
                print(i)
            async with y:
                pass
            return x + y

        self.reverseParseCheck(f)

    def test_parsing_nonlocal(self):
        """This code generates: ast.Nonlocal."""
        def f():
            x = "local"

            def inner():
                nonlocal x
                x = "nonlocal"
                print("inner:", x)

            inner()
            print("outer:", x)

        self.reverseParseCheck(f)

    def test_parsing_matmul(self):
        """This code generates: ast.MatMult."""
        def f(a, b):
            a @ b
            a @= b

        self.reverseParseCheck(f)

    def test_parsing_annotated_assignment(self):
        """This code generates: ast.AddAssign."""
        def f():
            a: int = 2
            return a

        self.reverseParseCheck(f)

    def test_parsing_bytes(self):
        def f():
            return b"aaa"

        self.reverseParseCheck(f)

    def test_converting_lambdas_in_expressions(self):
        def identity(x):
            return x
        someLambdas = (
            identity(lambda x: x + 1),
            identity(lambda x: x + 2),
            identity(lambda x: x + 3),
            identity(lambda x: x + 4),
            identity(lambda x: x + 5),
            identity(lambda x: x + 6),
            identity(lambda x: x + 7)
        )

        for lam in someLambdas:
            self.assertEqual(self.reverseParseCheck(lam)(10), lam(10))

    def test_converting_two_lambdas_on_same_line(self):
        someLambdas = (lambda x: x + 1, lambda x: x + 2)

        for lam in someLambdas:
            self.assertEqual(self.reverseParseCheck(lam)(10), lam(10))

    def test_converting_two_lambdas_with_similar_bodies_but_different_args_on_same_line(self):
        # shouldn't matter which one we pick
        someLambdas = (lambda x, y: x + 1, lambda x, z: x + 1)

        for lam in someLambdas:
            self.assertEqual(self.reverseParseCheck(lam)(10, 11), lam(10, 11))

    def test_converting_two_identical_lambdas_on_same_line(self):
        # shouldn't matter which one we pick
        someLambdas = (lambda x: x + 1, lambda x: x + 1)

        for lam in someLambdas:
            self.assertEqual(self.reverseParseCheck(lam)(10), lam(10))

    def test_converting_lambdas_pulled_out_of_binding(self):
        # shouldn't matter which one we pick
        aLambda = (lambda x: (lambda y: x + y))(10)

        pyast = python_ast.convertFunctionToAlgebraicPyAst(aLambda)
        aLambda2 = python_ast.evaluateFunctionPyAst(pyast, globals={'x': 10})

        self.assertEqual(aLambda(11), aLambda2(11))

    def test_converting_lambda_with_double_star_dicts(self):
        def f():
            x = {1: 2}
            y = {**x}
            return y

        self.assertEqual(self.reverseParseCheck(f)(), f())

    def test_conflicting_code_objects_for_list_comps(self):
        with tempfile.TemporaryDirectory() as tf:
            fname1 = os.path.join(tf, "a.py")
            fname2 = os.path.join(tf, "b.py")

            CODE1 = "def f(y):\n    return [x for x in lookupVar1]\n"
            CODE2 = "def f(y):\n    return [x for x in lookupVar2]\n"

            with open(fname1, "w") as f:
                f.write(CODE1)
            with open(fname2, "w") as f:
                f.write(CODE2)

            c1 = compile(CODE1, fname1, "exec").co_consts[0]
            c2 = compile(CODE2, fname2, "exec").co_consts[0]

            python_ast.convertFunctionToAlgebraicPyAst(c1)
            python_ast.convertFunctionToAlgebraicPyAst(c2)

            assert 'listcomp' in str(c1.co_consts[1])
            assert 'listcomp' in str(c2.co_consts[1])

            listCompAst1 = python_ast.convertFunctionToAlgebraicPyAst(c1.co_consts[1])
            listCompAst2 = python_ast.convertFunctionToAlgebraicPyAst(c2.co_consts[1])

            assert listCompAst1 == listCompAst2
            assert 'lookupVar' not in str(listCompAst1)
