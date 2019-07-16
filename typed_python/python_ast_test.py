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

import typed_python.python_ast as python_ast

import os
import unittest

ownName = os.path.abspath(__file__)


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

        def f(x):
            try:
                A()  # noqa: F821
            except Exception:
                pass
            else:
                print("hihi")

        self.reverseParseCheck(f)

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
