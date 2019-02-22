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

from typed_python import *
import typed_python.python_ast as python_ast

import os
import unittest

ownName = os.path.abspath(__file__)


class TestPythonAst(unittest.TestCase):
    def test_basic_parsing(self):
        pyast = python_ast.convertFunctionToAlgebraicPyAst(lambda: X)

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
        self.reverseParseCheck(lambda: X)
        self.reverseParseCheck(lambda x: X)
        self.reverseParseCheck(lambda x: X+1)
        self.reverseParseCheck(lambda x: 1.2)
        self.reverseParseCheck(lambda x: "hi")
        self.reverseParseCheck(lambda x: (x,True))
        self.reverseParseCheck(lambda x: (x,None))
        self.reverseParseCheck(lambda x: x.asdf)
        self.reverseParseCheck(lambda x: x(10))

        def f(x):
            try:
                A()
            except Exception as e:
                pass
            else:
                print("hihi")

        self.reverseParseCheck(f)

    def test_reverse_parse_classdef(self):
        def f():
            class A:
                z = 20
                @staticmethod
                def y(self, z:int):
                    pass

                @otherDecordator
                def z(self, z: int, *args: list, **kwargs: dict) -> float:
                    while a < b < c:
                        pass
                    return 12

        self.reverseParseCheck(f)

    def test_reverse_parse_functions_with_keywords(self):
        def f():
            def g(x=10, y=20, *args, q=30):
                return (x,y,args,q)
            g(x=20, y=30)

        self.reverseParseCheck(f)

    def test_reverse_parse_comprehensions(self):
        def f():
            [x for x in y]
            [x for x in y for z in q]
            {k:v for k in v}

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
        self.reverseParseAndEvalCheck(lambda x:x+x, 10)

    def test_reverse_parse_eval_withblock(self):
        def f(x, filename):
            with open(filename, 'r') as f:
                pass
            return x+x

        self.reverseParseAndEvalCheck(f, (10, ownName))

    def test_reverse_parse_eval_import(self):
        def f(x):
            from numpy import float64
            return float(float64(x))
        self.reverseParseAndEvalCheck(f, 10)
