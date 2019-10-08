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
import typed_python.compiler.python_ast_analysis as python_ast_analysis

import unittest


class TestPythonAstAnalysis(unittest.TestCase):
    def freeVarCheck(self, func, readVars, assignedVars):
        """Look at func's _body_ only, and compute which variables it reads and assigns."""
        pyast = python_ast.convertFunctionToAlgebraicPyAst(func)

        assert pyast.matches.FunctionDef or pyast.matches.Lambda, type(pyast)

        self.assertEqual(
            set(python_ast_analysis.computeAssignedVariables(pyast.body)),
            set(assignedVars)
        )
        self.assertEqual(
            set(python_ast_analysis.computeReadVariables(pyast.body)),
            set(readVars)
        )

    def test_free_vars_basic(self):
        self.freeVarCheck(lambda: x, ['x'], [])  # noqa
        self.freeVarCheck(lambda: x + y, ['x', 'y'], [])  # noqa
        self.freeVarCheck(lambda: f(*args), ['f', 'args'], [])  # noqa

    def test_assignment_shows_up(self):
        def f():
            x = 10
            return x

        self.freeVarCheck(f, ['x'], ['x'])

    def test_multi_assignment_shows_up(self):
        def f():
            x, y = (1, 2)  # noqa

        self.freeVarCheck(f, [], ['x', 'y'])

    def test_for_loops(self):
        def f():
            for i in 10:
                pass

        self.freeVarCheck(f, [], ['i'])

    def test_function_defs(self):
        def f():
            def aFun(x):
                return x + y  # noqa

            def aFun2(*args, **kwargs):
                return args + kwargs

        self.freeVarCheck(f, ['y'], ['aFun', 'aFun2'])

    def test_class_defs(self):
        def f():
            class AClass:
                # this doesn't mask the 'z' below!
                z = 20

                def f(x):
                    return z  # noqa

        self.freeVarCheck(f, ['z'], ['AClass'])
