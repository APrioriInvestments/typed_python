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

    def test_assigned_only_once(self):
        def f():
            x = 10
            x = 20  # noqa
            y = 30  # noqa

            def f():
                pass

            def f():  # noqa
                pass

            def g():
                pass

            def h():
                def h1():
                    pass

                def h1():  # noqa
                    pass

            def h1():
                pass

        pyast = python_ast.convertFunctionToAlgebraicPyAst(f)

        assignedOnce = python_ast_analysis.computeVariablesAssignedOnlyOnce(pyast.body)
        assignmentCount = python_ast_analysis.computeVariablesAssignmentCounts(pyast.body)

        self.assertEqual(sorted(assignedOnce), ["g", "h", "h1", "y"])
        self.assertEqual(assignmentCount, dict(x=2, y=1, f=2, g=1, h=1, h1=1))

    def test_read_by_closures(self):
        def f():
            def f1():
                return x  # noqa

            def f2():
                def g():
                    return y  # noqa

            print(printVar)  # noqa

            class Y:
                def y():
                    return z  # noqa

        pyast = python_ast.convertFunctionToAlgebraicPyAst(f)

        readByClosures = python_ast_analysis.computeVariablesReadByClosures(pyast.body)

        self.assertEqual(sorted(readByClosures), ["x", "y", "z"])

    def test_func_def_masking_self(self):
        def f():
            class C:
                def func(self):
                    return func()  # noqa

        pyast = python_ast.convertFunctionToAlgebraicPyAst(f)

        readByClosures = python_ast_analysis.computeVariablesReadByClosures(pyast.body)

        self.assertEqual(sorted(readByClosures), ["func"])

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

    def test_function_default_args(self):
        SomeVar = None

        def f():
            def someFunc(x=SomeVar()):
                return x

        self.freeVarCheck(f, ['SomeVar'], ['someFunc'])

    def test_function_annotations(self):
        Something = None

        def f():
            def someFunc(x: Something):
                return x

        self.freeVarCheck(f, ['Something'], ['someFunc'])

        def f():
            def someFunc(*x: Something):
                return x

        self.freeVarCheck(f, ['Something'], ['someFunc'])

        def f():
            def someFunc(**x: Something):
                return x

        self.freeVarCheck(f, ['Something'], ['someFunc'])

    def test_iterators_in_generators_not_assigned(self):
        something = None
        z = None

        def f():
            x = [y + z for y in something]  # noqa

        self.freeVarCheck(f, ['something', 'z'], ['x'])

    def test_iterators_in_generators_masking(self):
        # this does read from 'y' even though it masks the 'y' inside
        y = None
        z = None
        something = None

        def f():
            x = [y for y in y]  # noqa

        self.freeVarCheck(f, ['y'], ['x'])

        # this doesn't read from 'y'
        def f():
            x = [y for y in something if y]  # noqa

        self.freeVarCheck(f, ['something'], ['x'])

        # this doesn't read from 'y' or 'z' because they're bound before usage
        def f():
            x = [z for y in something if y for z in y]  # noqa

        self.freeVarCheck(f, ['something'], ['x'])

        # this reads from 'z' because 'z' gets used in the 'if' that happens before 'z' gets
        # bound
        def f():
            x = [z for y in something if y or z for z in y]  # noqa

        self.freeVarCheck(f, ['something', 'z'], ['x'])

    def test_import_aliases(self):
        def f():
            import a.b  # noqa
            import e.f as blah  # noqa

        pyast = python_ast.convertFunctionToAlgebraicPyAst(f)

        assert python_ast_analysis.computeAssignedVariables(pyast.body) == {'a', 'blah'}
