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
from typed_python import Int64, NoneType, TupleOf, ConstDict, Function


class NativeFunctionTypesTests(unittest.TestCase):
    def test_create_simple_function(self):
        @Function
        def f(x: int) -> int:
            return x + 1

        self.assertEqual(f(2), 3)

        with self.assertRaises(TypeError):
            f(3.5)

        self.assertEqual(len(f.overloads), 1)
        o = f.overloads[0]

        self.assertEqual(o.returnType, Int64)

        self.assertEqual(len(o.args), 1)
        self.assertEqual(o.args[0].name, "x")
        self.assertEqual(o.args[0].typeFilter, Int64)
        self.assertEqual(o.args[0].defaultValue, None)
        self.assertEqual(o.args[0].isStarArg, False)
        self.assertEqual(o.args[0].isKwarg, False)

    def test_create_function_with_kwargs_and_star_args_and_defaults(self):
        @Function
        def f(x: int, y=30, z: None = None, *args: TupleOf(float), **kwargs: ConstDict(str, float)) -> int:
            return x + 1

        self.assertEqual(len(f.overloads), 1)
        o = f.overloads[0]

        self.assertEqual(len(o.args), 5)
        self.assertEqual([a.name for a in o.args], ['x', 'y', 'z', 'args', 'kwargs'])
        self.assertEqual([a.typeFilter for a in o.args], [Int64, None, NoneType, TupleOf(float), ConstDict(str, float)])
        self.assertEqual([a.defaultValue for a in o.args], [None, (30,), (None,), None, None])
        self.assertEqual([a.isStarArg for a in o.args], [False, False, False, True, False])
        self.assertEqual([a.isKwarg for a in o.args], [False, False, False, False, True])
