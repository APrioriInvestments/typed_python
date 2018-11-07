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
from nativepython.python_to_native_ast import Converter
import unittest


class TestPythonToNativeAst(unittest.TestCase):
    def test_convert_self_add_function(self):
        converter = Converter()

        def f(x):
            return x+x

        callTarget = converter.convert(f, (int,))

        self.assertEqual(callTarget.input_types, [Int64()])
        self.assertEqual(callTarget.output_type, Int64())

        targets = converter.extract_new_function_definitions()
        assert callTarget.name in targets

        print(targets[callTarget.name])