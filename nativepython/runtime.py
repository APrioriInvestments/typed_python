#   Copyright 2017 Braxton Mckee
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

import nativepython.python_to_native_ast as python_to_native_ast
import nativepython.native_ast as native_ast
import nativepython.llvm_compiler as llvm_compiler
import nativepython

_singleton = [None]

class Runtime:
    @staticmethod
    def singleton():
        if _singleton[0] is None:
            _singleton[0] = Runtime()
        return _singleton[0]

    def __init__(self):
        self.compiler = llvm_compiler.Compiler()
        self.converter = python_to_native_ast.Converter()

    def compile(self, f, input_types):
        callTarget = self.converter.convert(f, input_types)

        targets = self.converter.extract_new_function_definitions()
        
        function_pointers = self.compiler.add_functions(targets)

        return function_pointers[callTarget.name]