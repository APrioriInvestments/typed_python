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
from typed_python import TypedFunction
from typed_python.internals import FunctionOverload

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

    def verboselyDisplayNativeCode(self):
        self.compiler.mark_converter_verbose()

    def compile(self, f):
        """Compile a single FunctionOverload and install the pointer"""
        if isinstance(f, FunctionOverload):
            for a in f.args:
                assert not a.isStarArg, 'dont support star args yet'
                assert not a.isKwarg, 'dont support keyword yet'

            output_wrapper = python_to_native_ast.typedPythonTypeToTypeWrapper(f.returnType or object)
            input_wrappers = [python_to_native_ast.typedPythonTypeToTypeWrapper(a.typeFilter or object) for a in f.args]

            callTarget = self.converter.convert(f.functionObj, input_wrappers)

            wrappingCallTargetName = self.converter.generateCallConverter(callTarget, output_wrapper)

            targets = self.converter.extract_new_function_definitions()

            function_pointers = self.compiler.add_functions(targets)

            fp = function_pointers[wrappingCallTargetName]

            f._installNativePointer(fp.fp)

            return f

        if hasattr(f, '__typed_python_category__') and f.__typed_python_category__ == 'Function':
            for o in f.overloads:
                self.compile(o)
            return f

        if callable(f):
            result = TypedFunction(f)
            self.compile(result)
            return result

        assert False, f