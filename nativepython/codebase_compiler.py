#   Copyright 2019 Nativepython Authors
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

import nativepython.python_to_native_converter as python_to_native_converter
import nativepython.llvm_compiler as llvm_compiler
import os
from typed_python import NoneType
from typed_python.internals import FunctionOverload

typeWrapper = lambda t: python_to_native_converter.typedPythonTypeToTypeWrapper(t)


class CompiledCodebase:
    def __init__(self, codebase, sharedObject, nativeTargets, typedTargets):
        self.codebase = codebase
        self.sharedObject = sharedObject
        self.nativeTargets = nativeTargets
        self.typedTargets = typedTargets

    def install(self):
        compiler = llvm_compiler.Compiler()

        function_pointers = compiler.link_binary_shared_object(
            self.sharedObject,
            self.nativeTargets,
            os.path.join(self.codebase.rootDirectory, "__pycache__", "nativepython")
        )

        for wrappingCallTargetName, (f, callTarget) in self.typedTargets.items():
            fp = function_pointers[wrappingCallTargetName]

            f._installNativePointer(
                fp.fp,
                callTarget.output_type.typeRepresentation if callTarget.output_type is not None else NoneType,
                [i.typeRepresentation for i in callTarget.input_types]
            )


class CodebaseCompiler:
    def __init__(self, codebase):
        self.codebase = codebase
        self.llvm_compiler = llvm_compiler.Compiler()
        self.converter = python_to_native_converter.PythonToNativeConverter()

        self.walkCodebase()
        self.compiledCodebase = self.compileModule()

    @staticmethod
    def compile(codebase):
        """Compile a typed_python.Codebase into a CompiledCodebase."""
        return CodebaseCompiler(codebase).compiledCodebase

    def walkCodebase(self):
        """Walk a typed_python.Codebase and compile all valid entrypoints, producing a CompiledCodebase.

        We find all 'Class' objects, all 'Function' objects, and all 'Instantation' objects,
        and compile them all into a single module.
        """
        functions = []

        for name, object in self.codebase.allModuleLevelValues():
            if hasattr(object, '__typed_python_category__'):
                if object.__typed_python_category__ == "Class":
                    for f in object.MemberFunctions.values():
                        functions.append(f)
                if object.__typed_python_category__ == "Function":
                    functions.append(object)

        self.targets = {}

        for f in functions:
            self._convert(f, None)

    def compileModule(self):
        native_targets = self.converter.extract_new_function_definitions()
        sharedObject = self.llvm_compiler.compile_functions_and_return_shared_object(native_targets)

        return CompiledCodebase(self.codebase, sharedObject, native_targets, self.targets)

    def _convert(self, f, argument_types):
        argument_types = argument_types or {}

        if isinstance(f, FunctionOverload):
            for a in f.args:
                assert not a.isStarArg, 'dont support star args yet'
                assert not a.isKwarg, 'dont support keyword yet'

            def chooseTypeFilter(a):
                return argument_types.pop(a.name, a.typeFilter or object)

            input_wrappers = [typeWrapper(chooseTypeFilter(a)) for a in f.args]

            if len(argument_types):
                raise Exception("No argument exists for type overrides %s" % argument_types)

            callTarget = self.converter.convert(f.functionObj, input_wrappers, f.returnType, assertIsRoot=True)

            assert callTarget is not None

            wrappingCallTargetName = self.converter.generateCallConverter(callTarget)

            self.targets[wrappingCallTargetName] = (f, callTarget)

        if hasattr(f, '__typed_python_category__') and f.__typed_python_category__ == 'Function':
            for o in f.overloads:
                self._convert(o, argument_types)
            return f

        if hasattr(f, '__typed_python_category__') and f.__typed_python_category__ == 'BoundMethod':
            for o in f.Function.overloads:
                arg_types = dict(argument_types)
                arg_types[o.args[0].name] = typeWrapper(f.Class)
                self._convert(o, arg_types)
            return f
