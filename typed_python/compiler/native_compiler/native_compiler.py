#   Copyright 2023 typed_python Authors
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

import llvmlite.binding as llvm
import typed_python.compiler.native_compiler.native_ast as native_ast
import typed_python.compiler.native_compiler.native_ast_to_llvm as native_ast_to_llvm
from typed_python.compiler.native_compiler.compiler_cache import CompilerCache
from typed_python.compiler.native_compiler.llvm_execution_engine import create_execution_engine
from typed_python.compiler.native_compiler.loaded_module import LoadedModule
from typed_python.compiler.native_compiler.native_function_pointer import NativeFunctionPointer
from typed_python.compiler.native_compiler.binary_shared_object import BinarySharedObject


class NativeCompiler:
    """"Engine for compiling bundles of native_ast.Function objects into NativeFunctionPointers.

    This class is responsible for
        * telling clients what named functions have been defined and what their types are.
        * compiling functions into a runnable form using llvm
        * performing any runtime-based performance optimizations
        * maintaining the compiler cache

    Note that this class is NOT threadsafe and clients are expected to serialize their
    access through Runtime.
    """
    def __init__(self, inlineThreshold):
        self.compilerCache = None
        self.engine, self.module_pass_manager = create_execution_engine(inlineThreshold)
        self.converter = native_ast_to_llvm.Converter()
        self.functions_by_name = {}
        self.inlineThreshold = inlineThreshold
        self.verbose = False
        self.optimize = True

    def initializeCompilerCache(self, compilerCacheDir):
        """Indicate that we should use a compiler cache from disk at 'compilerCacheDir'."""
        self.compilerCache = CompilerCache(compilerCacheDir)

    def markExternal(self, functionNameToType):
        """Provide type signatures for a set of external functions."""
        self.converter.markExternal(functionNameToType)

    def mark_converter_verbose(self):
        self.converter.verbose = True

    def mark_llvm_codegen_verbose(self):
        self.verbose = True

    def addFunctions(
        self,
        # map from str to native_ast.Function
        functionDefinitions,
        # map from str to the TypedCallTarget for any function that's actually typed
        typedCallTargets,
        externallyUsed
    ):
        """Add a collection of functions to the compiler.

        Once a function has been added, we can request a NativeFunctionPointer for it.
        """
        if self.compilerCache is None:
            loadedModule = self._buildModule(functionDefinitions)
            loadedModule.linkGlobalVariables()
        else:
            binary = self._buildSharedObject(functionDefinitions)

            self.compilerCache.addModule(
                binary,
                typedCallTargets,
                externallyUsed
            )

    def functionPointerByName(self, linkerName) -> NativeFunctionPointer:
        """Find a NativeFunctionPointer for a given link-time name.

        Args:
            linkerName (str) - the name of the compiled symbol we want.

        Returns:
            a NativeFunctionPointer or None if the function has never been defined.
        """
        if self.compilerCache is not None:
            # the compiler cache has every shared object and can load them
            return self.compilerCache.function_pointer_by_name(linkerName)

        # the llvm compiler is just building shared objects, but the
        # compiler cache has all the pointers.
        return self.functions_by_name.get(linkerName)

    def loadFromCache(self, linkName):
        """Attempt to load a cached copy of 'linkName' and all reachable code.

        If it isn't defined, or has already been defined, return None. If we're loading it
        for the first time, return a pair

            (typedCallTargets, nativeTypes)

        where typedCallTargets is a map from linkName to TypedCallTarget, and nativeTypes is
        a map from linkName to native_ast.Type.Function giving the native implementation type.

        WARNING: this will return None if you already called 'functionPointerByName' on it
        """
        if self.compilerCache:
            if self.compilerCache.hasSymbol(linkName):
                return self.compilerCache.loadForSymbol(linkName)

    def _buildSharedObject(self, functions):
        """Add native definitions and return a BinarySharedObject representing the compiled code."""
        module = self.converter.add_functions(functions)

        try:
            mod = llvm.parse_assembly(module.moduleText)
            mod.verify()
        except Exception:
            print("failing: ", module)
            raise

        # Now add the module and make sure it is ready for execution
        self.engine.add_module(mod)

        if self.optimize:
            self.module_pass_manager.run(mod)

        self.engine.finalize_object()

        return BinarySharedObject.fromModule(
            mod,
            module.globalVariableDefinitions,
            module.functionNameToType,
        )

    def _buildModule(self, functions):
        """Compile a list of functions into a new module.

        Args:
            functions - a map from name to native_ast.Function

        Returns:
            None, or a LoadedModule object.
        """
        if not functions:
            return None

        # module is a ModuleDefinition object
        module = self.converter.add_functions(functions)

        try:
            mod = llvm.parse_assembly(module.moduleText)
            mod.verify()
        except Exception:
            print("failing:\n\n", module.moduleText)
            raise

        # Now add the module and make sure it is ready for execution
        self.engine.add_module(mod)

        if self.optimize:
            self.module_pass_manager.run(mod)

        if self.verbose:
            print(mod)

        self.engine.finalize_object()

        # Look up the function pointer (a Python int)
        native_function_pointers = {}

        for fname in functions:
            func_ptr = self.engine.get_function_address(fname)
            input_types = [x[1] for x in functions[fname].args]
            output_type = functions[fname].output_type

            native_function_pointers[fname] = NativeFunctionPointer(
                fname, func_ptr, input_types, output_type
            )
            self.functions_by_name[fname] = native_function_pointers[fname]

        native_function_pointers[module.GET_GLOBAL_VARIABLES_NAME] = (
            NativeFunctionPointer(
                module.GET_GLOBAL_VARIABLES_NAME,
                self.engine.get_function_address(
                    module.GET_GLOBAL_VARIABLES_NAME
                ),
                [native_ast.Void.pointer().pointer()],
                native_ast.Void
            )
        )

        return LoadedModule(native_function_pointers, module.globalVariableDefinitions)
