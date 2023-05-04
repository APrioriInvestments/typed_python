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

from typed_python import OneOf

import llvmlite.binding as llvm
import typed_python.compiler.native_compiler.native_ast as native_ast
import typed_python.compiler.native_compiler.native_ast_to_llvm as native_ast_to_llvm
from typed_python.compiler.native_compiler.compiler_cache import CompilerCache
from typed_python.compiler.native_compiler.llvm_execution_engine import create_execution_engine
from typed_python.compiler.native_compiler.loaded_module import LoadedModule
from typed_python.compiler.native_compiler.native_function_pointer import NativeFunctionPointer
from typed_python.compiler.native_compiler.binary_shared_object import BinarySharedObject
from typed_python.compiler.native_compiler.typed_call_target import TypedCallTarget


class NativeCompiler:
    """"Engine for compiling bundles of native_ast.Function objects into NativeFunctionPointers.

    This class is responsible for

        * telling clients what named functions have been defined and what their types are.
        * compiling functions into a runnable form using llvm
        * performing any runtime-based performance optimizations
        * maintaining the compiler cache

    Note that this class is NOT threadsafe
    """
    def __init__(self, inlineThreshold):
        self.compilerCache = None
        self.hasEverHadFunctionsAdded = False

        self.engine, self.module_pass_manager = create_execution_engine(inlineThreshold)
        self.converter = native_ast_to_llvm.NativeAstToLlvmConverter()
        self.inlineThreshold = inlineThreshold
        self.verbose = False
        self.optimize = True

        # map from linkName: str -> NativeFunctionPointer
        # this is only populated if we don't have a compiler cache
        self.linkNameToFunctionPtr = {}

        # map from linkName: str -> TypedCallTarget.
        # this contains every typed call target that we have loaded
        # but not the ones that are available in the compiler cache but that
        # have not yet been loaded
        self._allTypedCallTargets = {}
        self._allFunctionsDefined = set()

    def isFunctionDefined(self, linkName: str) -> bool:
        """Is this function known to the compiler?"""
        if self.compilerCache is None:
            return linkName in self.linkNameToFunctionPtr
        else:
            if linkName in self._allFunctionsDefined:
                return True

            if not self.compilerCache.hasSymbol(linkName):
                return False

            # the compiler cache has the symbol but we can't currently be sure
            # that we can load it - so load it. If it fails, it won't tell us
            # that it can load that symbol a second time.
            self._loadFromCache(linkName)

            return linkName in self._allFunctionsDefined

    def typedCallTargetFor(self, linkName: str) -> OneOf(None, TypedCallTarget):
        """If this function is known, its TypedCallTarget, or None.

        If this function was added without a TypedCallTarget (say, because its
        a destructor or some other untyped function) then this will be None.
        """
        if self.compilerCache is None:
            # if we have no compiler cache, then _allTypedCallTargets will
            # contain everything we've ever seen
            return self._allTypedCallTargets.get(linkName)
        else:
            if linkName in self._allTypedCallTargets:
                return self._allTypedCallTargets[linkName]

            if linkName not in self._allFunctionsDefined:
                if self.compilerCache.hasSymbol(linkName):
                    self._loadFromCache(linkName)

            return self._allTypedCallTargets.get(linkName)

    def initializeCompilerCache(self, compilerCacheDir):
        """Indicate that we should use a compiler cache from disk at 'compilerCacheDir'."""
        if self.hasEverHadFunctionsAdded:
            raise Exception("Can't set the compiler cache if we've added functions.")

        self.compilerCache = CompilerCache(compilerCacheDir)

    def mark_converter_verbose(self):
        self.converter.verbose = True

    def mark_llvm_codegen_verbose(self):
        self.verbose = True

    def addFunctions(
        self,
        # map from str to native_ast.Function
        functionDefinitions,
        # map from str to the TypedCallTarget for any function that's actually typed
        typedCallTargets
    ):
        """Add a collection of functions to the compiler.

        Once a function has been added, we can request a NativeFunctionPointer for it.
        """
        self.hasEverHadFunctionsAdded = True

        if self.compilerCache is None:
            self._allTypedCallTargets.update(typedCallTargets)
            self._allFunctionsDefined.update(functionDefinitions)

            loadedModule = self._buildModule(functionDefinitions)
            loadedModule.linkGlobalVariables()
        else:
            binary = self._buildSharedObject(functionDefinitions)

            self.compilerCache.addModule(
                binary,
                typedCallTargets,
            )

    def functionPointerByName(self, linkName) -> NativeFunctionPointer:
        """Find a NativeFunctionPointer for a given link-time name.

        Args:
            linkName (str) - the name of the compiled symbol we want.

        Returns:
            a NativeFunctionPointer or None if the function has never been defined.
        """
        if self.compilerCache is not None:
            # the compiler cache has every shared object and can load them
            if linkName in self.linkNameToFunctionPtr:
                return self.linkNameToFunctionPtr[linkName]

            if not self.compilerCache.hasSymbol(linkName):
                return None

            self.compilerCache.loadForSymbol(linkName)

            funcPtr = self.compilerCache.function_pointer_by_name(linkName)

            assert funcPtr is not None

            self.linkNameToFunctionPtr[linkName] = funcPtr

            return funcPtr

        # the llvm compiler is just building shared objects, but the
        # compiler cache has all the pointers.
        return self.linkNameToFunctionPtr.get(linkName)

    def _loadFromCache(self, linkName):
        """Attempt to load a cached copy of 'linkName' and all reachable code.

        The compilerCache must exist and agree that this function exists.
        """
        assert self.compilerCache and self.compilerCache.hasSymbol(linkName)

        callTargetsAndTypes = self.compilerCache.loadForSymbol(linkName)

        if callTargetsAndTypes is not None:
            newTypedCallTargets, newNativeFunctionTypes = callTargetsAndTypes
            self.converter.markExternal(newNativeFunctionTypes)
            self._allTypedCallTargets.update(newTypedCallTargets)
            self._allFunctionsDefined.update(newNativeFunctionTypes)

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
            module.usedExternalFunctions
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
            self.linkNameToFunctionPtr[fname] = native_function_pointers[fname]

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
