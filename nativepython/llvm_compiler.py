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

import llvmlite.binding as llvm
import llvmlite.ir
import nativepython.native_ast_to_llvm as native_ast_to_llvm
import sys
import ctypes
import time
from typed_python import _types

llvm.initialize()
llvm.initialize_native_target()
llvm.initialize_native_asmprinter()  # yes, even this one

target_triple = llvm.get_process_triple()
target = llvm.Target.from_triple(target_triple)
target_machine = target.create_target_machine()
target_machine_shared_object = target.create_target_machine(reloc='pic', codemodel='default')

#we need to load the appropriate libstdc++ so that we can get __cxa_begin_catch and friends
if sys.platform == "darwin":
    ctypes.CDLL("libstdc++.dylib",mode=ctypes.RTLD_GLOBAL)
else:
    ctypes.CDLL("libstdc++.so.6",mode=ctypes.RTLD_GLOBAL)

ctypes.CDLL(_types.__file__,mode=ctypes.RTLD_GLOBAL)


pointer_size = (
    llvmlite.ir.PointerType(llvmlite.ir.DoubleType())
        .get_abi_size(target_machine.target_data)
    )

assert pointer_size == native_ast_to_llvm.pointer_size

def sizeof_native_type(native_type):
    if native_type.matches.Void:
        return 0

    return (
        native_ast_to_llvm.type_to_llvm_type(native_type)
            .get_abi_size(target_machine.target_data)
        )

#there can be only one llvm engine alive at once.
_engineCache = []
def create_execution_engine():
    if _engineCache:
        return _engineCache[0]

    pmb = llvm.create_pass_manager_builder()
    pmb.opt_level = 3

    pass_manager = llvm.create_module_pass_manager()
    pmb.populate(pass_manager)

    target_machine.add_analysis_passes(pass_manager)

    # And an execution engine with an empty backing module
    backing_mod = llvm.parse_assembly("")
    engine = llvm.create_mcjit_compiler(backing_mod, target_machine)

    _engineCache.append((engine, pass_manager))

    return engine, pass_manager

class NativeFunctionPointer:
    def __init__(self, fname, fp, input_types, output_type):
        self.fp = fp
        self.fname = fname
        self.input_types = input_types
        self.output_type = output_type

    def __repr__(self):
        return "NativeFunctionPointer(name=%s,addr=%x,in=%s,out=%s)" \
            % (self.fname, self.fp, [str(x) for x in self.input_types], str(self.output_type))

class Compiler:
    def __init__(self):
        self.engine, self.module_pass_manager = create_execution_engine()
        self.converter = native_ast_to_llvm.Converter()
        self.functions_by_name = {}
        self.verbose = False
        self.optimize = True

    def mark_converter_verbose(self):
        self.converter.verbose = True

    def mark_llvm_codegen_verbose(self):
        self.verbose = True

    def optimize_functions(self, functions):
        """Add native definitions and return the text of a new module."""
        module = self.converter.add_functions(functions)

        try:
            mod = llvm.parse_assembly(module)
            mod.verify()
        except Exception:
            print("failing: ", module)
            raise

        # Now add the module and make sure it is ready for execution
        self.engine.add_module(mod)

        if self.optimize:
            self.module_pass_manager.run(mod)

        return mod.as_bitcode()

    def compile_from_bitcode(self, bitcode, functions):
        """Compile a module from pre-optimized text."""
        mod = llvmlite.binding.parse_bitcode(bitcode)
        self.engine.add_module(mod)
        self.engine.finalize_object()

        # Look up the function pointer (a Python int)
        native_function_pointers = {}

        for fname in functions:
            func_ptr = self.engine.get_function_address(fname)
            input_types = [x[1] for x in functions[fname].args]
            output_type = functions[fname].output_type

            native_function_pointers[fname] = NativeFunctionPointer(fname, func_ptr,
                                                  input_types, output_type)

            self.functions_by_name[fname] = native_function_pointers[fname]

        return native_function_pointers

    def add_functions(self, functions):
        if not functions:
            return {}

        module = self.converter.add_functions(functions)

        try:
            mod = llvm.parse_assembly(module)
            mod.verify()
        except Exception:
            print("failing: ", module)
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

            native_function_pointers[fname] = NativeFunctionPointer(fname, func_ptr,
                                                  input_types, output_type)

            self.functions_by_name[fname] = native_function_pointers[fname]

        return (native_function_pointers, str(mod))

