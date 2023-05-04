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
import llvmlite.ir
import typed_python.compiler.native_compiler.native_ast_to_llvm_function_converter as \
    native_ast_to_llvm_function_converter


import ctypes
from typed_python import _types

llvm.initialize()
llvm.initialize_native_target()
llvm.initialize_native_asmprinter()  # yes, even this one

target_triple = llvm.get_process_triple()
target = llvm.Target.from_triple(target_triple)
target_machine = target.create_target_machine()
target_machine_shared_object = target.create_target_machine(reloc='pic', codemodel='default')

ctypes.CDLL(_types.__file__, mode=ctypes.RTLD_GLOBAL)


pointer_size = (
    llvmlite.ir.PointerType(llvmlite.ir.DoubleType())
    .get_abi_size(target_machine.target_data)
)

assert pointer_size == native_ast_to_llvm_function_converter.pointer_size


def sizeof_native_type(native_type):
    if native_type.matches.Void:
        return 0

    return (
        native_ast_to_llvm_function_converter.type_to_llvm_type(native_type)
        .get_abi_size(target_machine.target_data)
    )


# there can be only one llvm engine alive at once.
_engineCache = []


def create_execution_engine(inlineThreshold):
    if _engineCache:
        return _engineCache[0]

    pmb = llvm.create_pass_manager_builder()
    pmb.opt_level = 3
    pmb.size_level = 0
    pmb.inlining_threshold = inlineThreshold
    pmb.loop_vectorize = True
    pmb.slp_vectorize = True

    pass_manager = llvm.create_module_pass_manager()
    pmb.populate(pass_manager)

    target_machine.add_analysis_passes(pass_manager)

    # And an execution engine with an empty backing module
    backing_mod = llvm.parse_assembly("")
    engine = llvm.create_mcjit_compiler(backing_mod, target_machine)

    _engineCache.append((engine, pass_manager))

    return engine, pass_manager
