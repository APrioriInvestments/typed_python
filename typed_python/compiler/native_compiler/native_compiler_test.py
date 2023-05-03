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

from typed_python.compiler.native_compiler.native_ast import (
    Expression, Int64, Function, FunctionBody
)
import tempfile
from typed_python import PointerTo, ListOf, Runtime
from typed_python.compiler.native_compiler.module_definition import ModuleDefinition
from typed_python.compiler.native_compiler.global_variable_definition import GlobalVariableMetadata

import pytest
import ctypes


def test_global_variable_pointers():
    readGlobalVarFunc = Function(
        args=[],
        output_type=Int64,
        body=FunctionBody.Internal(
            Expression.Return(
                arg=Expression.GlobalVariable(
                    name="globalvar",
                    type=Int64,
                    metadata='aGlobal'
                ).load()
            )
        )
    )

    returnGlobalVarPtrFunc = Function(
        args=[],
        output_type=Int64.pointer(),
        body=FunctionBody.Internal(
            Expression.Return(
                arg=Expression.GlobalVariable(
                    name="globalvar",
                    type=Int64,
                    metadata='aGlobal'
                )
            )
        )
    )

    # we have to use the singleton here because the llvm context
    # is global.
    nativeCompiler = Runtime.singleton().native_compiler

    loadedModule = nativeCompiler._buildModule(
        dict(
            _readGlobalVarFunc=readGlobalVarFunc,
            _returnGlobalVarPtrFunc=returnGlobalVarPtrFunc
        )
    )

    getGlobals = loadedModule.functionPointers[loadedModule.GET_GLOBAL_VARIABLES_NAME]
    readGlobalVar = loadedModule.functionPointers['_readGlobalVarFunc']
    returnGlobalVarPtr = loadedModule.functionPointers['_returnGlobalVarPtrFunc']

    pointers = ListOf(PointerTo(int))()
    pointers.resize(1)
    pointers[0] = PointerTo(int)() + int(0xdeadbeef / 8)

    # it should be initialized to zero
    assert ctypes.CFUNCTYPE(ctypes.c_long)(readGlobalVar.fp)() == 0

    # get out the pointer table
    ctypes.CFUNCTYPE(None, ctypes.c_void_p)(getGlobals.fp)(
        ctypes.c_void_p(int(pointers.pointerUnsafe(0)))
    )

    # read the pointer directly
    ptrFromCode = ctypes.CFUNCTYPE(ctypes.c_void_p)(returnGlobalVarPtr.fp)()

    # read the pointer directly
    assert ptrFromCode == int(pointers[0])

    assert pointers[0] and pointers[0] != int(0xdeadbeef / 8)
    pointers[0].set(123)

    # it should be initialized to 123 now
    assert ctypes.CFUNCTYPE(ctypes.c_long)(readGlobalVar.fp)() == 123


@pytest.mark.skipif('sys.platform=="darwin"')
def test_create_binary_shared_object():
    f = Function(
        args=[],
        output_type=Int64,
        body=FunctionBody.Internal(
            Expression.Return(
                arg=Expression.GlobalVariable(
                    name="globalvar",
                    type=Int64,
                    metadata=GlobalVariableMetadata.IntegerConstant(value=0)
                ).load()
            )
        )
    )

    nativeCompiler = Runtime.singleton().native_compiler

    bso = nativeCompiler._buildSharedObject(
        {'__test_f_2': f}
    )

    assert len(bso.globalVariableDefinitions) == 1

    with tempfile.TemporaryDirectory() as tf:
        loaded = bso.load(tf)

        pointers = ListOf(PointerTo(int))()
        pointers.resize(1)

        loaded.functionPointers[ModuleDefinition.GET_GLOBAL_VARIABLES_NAME](
            pointers.pointerUnsafe(0)
        )

        assert pointers[0]
        pointers[0].set(5)

        assert loaded.functionPointers['__test_f_2']() == 5
