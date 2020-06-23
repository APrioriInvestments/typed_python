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

from typed_python.compiler.native_ast import (
    Expression, Void, Int32, Int64, nullExpr, Function, FunctionBody,
    Teardown, const_int32_expr, CallTarget, NamedCallTarget
)
from typed_python.compiler.runtime import Runtime
from typed_python.compiler.llvm_compiler import llvm
import typed_python.compiler.native_ast_to_llvm as native_ast_to_llvm
import unittest
from typed_python import PointerTo, ListOf
import ctypes


def externalCallTarget(name, output, *inputs):
    return CallTarget.Named(
        target=NamedCallTarget(
            name=name,
            arg_types=inputs,
            output_type=output,
            external=True,
            varargs=False,
            intrinsic=False,
            can_throw=True
        )
    )


class TestNativeAstToLlvm(unittest.TestCase):
    def test_teardowns(self):
        converter = native_ast_to_llvm.Converter()

        ct = externalCallTarget("thrower", Void)

        f = Function(
            args=[('a', Int32)],
            output_type=Void,
            body=FunctionBody.Internal(
                Expression.Finally(
                    expr=(
                        ct.call() >>
                        Expression.ActivatesTeardown('a1') >>
                        ct.call() >>
                        Expression.ActivatesTeardown('a2') >>
                        nullExpr
                    ),
                    teardowns=[
                        Teardown.ByTag(tag='a1', expr=Expression.Branch(cond=const_int32_expr(10), true=ct.call(), false=nullExpr)),
                        Teardown.ByTag(tag='a1', expr=nullExpr)
                    ]
                )
            )
        )

        moduleDef = converter.add_functions({'f': f})
        mod = llvm.parse_assembly(moduleDef.moduleText)
        mod.verify()

    def test_global_pointers(self):
        f = Function(
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

        # we have to use the singleton here because the llvm context
        # is global.
        llvmCompiler = Runtime.singleton().llvm_compiler

        moduleDef, functionPointers = llvmCompiler.add_functions({'f': f})

        getGlobals = functionPointers[moduleDef.globalDefName]
        fPtr = functionPointers['f']

        pointers = ListOf(PointerTo(int))()
        pointers.resize(1)

        # it should be initialized to zero
        assert ctypes.CFUNCTYPE(ctypes.c_long)(fPtr.fp)() == 0

        # get out the pointer table
        ctypes.CFUNCTYPE(None, ctypes.c_void_p)(getGlobals.fp)(
            ctypes.c_void_p(int(pointers.pointerUnsafe(0)))
        )

        pointers[0].set(123)

        # it should be initialized to 123 now
        assert ctypes.CFUNCTYPE(ctypes.c_long)(fPtr.fp)() == 123
