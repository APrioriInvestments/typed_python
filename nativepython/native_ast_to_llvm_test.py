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

from nativepython.native_ast import Expression, Void, Int32, nullExpr, Function, FunctionBody, Teardown, const_int32_expr, CallTarget, NamedCallTarget
from nativepython.llvm_compiler import llvm
import nativepython.native_ast_to_llvm as native_ast_to_llvm
import unittest


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

        text = converter.add_functions({'f': f})
        print(text)
        mod = llvm.parse_assembly(text)
        mod.verify()
