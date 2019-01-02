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

from nativepython.type_wrappers.wrapper import Wrapper
from nativepython.type_wrappers.exceptions import generateThrowException
import nativepython.type_wrappers.runtime_functions as runtime_functions

from typed_python import NoneType, Int64

import nativepython.native_ast as native_ast
import nativepython

class RefcountedWrapper(Wrapper):
    """Common functions for types which are pointers to a refcount and data."""

    def on_refcount_zero(self, context, expr):
        """Generate code to tear down the object when refcount hits zero."""
        raise NotImplementedError()

    def convert_incref(self, context, expr):
        expr = expr.ensureNonReference()

        return context.RefExpr(
            native_ast.Expression.Branch(
                cond=expr.expr,
                false=native_ast.nullExpr,
                true=expr.expr.ElementPtrIntegers(0,0).store(
                        native_ast.Expression.Binop(
                            l=expr.expr.ElementPtrIntegers(0,0).load(),
                            op=native_ast.BinaryOp.Add(),
                            r=native_ast.const_int_expr(1)
                        )
                    )
                ) >> expr.expr,
            self
            )

    def convert_assign(self, context, expr, other):
        assert expr.isReference
        other = other.ensureNonReference()

        return context.NoneExpr(
            native_ast.Expression.Let(
                expr.expr.load(),
                lambda oldSelf:
                    self.convert_copy_initialize(context, expr, other).expr
                        >> self.convert_destroy(context, context.ValueExpr(self, oldSelf))
                )
            )

    def convert_copy_initialize(self, context, expr, other):
        other = other.ensureNonReference()
        assert expr.isReference

        return context.NoneExpr(
            native_ast.Expression.Branch(
                cond=other.expr,
                false=expr.expr.store(other.expr),
                true=
                    expr.expr.store(other.expr) >>
                    expr.expr.load().ElementPtrIntegers(0,0).store(
                        native_ast.Expression.Binop(
                            l=expr.expr.load().ElementPtrIntegers(0,0).load(),
                            op=native_ast.BinaryOp.Add(),
                            r=native_ast.const_int_expr(1)
                        )
                    )
                )
            )

    def convert_destroy(self, context, target):
        target = target.ensureNonReference()

        return context.NoneExpr(
            native_ast.Expression.Branch(
                cond=target.expr,
                false=native_ast.nullExpr,
                true=
                    target.expr.ElementPtrIntegers(0,0).store(
                        native_ast.Expression.Binop(
                            l=target.expr.ElementPtrIntegers(0,0).load(),
                            op=native_ast.BinaryOp.Sub(),
                            r=native_ast.const_int_expr(1)
                        )
                    ) >>
                    native_ast.Expression.Branch(
                        cond=target.expr.ElementPtrIntegers(0,0).load(),
                        true=native_ast.nullExpr,
                        false=self.on_refcount_zero(context, target)
                        )
                )
            )
