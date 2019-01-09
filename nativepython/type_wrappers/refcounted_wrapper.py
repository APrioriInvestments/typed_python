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
        return native_ast.Expression.Branch(
                cond=expr.nonref_expr,
                false=native_ast.nullExpr,
                true=expr.nonref_expr.ElementPtrIntegers(0,0).store(
                        native_ast.Expression.Binop(
                            l=expr.nonref_expr.ElementPtrIntegers(0,0).load(),
                            op=native_ast.BinaryOp.Add(),
                            r=native_ast.const_int_expr(1)
                        )
                    )
                )

    def convert_assign(self, context, expr, other):
        assert expr.isReference

        return (self.convert_incref(context, other)
            >> self.convert_destroy(context, expr)
            >> expr.expr.store(other.nonref_expr))

    def convert_copy_initialize(self, context, expr, other):
        expr = expr.expr
        other = other.nonref_expr

        return native_ast.Expression.Branch(
                cond=other,
                false=expr.store(other),
                true=
                    expr.store(other) >>
                    expr.load().ElementPtrIntegers(0,0).store(
                        expr.load().ElementPtrIntegers(0,0).load().add(native_ast.const_int_expr(1))
                        )
                )

    def convert_destroy(self, context, target):
        assert target.isReference
        targetExpr = target.nonref_expr

        with context.ifelse(targetExpr) as (true, false):
            with true:
                context.pushEffect(
                    targetExpr.ElementPtrIntegers(0,0).store(
                        targetExpr.ElementPtrIntegers(0,0).load().sub(native_ast.const_int_expr(1))
                        )
                    )
                with context.ifelse(targetExpr.ElementPtrIntegers(0,0).load()) as (subtrue, subfalse):
                    with subfalse:
                        context.pushEffect(self.on_refcount_zero(context, target))

        return native_ast.nullExpr
