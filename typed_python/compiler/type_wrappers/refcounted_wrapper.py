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

from typed_python.compiler.type_wrappers.wrapper import Wrapper
from typed_python.compiler.typed_expression import TypedExpression
import typed_python.compiler.native_ast as native_ast


class RefcountedWrapper(Wrapper):
    """Common functions for types which are pointers to a refcount and data."""

    # override if we can never be null
    CAN_BE_NULL = True

    def on_refcount_zero(self, context, expr):
        """Generate code to tear down the object when refcount hits zero."""
        raise NotImplementedError()

    def get_refcount_ptr_expr(self, nonref_expr):
        """Return a pointer to the object's refcount. Subclasses can override.

        Args:
            nonref_expr - a native expression equivalent to 'self.nonref_expr'. In most cases
                this will be the pointer to the actual refcounted data structure.
        """
        return nonref_expr.ElementPtrIntegers(0, 0)

    def getRefcount(self, context, inst):
        return TypedExpression(
            context,
            self.get_refcount_ptr_expr(inst.nonref_expr).load(),
            int,
            False
        )

    def convert_incref(self, context, expr):
        if self.CAN_BE_NULL:
            context.pushEffect(
                native_ast.Expression.Branch(
                    cond=expr.nonref_expr,
                    false=native_ast.nullExpr,
                    true=self.get_refcount_ptr_expr(expr.nonref_expr).atomic_add(1) >> native_ast.nullExpr
                )
            )
        else:
            context.pushEffect(
                self.get_refcount_ptr_expr(expr.nonref_expr).atomic_add(1) >> native_ast.nullExpr
            )

    def convert_assign(self, context, expr, other):
        assert expr.isReference

        self.convert_incref(context, other)
        self.convert_destroy(context, expr)

        context.pushEffect(
            expr.expr.store(other.nonref_expr)
        )

    def convert_copy_initialize(self, context, expr, other):
        expr = expr.expr
        other = other.nonref_expr

        if self.CAN_BE_NULL:
            context.pushEffect(
                native_ast.Expression.Branch(
                    cond=other,
                    false=expr.store(other),
                    true=(
                        expr.store(other) >>
                        self.get_refcount_ptr_expr(expr.load()).atomic_add(1) >>
                        native_ast.nullExpr
                    )
                )
            )
        else:
            context.pushEffect(
                expr.store(other) >>
                self.get_refcount_ptr_expr(expr.load()).atomic_add(1) >>
                native_ast.nullExpr
            )

    def convert_destroy(self, context, target):
        res = context.expressionAsFunctionCall(
            "decref_" + str(self),
            (target,),
            lambda instance: self.convert_destroy_inner(instance.context, instance),
            ("decref", self),
            outputType=context.constant(None).expr_type
        )

        if res is not None:
            context.pushEffect(res.expr)

    def convert_destroy_inner(self, context, target):
        assert isinstance(target, TypedExpression)

        assert target.isReference
        targetExpr = target.nonref_expr

        if self.CAN_BE_NULL:
            with context.ifelse(targetExpr) as (true, false):
                with true:
                    with context.ifelse(self.get_refcount_ptr_expr(targetExpr).atomic_add(-1).eq(1)) as (subtrue, subfalse):
                        with subtrue:
                            context.pushEffect(self.on_refcount_zero(context, target))
        else:
            with context.ifelse(self.get_refcount_ptr_expr(targetExpr).atomic_add(-1).eq(1)) as (subtrue, subfalse):
                with subtrue:
                    context.pushEffect(self.on_refcount_zero(context, target))
