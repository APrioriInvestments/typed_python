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

import nativepython
import nativepython.native_ast as native_ast

from nativepython.exceptions import ConversionException

class TypedExpression(object):
    def __init__(self, expr, expr_type):
        object.__init__(self)

        assert expr._alternative is native_ast.Expression

        if expr_type is not None and expr is not None:
            if expr.matches.StackSlot:
                assert expr_type.lower() == native_ast.Type.Pointer(expr.type), (
                    str(expr), 
                    str(expr_type),
                    expr_type.lower(),
                    native_ast.Type.Pointer(expr.type)
                    )

        if expr_type is not None:
            if not expr_type.is_pod:
                raise ConversionException("Can't make a typed expression to a %s" % expr_type)

        self.expr = expr
        self.expr_type = expr_type

    @property
    def address_of(self):
        if self.expr_type.is_ref:
            return TypedExpression(self.expr, self.expr_type.unwrap_reference().pointer)
        raise ConversionException("Can't take address of something of type %s" % self.expr_type)

    @staticmethod
    def Void(expr):
        assert not isinstance(expr, TypedExpression)
        return TypedExpression(
            expr,
            nativepython.type_model.Void
            )

    def drop_create_reference(self):
        return TypedExpression(self.expr, self.expr_type.value_type.reference)

    def drop_double_references(self):
        if self.expr_type.is_ref and self.expr_type.value_type.is_ref:
            return TypedExpression(self.expr.load(), self.expr_type.value_type)\
                .drop_double_references()
        return self

    def __add__(self, other):
        if self.expr_type is None:
            return self

        return TypedExpression(self.expr + other.expr, other.expr_type)

    def with_comment(self, comment):
        return TypedExpression(self.expr.with_comment(comment), self.expr_type)

    def dereference(self):
        if not self.expr_type.is_ref:
            return self
        return TypedExpression(
            self.expr.load(),
            self.expr_type.value_type
            )

    def reference_from_pointer(self):
        assert self.expr_type.is_pointer

        return TypedExpression(
            self.expr,
            self.expr_type.value_type.reference
            )

    @property
    def actor_type(self):
        return self.expr_type.unwrap_reference()

    @property
    def as_creates_reference(self):
        if self.expr_type.is_ref_to_temp:
            raise ConversionException("Can't create a reference on the stack to a temporary!")

        if not self.expr_type.is_ref:
            raise ConversionException("This expression is not already a reference")
        
        return TypedExpression(
            self.expr,
            self.expr_type.create_reference
            )

    def as_call_arg(self, context):
        if self.expr_type.is_create_ref or self.expr_type.is_ref_to_temp:
            return TypedExpression(
                self.expr,
                self.expr_type.value_type.reference
                )

        #never bother to pass references to pod null-value types.
        if self.expr_type.is_ref and \
                self.expr_type.nonref_type.sizeof == 0 and self.expr_type.nonref_type.is_pod:
            return self.dereference().as_call_arg(context)

        if self.expr_type.is_pod and not self.expr_type.is_ref and self.expr_type.sizeof > 0:
            #pass this as a reference so we don't end up with many copies of the same function
            #with different 'reftypes' on the arguments
            temp_ref = context.allocate_temporary(self.expr_type, type_is_temp_ref=False)
            return TypedExpression(
                native_ast.Expression.Store(
                    ptr=temp_ref.expr,
                    val=self.expr
                    ) + temp_ref.expr,
                temp_ref.expr_type
                )

        return self

    def convert_unary_op(self, op):
        return self.actor_type.convert_unary_op(self, op)

    def convert_bin_op(self, op, r):
        return self.actor_type.convert_bin_op(op, self, r)

    def convert_call(self, context, args):
        return self.actor_type.convert_call(context, self, args)

    def convert_attribute(self, context, attr):
        return self.actor_type.convert_attribute(context, self, attr)

    def convert_initialize_copy(self, context, other):
        return self.actor_type.convert_initialize_copy(context, self, other)

    def convert_assign(self, context, other):
        return self.actor_type.convert_assign(context, self, other)

    def convert_set_attribute(self, context, attr, val):
        return self.actor_type.convert_set_attribute(context, self, attr, val)

    def convert_to_type(self, type):
        return self.actor_type.convert_to_type(self, type)

    def convert_setitem(self, context, index, value):
        return self.actor_type.convert_setitem(context, self, index, value)

    def convert_getitem(self, context, index):
        return self.actor_type.convert_getitem(context, self, index)

    @property
    def load(self):
        assert self.expr_type.is_pointer
        return TypedExpression(self.expr.load(), self.expr_type.value_type)

    def __repr__(self):
        return "TypedExpression(t=%s)" % (self.expr_type)
