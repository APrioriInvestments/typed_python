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

from typed_python import Alternative, OneOf, Bool, Float64, Int64

import nativepython.native_ast as native_ast
import nativepython

from nativepython.type_wrappers.wrapper import Wrapper
from nativepython.type_wrappers.none_wrapper import NoneWrapper

typeWrapper = lambda t: nativepython.python_object_representation.typedPythonTypeToTypeWrapper(t)

class TypedExpression(object):
    def __init__(self, context, expr, t, isReference):
        """Initialize a TypedExpression

        expr - a native_ast containing an expression
        t - a subclass of Wrapper, or a type that we'll convert to a wrapper, or None (meaning
            control doesn't return)
        isReference - is this a pointer to a memory location holding the value, or the actual value?
            if it's a reference, the reference is guaranteed to be valid for the lifetime of the
            expression. if its the value, then the value contains an implicit incref which must
            either be transferred or decreffed.
        """
        super().__init__()
        if isinstance(t, type) or hasattr(t, "__typed_python_category__"):
            t = typeWrapper(t)

        assert isinstance(t, Wrapper) or t is None, t
        assert isinstance(expr, native_ast.Expression), expr

        self.context = context
        self.expr = expr
        self.expr_type = t
        self.isReference = isReference

    def as_call_arg(self):
        """Convert this expression to a call-argument form."""
        if self.expr_type.is_pod:
            return self.ensureNonReference()
        else:
            if self.isReference:
                return self

            assert False, "we should be jamming this rvalue object into a temporary that we can pass"

    @property
    def nonref_expr(self):
        """Get our expression (deferenced if necessary) so that it definitely represents the real object, not its location"""
        return self.expr_type.ensureNonReference(self).expr

    def ensureNonReference(self):
        return self.expr_type.ensureNonReference(self)

    def convert_incref(self):
        return self.expr_type.convert_incref(self.context, self)

    def convert_set_attribute(self, attribute, expr):
        return self.expr_type.convert_set_attribute(self.context, self, attribute, expr)

    def convert_assign(self, toStore):
        return self.expr_type.convert_assign(self.context, self, toStore)

    def convert_destroy(self):
        return self.expr_type.convert_destroy(self.context, self)

    def convert_copy_initialize(self, toStore):
        return self.expr_type.convert_copy_initialize(self.context, self, toStore)

    def convert_attribute(self, attribute):
        return self.context.wrapInTemporaries(
            lambda self: self.expr_type.convert_attribute(self.context, self, attribute),
            (self,)
            )

    def convert_getitem(self, item):
        return self.context.wrapInTemporaries(
            lambda self, item:
                self.expr_type.convert_getitem(self.context, self, item),
            (self, item)
            )

    def convert_len(self):
        return self.context.wrapInTemporaries(
            lambda self:
                self.expr_type.convert_len(self.context, self),
            (self,)
            )

    def convert_unary_op(self, op):
        return self.context.wrapInTemporaries(
            lambda self: 
                self.expr_type.convert_unary_op(self.context, self, op),
            (self,)
            )

    def convert_bin_op(self, op, rhs):
        return self.context.wrapInTemporaries(
            lambda l,r: 
                self.expr_type.convert_bin_op(self.context, l, op, r),
            (self, rhs)
            )

    def convert_call(self, args):
        return self.context.wrapInTemporaries(
            lambda self, *args:
                self.expr_type.convert_call(self.context, self, args),
            (self,) + tuple(args)
            )

    def convert_to_type(self, target_type):
        return self.context.wrapInTemporaries(
            lambda self: self.expr_type.convert_to_type(self.context, self, target_type),
            (self,)
            )

    def toFloat64(self):
        return self.context.wrapInTemporaries(
            lambda self: self.expr_type.convert_to_type(self.context, self, typeWrapper(Float64())),
            (self,)
            )

    def toInt64(self):
        return self.context.wrapInTemporaries(
            lambda self: self.expr_type.convert_to_type(self.context, self, typeWrapper(Int64())),
            (self,)
            )

    def toBool(self):
        return self.context.wrapInTemporaries(
            lambda s: self.expr_type.convert_to_type(self.context, s, typeWrapper(Bool())),
            (self,)
            )

    def __str__(self):
        return "TypedExpression(%s%s)" % (self.expr_type, ",[ref]" if self.isReference else "")

    def __rshift__(self, other):
        return TypedExpression(self.context, self.expr >> other.expr, other.expr_type, other.isReference)

    def __or__(self, other):
        return self.expr_type.sugar_operator(self, other, "BitOr")

    def __and__(self, other):
        return self.expr_type.sugar_operator(self, other, "BitAnd")

    def __lt__(self, other):
        return self.expr_type.sugar_comparison(self, other, "Lt")

    def __le__(self, other):
        return self.expr_type.sugar_comparison(self, other, "LtE")

    def __gt__(self, other):
        return self.expr_type.sugar_comparison(self, other, "Gt")

    def __ge__(self, other):
        return self.expr_type.sugar_comparison(self, other, "GtE")

    def __eq__(self, other):
        return self.expr_type.sugar_comparison(self, other, "Eq")

    def __ne__(self, other):
        return self.expr_type.sugar_comparison(self, other, "NotEq")
