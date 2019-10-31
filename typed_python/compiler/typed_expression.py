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

from typed_python import Bool, Float64, Int64, UInt64
from typed_python.python_ast import BinaryOp, ComparisonOp, BooleanOp
import typed_python.compiler.native_ast as native_ast
import typed_python.compiler

from typed_python.compiler.type_wrappers.wrapper import Wrapper

typeWrapper = lambda t: typed_python.compiler.python_object_representation.typedPythonTypeToTypeWrapper(t)


class TypedExpression(object):
    def __init__(self, context, expr, t, isReference):
        """Initialize a TypedExpression

        context - an ExpressionConversionContext
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

    def changeContext(self, newContext):
        return TypedExpression(newContext, self.expr, self.expr_type, self.isReference)

    def changeType(self, newType, isReferenceOverride=None):
        """Return a TypedExpression with the same native_ast content but a different type-wrapper."""
        return TypedExpression(
            self.context,
            self.expr,
            typeWrapper(newType),
            self.isReference if isReferenceOverride is None else isReferenceOverride
        )

    def as_native_call_arg(self):
        """Convert this expression to a call-argument form."""
        if self.expr_type.is_pass_by_ref:
            assert self.isReference
            return self.expr
        else:
            return self.nonref_expr

    def canUnwrap(self):
        return self.expr_type.can_unwrap

    def unwrap(self, generator):
        """If we 'canUnwrap', call generator back with 'self' in the lowered form.

        In the case of a OneOf, this may produce a compound expression that merges the operation
        over the possible subtypes.
        """
        return self.expr_type.unwrap(self.context, self, generator)

    @property
    def nonref_expr(self):
        """Get our expression (dereferenced if necessary) so that it definitely represents the real object, not its location"""
        if self.expr_type is None:
            return self.expr

        if self.isReference:
            if self.expr_type.is_empty:
                return self.expr >> native_ast.nullExpr
            return self.expr.load()
        else:
            return self.expr

    def ensureNonReference(self):
        return self.expr_type.ensureNonReference(self)

    def convert_incref(self):
        return self.expr_type.convert_incref(self.context, self)

    def convert_set_attribute(self, attribute, expr):
        return self.expr_type.convert_set_attribute(self.context, self, attribute, expr)

    def convert_assign(self, toStore):
        return self.expr_type.convert_assign(self.context, self, toStore)

    def convert_initialize_from_args(self, *args):
        return self.expr_type.convert_initialize_from_args(self.context, self, *args)

    def convert_default_initialize(self):
        return self.expr_type.convert_default_initialize(self.context, self)

    def convert_destroy(self):
        return self.expr_type.convert_destroy(self.context, self)

    def convert_copy_initialize(self, toStore):
        return self.expr_type.convert_copy_initialize(self.context, self, toStore)

    def convert_attribute(self, attribute):
        return self.expr_type.convert_attribute(self.context, self, attribute)

    def convert_setitem(self, index, value):
        return self.expr_type.convert_setitem(self.context, self, index, value)

    def convert_format(self, format_spec):
        return self.expr_type.convert_format(self.context, self, format_spec)

    def convert_delitem(self, item):
        return self.expr_type.convert_delitem(self.context, self, item)

    def convert_getitem(self, item):
        return self.expr_type.convert_getitem(self.context, self, item)

    def convert_getitem_unsafe(self, item):
        return self.expr_type.convert_getitem_unsafe(self.context, self, item)

    def convert_len(self):
        return self.expr_type.convert_len(self.context, self)

    def convert_hash(self):
        return self.expr_type.convert_hash(self.context, self)

    def convert_abs(self):
        return self.expr_type.convert_abs(self.context, self)

    def convert_bool_cast(self):
        return self.expr_type.convert_bool_cast(self.context, self)

    def convert_int_cast(self):
        return self.expr_type.convert_int_cast(self.context, self)

    def convert_float_cast(self):
        return self.expr_type.convert_float_cast(self.context, self)

    def convert_str_cast(self):
        return self.expr_type.convert_str_cast(self.context, self)

    def convert_bytes_cast(self):
        return self.expr_type.convert_bytes_cast(self.context, self)

    def convert_builtin(self, f, a1=None):
        return self.expr_type.convert_builtin(f, self.context, self, a1)

    def convert_repr(self):
        return self.expr_type.convert_repr(self.context, self)

    def convert_reserved(self):
        return self.expr_type.convert_reserved(self.context, self)

    def convert_unary_op(self, op):
        return self.expr_type.convert_unary_op(self.context, self, op)

    def convert_bin_op(self, op, rhs, inplace=False):
        return self.expr_type.convert_bin_op(self.context, self, op, rhs, inplace)

    def convert_bin_op_reverse(self, op, rhs, inplace=False):
        return self.expr_type.convert_bin_op_reverse(self.context, self, op, rhs, inplace)

    def convert_call(self, args, kwargs):
        return self.expr_type.convert_call(self.context, self, args, kwargs)

    def convert_method_call(self, methodname, args, kwargs):
        return self.expr_type.convert_method_call(self.context, self, methodname, args, kwargs)

    def convert_to_type(self, target_type, explicit=True):
        """Convert to a target type as a function argument.

        If 'explicit', then allow conversions that may change type (e.g. int->float). Otherwise
        insist on strict conversion.
        """
        target_type = typeWrapper(target_type)

        return self.expr_type.convert_to_type(self.context, self, target_type, explicit=explicit)

    def convert_context_manager_enter(self):
        return self.expr_type.convert_context_manager_enter(self.context, self)

    def convert_context_manager_exit(self, args):
        return self.expr_type.convert_context_manager_exit(self.context, self, args)

    def convert_to_type_with_target(self, targetVal, explicit=True):
        return self.expr_type.convert_to_type_with_target(self.context, self, targetVal, explicit=explicit)

    def get_iteration_expressions(self):
        return self.expr_type.get_iteration_expressions(self.context, self)

    def convert_next(self):
        return self.expr_type.convert_next(self.context, self)

    def toFloat64(self):
        return self.expr_type.convert_to_type(self.context, self, typeWrapper(Float64))

    def toInt64(self):
        return self.expr_type.convert_to_type(self.context, self, typeWrapper(Int64))

    def toUInt64(self):
        return self.expr_type.convert_to_type(self.context, self, typeWrapper(UInt64))

    def toBool(self):
        return self.expr_type.convert_to_type(self.context, self, typeWrapper(Bool))

    @staticmethod
    def asBool(typedExpressionOrNone):
        if typedExpressionOrNone is not None:
            return typedExpressionOrNone.toBool()
        else:
            return None

    def __str__(self):
        return "TypedExpression(%s%s)" % (self.expr_type, ",[ref]" if self.isReference else "")

    def __repr__(self):
        return "TypedExpression(%s%s)" % (self.expr_type, ",[ref]" if self.isReference else "")

    def __rshift__(self, other):
        return TypedExpression(self.context, self.expr >> other.expr, other.expr_type, other.isReference)

    @staticmethod
    def sugar_operator(left, right, opname):
        if isinstance(right, (int, float, bool)):
            right = left.context.constant(right)

        if hasattr(BinaryOp, opname):
            op = getattr(BinaryOp, opname)()
        elif hasattr(ComparisonOp, opname):
            op = getattr(ComparisonOp, opname)()
        elif hasattr(BooleanOp, opname):
            op = getattr(BooleanOp, opname)()
        else:
            assert False, opname

        return left.convert_bin_op(op, right)

    def __add__(self, other):
        return TypedExpression.sugar_operator(self, other, "Add")

    def __sub__(self, other):
        return TypedExpression.sugar_operator(self, other, "Sub")

    def __mul__(self, other):
        return TypedExpression.sugar_operator(self, other, "Mult")

    def __truediv__(self, other):
        return TypedExpression.sugar_operator(self, other, "Div")

    def __and__(self, other):
        return TypedExpression.sugar_operator(self, other, "BitAnd")

    def __or__(self, other):
        return TypedExpression.sugar_operator(self, other, "BitOr")

    def __xor__(self, other):
        return TypedExpression.sugar_operator(self, other, "BitXor")

    def __lt__(self, other):
        return TypedExpression.sugar_operator(self, other, "Lt")

    def __le__(self, other):
        return TypedExpression.sugar_operator(self, other, "LtE")

    def __gt__(self, other):
        return TypedExpression.sugar_operator(self, other, "Gt")

    def __ge__(self, other):
        return TypedExpression.sugar_operator(self, other, "GtE")

    def __eq__(self, other):
        return TypedExpression.sugar_operator(self, other, "Eq")

    def __ne__(self, other):
        return TypedExpression.sugar_operator(self, other, "NotEq")
