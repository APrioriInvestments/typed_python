#   Copyright 2017-2020 typed_python Authors
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

import typed_python.compiler
from math import trunc, floor, ceil

typeWrapper = lambda x: typed_python.compiler.python_object_representation.typedPythonTypeToTypeWrapper(x)


class ClassOrAlternativeWrapperMixin:
    """A Mixin class that defines conversions on class and alternatives in terms of method calls."""

    def convert_call(self, context, expr, args, kwargs):
        """Generates code for calling expr (Class or Alternative).

        Returns:
            TypedExpression of result of call
        """
        if self.has_method(context, expr, "__call__"):
            return self.convert_method_call(context, expr, "__call__", args, kwargs)
        return super().convert_call(context, expr, args, kwargs)

    def convert_len(self, context, expr):
        """Generates code for calling len on expr (Class or Alternative).

        Returns:
            TypedExpression of type int representing len
        """
        if self.has_method(context, expr, "__len__"):
            return self.convert_method_call(context, expr, "__len__", (), {})
        return super().convert_len(self, context, expr)

    def convert_abs(self, context, expr):
        """Generates code for calling abs on expr (Class or Alternative).

        Returns:
            TypedExpression of abs result
        """
        return self.convert_method_call(context, expr, "__abs__", (), {})

    def convert_getitem(self, context, instance, item):
        """Generates code for getting an item in expr (Class or Alternative).

        Args:
            context: ExpressionConversionContext
            instance: TypedExpression (Class or Alternative)
            item: TypedExpression index for getitem
        Returns:
            TypedExpression of result of getitem, i.e. instance[item]
        """
        if self.has_method(context, instance, "__getitem__"):
            return self.convert_method_call(context, instance, "__getitem__", (item,), {})
        return super().convert_getitem(context, instance, item)

    def convert_setitem(self, context, instance, item, value):
        """Generates code for setting an item in expr (Class or Alternative).

        In other words, generates code for instance[item] = value

        Args:
            context: ExpressionConversionContext
            instance: TypedExpression (Class or Alternative)
            item: TypedExpression index for setitem
            value: TypedExpression for value
        """

        if self.has_method(context, instance, "__setitem__"):
            return self.convert_method_call(context, instance, "__setitem__", (item, value), {})
        return super().convert_setitem(context, instance, item, value)

    def convert_set_attribute(self, context, instance, attribute, value):
        """Generates code for setting an attribute of a Class or Alternative.

        In other words, generates code for instance.attribute = value.
        Use value=None to indicate deleting an attribute.

        Args:
            context: ExpressionConversionContext
            instance: TypedExpression (Class or Alternative)
            attribute: TypedExpression name of attribute
            value: TypedExpression value to be set
        """
        if value is None:
            if self.has_method(context, instance, "__delattr__"):
                return self.convert_method_call(context, instance, "__delattr__", (context.constant(attribute),), {})

            return super().convert_set_attribute(context, instance, attribute, value)

        if self.has_method(context, instance, "__setattr__"):
            return self.convert_method_call(context, instance, "__setattr__", (context.constant(attribute), value), {})

        return super().convert_set_attribute(context, instance, attribute, value)

    def convert_repr(self, context, instance):
        """Generates code for calling repr on a Class or Alternative instance.

        Returns:
            TypedExpression of type str
        """
        if self.has_method(context, instance, "__repr__"):
            return self.convert_method_call(context, instance, "__repr__", (), {})
        return super().convert_repr(context, instance)

    def can_cast_to_primitive(self, context, e, primitiveType) -> bool:
        """Returns true if we can call the specified 'convert_X_cast' function.

        Args:
            context: ExpressionConversionContext
            e: TypedExpression (Class or Alternative)
            primitiveType: one of bool, int, float, str, bytes
        """
        if primitiveType is bool:
            return True

        if primitiveType is str:
            return True

        if primitiveType is int:
            return self.has_method(context, e, "__int__")

        if primitiveType is float:
            return self.has_method(context, e, "__float__")

        if primitiveType is bool:
            return self.has_method(context, e, "__bool__")

        return super().can_cast_to_primitive(context, e, primitiveType)

    def convert_bool_cast(self, context, e):
        """Generates code to cast e to bool.
        """
        if self.has_method(context, e, "__bool__"):
            return self.convert_method_call(context, e, "__bool__", (), {})

        if self.has_method(context, e, "__len__"):
            res = self.convert_method_call(context, e, "__len__", (), {})

            if res is not None:
                return context.pushPod(bool, res.nonref_expr.neq(0))
            return res

        return context.constant(True)

    def convert_str_cast(self, context, e):
        """Generates code to cast e to str.
        """
        if self.has_method(context, e, "__str__"):
            return self.convert_method_call(context, e, "__str__", (), {})
        return super().convert_str_cast(context, e)

    def convert_bytes_cast(self, context, e):
        """Generates code to cast e to bytes.
        """
        if self.has_method(context, e, "__bytes__"):
            return self.convert_method_call(context, e, "__bytes__", (), {})
        return super().convert_bytes_cast(context, e)

    def convert_index_cast(self, context, e):
        """Generates code to cast e to index.
        """
        if self.has_method(context, e, '__index__'):
            res = self.convert_method_call(context, e, "__index__", (), {})
            if res is None:
                return None

            if res.expr_type == typeWrapper(int):
                return res

            intRes = context.allocateUninitializedSlot(int)

            succeeded = res.convert_to_type_with_target(intRes, explicit=False)

            with context.ifelse(succeeded.nonref_expr) as (ifTrue, ifFalse):
                with ifFalse:
                    context.pushException(TypeError, "__index__ returned non-int")

            return intRes

        return context.pushException(TypeError, f"__index__ not implemented for {self.typeRepresentation}")

    def convert_int_cast(self, context, e):
        """Generates code to cast e to int.
        """
        if self.has_method(context, e, '__int__'):
            return self.convert_method_call(context, e, "__int__", (), {})
        return context.pushException(TypeError, f"__int__ not implemented for {self.typeRepresentation}")

    def convert_float_cast(self, context, e):
        """Generates code to cast e to float.
        """
        if self.has_method(context, e, "__float__"):
            return self.convert_method_call(context, e, "__float__", (), {})
        return context.pushException(TypeError, f"__float__ not implemented for {self.typeRepresentation}")

    def convert_builtin(self, f, context, expr, a1=None):
        """Generates code to call one of the BuiltinWrapper functions.

        See BuiltinWrapper for list of functions.
        """
        if f is format:
            if self.has_method(context, expr, "__format__"):
                return self.convert_method_call(
                    context,
                    expr,
                    "__format__", (a1 if a1 is not None else context.constant(''),), {}
                )

            return expr.convert_str_cast()

        if f is round:
            if a1 is None:
                a1 = context.constant(0)

            if self.has_method(context, expr, "__round__"):
                return self.convert_method_call(context, expr, "__round__", (a1,), {})

            expr = expr.toFloat64()
            if expr is None:
                return None

            return expr.convert_builtin(f, a1)

        if a1 is not None:
            return super().convert_builtin(f, context, expr, a1)

        # handle builtins with no additional arguments here:
        methodName = {trunc: '__trunc__', floor: '__floor__', ceil: '__ceil__', complex: '__complex__', dir: '__dir__'}

        if f in methodName:
            if self.has_method(context, expr, methodName[f]):
                return self.convert_method_call(context, expr, methodName[f], (), {})

            if f in (floor, ceil, trunc):
                expr = expr.toFloat64()
                if expr is None:
                    return expr
                return expr.convert_builtin(f)

        return super().convert_builtin(f, context, expr, a1)

    def convert_unary_op(self, context, expr, op):
        """Generates code for unary operator op on expr.
        """
        magic = "__pos__" if op.matches.UAdd else \
            "__neg__" if op.matches.USub else \
            "__invert__" if op.matches.Invert else \
            "__not__" if op.matches.Not else \
            ""

        if self.has_method(context, expr, magic):
            return self.convert_method_call(context, expr, magic, (), {})

        return super().convert_unary_op(context, expr, op)

    def convert_bin_op(self, context, l, op, r, inplace: bool):
        """Generates code for binary operator op on l and r: l op r
        """
        magic = "__add__" if op.matches.Add else \
            "__sub__" if op.matches.Sub else \
            "__mul__" if op.matches.Mult else \
            "__truediv__" if op.matches.Div else \
            "__floordiv__" if op.matches.FloorDiv else \
            "__mod__" if op.matches.Mod else \
            "__matmul__" if op.matches.MatMult else \
            "__pow__" if op.matches.Pow else \
            "__lshift__" if op.matches.LShift else \
            "__rshift__" if op.matches.RShift else \
            "__or__" if op.matches.BitOr else \
            "__xor__" if op.matches.BitXor else \
            "__and__" if op.matches.BitAnd else \
            "__eq__" if op.matches.Eq else \
            "__ne__" if op.matches.NotEq else \
            "__lt__" if op.matches.Lt else \
            "__gt__" if op.matches.Gt else \
            "__le__" if op.matches.LtE else \
            "__ge__" if op.matches.GtE else \
            ""

        magic_inplace = '__i' + magic[2:] if magic and inplace else None

        if magic_inplace and self.has_method(context, l, magic_inplace):
            return self.convert_method_call(context, l, magic_inplace, (r,), {})

        if self.has_method(context, l, magic):
            return self.convert_method_call(context, l, magic, (r,), {})

        isComparison = (
            op.matches.Eq or op.matches.NotEq or op.matches.Lt
            or op.matches.Gt or op.matches.LtE or op.matches.GtE
        )

        if isComparison:
            return self.convert_comparison(context, l, op, r)
        else:
            return super().convert_bin_op(context, l, op, r, inplace)

    def convert_comparison(self, context, l, op, r):
        """Return the result of a comparison operator when no magic method exists.

        Subclasses can override this to provide specific implementations.
        """
        return super().convert_bin_op(context, l, op, r, False)

    def convert_bin_op_reverse(self, context, r, op, l, inplace):
        """Generates code for binary operator op on l and r: l op r, where r is of this wrapper's type.
        """
        if op.matches.In:
            if self.has_method(context, r, "__contains__"):
                ret = self.convert_method_call(context, r, "__contains__", (l,), {})
                if ret is not None:
                    ret = ret.toBool()
                return ret

            return super().convert_bin_op_reverse(context, r, op, l, inplace)

        magic = (
            "__radd__" if op.matches.Add else
            "__rsub__" if op.matches.Sub else
            "__rmul__" if op.matches.Mult else
            "__rtruediv__" if op.matches.Div else
            "__rfloordiv__" if op.matches.FloorDiv else
            "__rmod__" if op.matches.Mod else
            "__rmatmul__" if op.matches.MatMult else
            "__rpow__" if op.matches.Pow else
            "__rlshift__" if op.matches.LShift else
            "__rrshift__" if op.matches.RShift else
            "__ror__" if op.matches.BitOr else
            "__rxor__" if op.matches.BitXor else
            "__rand__" if op.matches.BitAnd else
            ""
        )

        if self.has_method(context, r, magic):
            return self.convert_method_call(context, r, magic, (l,), {})

        return super().convert_bin_op_reverse(context, r, op, l, inplace)
