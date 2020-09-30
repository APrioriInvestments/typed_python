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

import typed_python.compiler.type_wrappers.runtime_functions as runtime_functions
from typed_python.compiler.native_ast import VoidPtr
from typed_python import Int32
import typed_python.compiler

from math import trunc, floor, ceil

typeWrapper = lambda x: typed_python.compiler.python_object_representation.typedPythonTypeToTypeWrapper(x)


class ClassOrAlternativeWrapperMixin:
    """A Mixin class that defines conversions on class and alternatives in terms of method calls."""
    def convert_hash(self, context, expr):
        if self.has_method(context, expr, "__hash__"):
            return self.convert_method_call(context, expr, "__hash__", (), {})

        return context.pushPod(
            Int32,
            runtime_functions.hash_alternative.call(
                expr.nonref_expr.cast(VoidPtr),
                context.getTypePointer(expr.expr_type.typeRepresentation)
            )
        )

        return None

    def convert_call(self, context, expr, args, kwargs):
        if self.has_method(context, expr, "__call__"):
            return self.convert_method_call(context, expr, "__call__", args, kwargs)
        return super().convert_call(context, expr, args, kwargs)

    def convert_len(self, context, expr):
        if self.has_method(context, expr, "__len__"):
            return self.convert_method_call(context, expr, "__len__", (), {})
        return super().convert_len(self, context, expr)

    def convert_abs(self, context, expr):
        return self.convert_method_call(context, expr, "__abs__", (), {})

    def convert_getitem(self, context, instance, item):
        if self.has_method(context, instance, "__getitem__"):
            return self.convert_method_call(context, instance, "__getitem__", (item,), {})
        return super().convert_getitem(context, instance, item)

    def convert_setitem(self, context, instance, item, value):
        if self.has_method(context, instance, "__setitem__"):
            return self.convert_method_call(context, instance, "__setitem__", (item, value), {})
        return super().convert_setitem(context, instance, item, value)

    def convert_set_attribute(self, context, instance, attribute, value):
        if value is None:
            if self.has_method(context, instance, "__delattr__"):
                return self.convert_method_call(context, instance, "__delattr__", (context.constant(attribute),), {})

            return super().convert_set_attribute(context, instance, attribute, value)

        if self.has_method(context, instance, "__setattr__"):
            return self.convert_method_call(context, instance, "__setattr__", (context.constant(attribute), value), {})

        return super().convert_set_attribute(context, instance, attribute, value)

    def convert_repr(self, context, instance):
        if self.has_method(context, instance, "__repr__"):
            return self.convert_method_call(context, instance, "__repr__", (), {})
        return super().convert_repr(context, instance)

    def can_cast_to_primitive(self, context, e, primitiveType):
        if primitiveType is bool:
            return True

        if primitiveType is str:
            return self.has_method(context, e, "__str__")

        if primitiveType is int:
            return self.has_method(context, e, "__int__")

        if primitiveType is float:
            return self.has_method(context, e, "__float__")

        if primitiveType is bool:
            return self.has_method(context, e, "__bool__")

        return super().can_cast_to_primitive(context, e, primitiveType)

    def convert_bool_cast(self, context, e):
        if self.has_method(context, e, "__bool__"):
            return self.convert_method_call(context, e, "__bool__", (), {})

        if self.has_method(context, e, "__len__"):
            res = self.convert_method_call(context, e, "__len__", (), {})

            if res is not None:
                return context.pushPod(bool, res.nonref_expr.neq(0))
            return res

        return context.constant(True)

    def convert_str_cast(self, context, e):
        if self.has_method(context, e, "__str__"):
            return self.convert_method_call(context, e, "__str__", (), {})
        return super().convert_str_cast(context, e)

    def convert_bytes_cast(self, context, e):
        if self.has_method(context, e, "__bytes__"):
            return self.convert_method_call(context, e, "__bytes__", (), {})
        return super().convert_bytes_cast(context, e)

    def convert_index_cast(self, context, e):
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
        if self.has_method(context, e, '__int__'):
            return self.convert_method_call(context, e, "__int__", (), {})
        return context.pushException(TypeError, f"__int__ not implemented for {self.typeRepresentation}")

    def convert_float_cast(self, context, e):
        if self.has_method(context, e, "__float__"):
            return self.convert_method_call(context, e, "__float__", (), {})
        return context.pushException(TypeError, f"__float__ not implemented for {self.typeRepresentation}")

    def convert_builtin(self, f, context, expr, a1=None):
        # TODO: this should go in some common wrapper base class for alternatives and classes, along with
        # generate method call
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
        magic = "__pos__" if op.matches.UAdd else \
            "__neg__" if op.matches.USub else \
            "__invert__" if op.matches.Invert else \
            "__not__" if op.matches.Not else \
            ""

        if self.has_method(context, expr, magic):
            return self.convert_method_call(context, expr, magic, (), {})

        return super().convert_unary_op(context, expr, op)

    def convert_bin_op(self, context, l, op, r, inplace):
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
