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
from typed_python._types import refTo
import typed_python.compiler.native_ast as native_ast

import typed_python.compiler

typeWrapper = lambda t: typed_python.compiler.python_object_representation.typedPythonTypeToTypeWrapper(t)


class RefToObjectWrapper(Wrapper):
    is_pod = True
    is_empty = False
    is_pass_by_ref = False

    def __init__(self):
        super().__init__(refTo)

    def getNativeLayoutType(self):
        return native_ast.Type.Void()

    @Wrapper.unwrapOneOfAndValue
    def convert_call(self, context, expr, args, kwargs):
        if len(args) != 1 or kwargs:
            return super().convert_call(context, expr, args, kwargs)

        return args[0].expr_type.convert_refTo(context, args[0])


class RefToWrapper(Wrapper):
    is_pod = True
    is_empty = False
    is_pass_by_ref = False

    def __init__(self, t):
        super().__init__(t)

        self.layoutType = typeWrapper(t.ElementType).getNativeLayoutType().pointer()

    def underlyingTypeWrapper(self):
        return typeWrapper(self.typeRepresentation.ElementType)

    def getNativeLayoutType(self):
        return self.layoutType

    def convert_assign(self, context, target, toStore):
        assert target.isReference
        context.pushEffect(
            target.expr.store(toStore.nonref_expr)
        )

    def convert_copy_initialize(self, context, target, toStore):
        assert target.isReference
        context.pushEffect(
            target.expr.store(toStore.nonref_expr)
        )

    def deref(self, instance):
        return TypedExpression(
            instance.context,
            instance.nonref_expr,
            typeWrapper(self.typeRepresentation.ElementType),
            True
        )

    def convert_destroy(self, context, instance):
        pass

    def _can_convert_to_type(self, targetType, conversionLevel):
        return self.underlyingTypeWrapper._can_convert_to_type(targetType, conversionLevel)

    def convert_to_type_with_target(self, context, instance, targetVal, conversionLevel, mayThrowOnFailure=False):
        return self.deref(instance).convert_to_type_with_target(targetVal, conversionLevel)

    def convert_bin_op(self, context, left, op, right, inplace):
        return self.deref(left).convert_bin_op(op, right, inplace)

    def convert_unary_op(self, context, left, op):
        return self.deref(left).convert_unary_op(op)

    def convert_attribute(self, context, instance, attr):
        return self.deref(instance).convert_attribute(attr)

    def convert_getitem(self, context, instance, key):
        return self.deref(instance).convert_getitem(key)

    def convert_setitem(self, context, instance, key, val):
        return self.deref(instance).convert_setitem(key, val)

    def convert_method_call(self, context, instance, methodname, args, kwargs):
        return self.deref(instance).convert_method_call(methodname, args, kwargs)

    def convert_set_attribute(self, context, instance, attribute, value):
        return self.deref(instance).convert_set_attribute(attribute, value)

    def convert_hash(self, context, expr):
        return self.deref(expr).convert_hash()

    def convert_call(self, context, expr, args, kwargs):
        self.deref(expr).convert_call(args, kwargs)

    def convert_len(self, context, expr):
        self.deref(expr).convert_len()

    def convert_abs(self, context, expr):
        self.deref(expr).convert_abs()

    def convert_repr(self, context, expr):
        self.deref(expr).convert_repr()

    def convert_builtin(self, f, context, expr, a1=None):
        self.deref(expr).convert_builtin(a1)

    def convert_comparison(self, context, l, op, r):
        self.deref(l).convert_comparison(op, r)

    def convert_bin_op_reverse(self, context, r, op, l, inplace):
        self.deref(r).convert_bin_op_reverse(op, l, inplace)
