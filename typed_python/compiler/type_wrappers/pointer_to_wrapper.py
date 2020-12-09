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
from typed_python.compiler.type_wrappers.python_type_object_wrapper import PythonTypeObjectWrapper
from typed_python.compiler.type_wrappers.bound_method_wrapper import BoundMethodWrapper
from typed_python.compiler.conversion_level import ConversionLevel

from typed_python import PointerTo, pointerTo

import typed_python.compiler.native_ast as native_ast
import typed_python.compiler

typeWrapper = lambda t: typed_python.compiler.python_object_representation.typedPythonTypeToTypeWrapper(t)


class PointerToObjectWrapper(Wrapper):
    is_pod = True
    is_empty = False
    is_pass_by_ref = False

    def __init__(self):
        super().__init__(pointerTo)

    def getNativeLayoutType(self):
        return native_ast.Type.Void()

    @Wrapper.unwrapOneOfAndValue
    def convert_call(self, context, expr, args, kwargs):
        if len(args) != 1 or kwargs:
            return super().convert_call(context, expr, args, kwargs)

        return args[0].expr_type.convert_pointerTo(context, args[0])


class PointerToWrapper(Wrapper):
    is_pod = True
    is_empty = False
    is_pass_by_ref = False

    def __init__(self, t):
        super().__init__(t)

        self.layoutType = typeWrapper(t.ElementType).getNativeLayoutType().pointer()

    def getNativeLayoutType(self):
        return self.layoutType

    def convert_default_initialize(self, context, target):
        self.convert_copy_initialize(
            context,
            target,
            typed_python.compiler.python_object_representation.pythonObjectRepresentation(context, self.typeRepresentation())
        )

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

    def convert_destroy(self, context, instance):
        pass

    def convert_bin_op(self, context, left, op, right, inplace):
        if op.matches.Add:
            right_int = right.toIndex()
            if right_int is None:
                return None

            return context.pushPod(self, left.nonref_expr.elemPtr(right_int.nonref_expr))

        if op.matches.Sub and right.expr_type == left.expr_type:
            right_int = right.toInt64()
            if right_int is None:
                return None

            left_int = left.toInt64()
            if left_int is None:
                return None

            return context.pushPod(
                int,
                left_int.nonref_expr.sub(right_int).div(typeWrapper(self.typeRepresentation.ElementType).getBytecount())
            )

        if op.matches.Lt and right.expr_type == left.expr_type:
            return context.pushPod(bool, left.nonref_expr.cast(native_ast.Int64).lt(right.nonref_expr.cast(native_ast.Int64)))

        if op.matches.Gt and right.expr_type == left.expr_type:
            return context.pushPod(bool, left.nonref_expr.cast(native_ast.Int64).gt(right.nonref_expr.cast(native_ast.Int64)))

        if op.matches.LtE and right.expr_type == left.expr_type:
            return context.pushPod(bool, left.nonref_expr.cast(native_ast.Int64).lte(right.nonref_expr.cast(native_ast.Int64)))

        if op.matches.GtE and right.expr_type == left.expr_type:
            return context.pushPod(bool, left.nonref_expr.cast(native_ast.Int64).gte(right.nonref_expr.cast(native_ast.Int64)))

        if op.matches.Eq and right.expr_type == left.expr_type:
            return context.pushPod(bool, left.nonref_expr.cast(native_ast.Int64).eq(right.nonref_expr.cast(native_ast.Int64)))

        if op.matches.NotEq and right.expr_type == left.expr_type:
            return context.pushPod(bool, left.nonref_expr.cast(native_ast.Int64).neq(right.nonref_expr.cast(native_ast.Int64)))

        return super().convert_bin_op(context, left, op, right, inplace)

    def convert_unary_op(self, context, left, op):
        if op.matches.Not:
            return left == context.zero(self)

        return super().convert_unary_op(context, left, op)

    def convert_attribute(self, context, instance, attr):
        if attr in ("set", "get", "initialize", "cast", "destroy"):
            return instance.changeType(BoundMethodWrapper.Make(self, attr))

        return typeWrapper(self.typeRepresentation.ElementType).convert_attribute_pointerTo(
            context,
            instance,
            attr
        )

    def convert_getitem(self, context, instance, key):
        addedValue = instance + key

        if addedValue is None:
            return None

        return addedValue.convert_method_call("get", (), {})

    def convert_setitem(self, context, instance, key, val):
        addedValue = instance + key

        if addedValue is None:
            return None

        return addedValue.convert_method_call("set", (val,), {})

    def convert_method_call(self, context, instance, methodname, args, kwargs):
        if kwargs:
            return super().convert_method_call(context, instance, methodname, args, kwargs)

        if methodname == "set":
            if len(args) == 1:
                val = args[0].convert_to_type(self.typeRepresentation.ElementType, ConversionLevel.Implicit)
                if val is None:
                    return None

                context.pushReference(self.typeRepresentation.ElementType, instance.nonref_expr).convert_assign(val)
                return context.pushVoid()

        if methodname == "destroy":
            if len(args) == 0:
                context.pushReference(self.typeRepresentation.ElementType, instance.nonref_expr).convert_destroy()
                return context.pushVoid()

        if methodname == "initialize":
            if len(args) == 0:
                context.pushReference(self.typeRepresentation.ElementType, instance.nonref_expr).convert_default_initialize()
                return context.pushVoid()

            if len(args) == 1:
                val = args[0].convert_to_type(self.typeRepresentation.ElementType, ConversionLevel.Implicit)
                if val is None:
                    return None

                context.pushReference(self.typeRepresentation.ElementType, instance.nonref_expr).convert_copy_initialize(val)
                return context.pushVoid()

        if methodname == "get":
            if len(args) == 0:
                return context.pushReference(self.typeRepresentation.ElementType, instance.nonref_expr)

        if methodname == "cast":
            if len(args) == 1 and isinstance(args[0].expr_type, PythonTypeObjectWrapper):
                tgtType = typeWrapper(PointerTo(args[0].expr_type.typeRepresentation.Value))
                return context.pushPod(tgtType, instance.nonref_expr.cast(tgtType.getNativeLayoutType()))

        return super().convert_method_call(context, instance, methodname, args, kwargs)

    def convert_type_call(self, context, typeInst, args, kwargs):
        if len(args) == 0 and not kwargs:
            return context.push(self, lambda x: x.convert_default_initialize())

        if len(args) == 1 and not kwargs:
            return args[0].convert_to_type(self, ConversionLevel.New)

        return super().convert_type_call(context, typeInst, args, kwargs)

    def _can_convert_to_type(self, targetType, conversionLevel):
        if not conversionLevel.isNewOrHigher():
            return False

        return targetType.typeRepresentation in (bool, int, str)

    def convert_to_type_with_target(self, context, instance, targetVal, conversionLevel, mayThrowOnFailure=False):
        if targetVal.expr_type.typeRepresentation is bool:
            context.pushEffect(targetVal.expr.store(instance.nonref_expr.cast(native_ast.Int64).neq(0)))
            return context.constant(True)

        if targetVal.expr_type.typeRepresentation is int:
            context.pushEffect(targetVal.expr.store(instance.nonref_expr.cast(native_ast.Int64)))
            return context.constant(True)

        if targetVal.expr_type.typeRepresentation is str:
            asInt = instance.toInt64()
            asStr = asInt.convert_str_cast()
            targetVal.convert_copy_initialize(context.constant("0x") + asStr)
            return context.constant(True)

        return super().convert_to_type_with_target(context, instance, targetVal, conversionLevel, mayThrowOnFailure)
