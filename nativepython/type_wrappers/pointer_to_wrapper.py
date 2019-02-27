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
from nativepython.type_wrappers.python_type_object_wrapper import PythonTypeObjectWrapper
from nativepython.type_wrappers.bound_compiled_method_wrapper import BoundCompiledMethodWrapper

from typed_python import Int64, PointerTo

import nativepython.native_ast as native_ast
import nativepython

typeWrapper = lambda t: nativepython.python_object_representation.typedPythonTypeToTypeWrapper(t)


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
            nativepython.python_object_representation.pythonObjectRepresentation(context, self.typeRepresentation())
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

    def convert_to_type(self, context, e, target_type):
        if target_type.typeRepresentation == Int64:
            return context.pushPod(int, e.nonref_expr.cast(native_ast.Int64))

        return super().convert_to_type(context, e, target_type)

    def convert_getitem(self, context, expr, item):
        return (expr + item).convert_method_call("get", (), None)

    def convert_bin_op(self, context, left, op, right):
        if op.matches.Add:
            right = right.toInt64()
            if right is None:
                return None

            return context.pushPod(self, left.nonref_expr.elemPtr(right.nonref_expr))

        if op.matches.Sub:
            right = right.toInt64()

            if right is None:
                return None

            left = left.toInt64()

            return context.pushPod(
                int,
                left.nonref_expr
                .sub(right.nonref_expr)
                .div(typeWrapper(self.typeRepresentation.ElementType).getBytecount())
            )

        if op.matches.Lt and right.expr_type == left.expr_type:
            return context.pushPod(bool, left.nonref_expr.cast(native_ast.Int64).lt(right.nonref_expr.cast(native_ast.Int64)))

        return super().convert_bin_op(context, left, op, right)

    def convert_attribute(self, context, instance, attr):
        if attr in ("set", "get", "initialize", "cast"):
            return instance.changeType(BoundCompiledMethodWrapper(self, attr))

        return super().convert_attribute(context, instance, attr)

    def convert_method_call(self, context, instance, methodname, args, kwargs):
        if kwargs:
            return super().convert_method_call(context, instance, methodname, args, kwargs)

        if methodname == "set":
            if len(args) == 1:
                context.pushReference(self.typeRepresentation.ElementType, instance.nonref_expr).convert_assign(args[0])
                return context.pushVoid()

        if methodname == "initialize":
            if len(args) == 0:
                context.pushReference(self.typeRepresentation.ElementType, instance.nonref_expr).convert_default_initialize()
                return context.pushVoid()

            if len(args) == 1:
                context.pushReference(self.typeRepresentation.ElementType, instance.nonref_expr).convert_copy_initialize(args[0])
                return context.pushVoid()

        if methodname == "get":
            if len(args) == 0:
                return context.pushReference(self.typeRepresentation.ElementType, instance.nonref_expr)

        if methodname == "cast":
            if len(args) == 1 and isinstance(args[0].expr_type, PythonTypeObjectWrapper):
                tgtType = typeWrapper(PointerTo(args[0].expr_type.typeRepresentation))
                return context.pushPod(tgtType, instance.nonref_expr.cast(tgtType.getNativeLayoutType()))

        return super().convert_method_call(context, instance, methodname, args, kwargs)
