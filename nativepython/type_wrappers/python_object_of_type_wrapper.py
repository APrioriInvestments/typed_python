#   Coyright 2017-2019 Nativepython Authors
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
from typed_python import Int64, Int32, Int16, Int8, UInt64, UInt32, UInt16, UInt8, Float32, Float64, Bool
from nativepython.typed_expression import TypedExpression

import nativepython.native_ast as native_ast
from nativepython.native_ast import VoidPtr
import nativepython.type_wrappers.runtime_functions as runtime_functions


class PythonObjectOfTypeWrapper(Wrapper):
    is_pod = False
    is_empty = False
    is_pass_by_ref = True

    def __init__(self, pytype):
        super().__init__(pytype)

    def getNativeLayoutType(self):
        return native_ast.Type.Void().pointer()

    def convert_incref(self, context, expr):
        context.pushEffect(
            runtime_functions.incref_pyobj.call(expr.nonref_expr)
        )

    def convert_default_initialize(self, context, target):
        if isinstance(None, self.typeRepresentation):
            target.convert_copy_initialize(
                TypedExpression(context, self, runtime_functions.get_pyobj_None.call(), False)
            )
            return

        context.pushException(TypeError, "Can't default-initialize %s" % self.typeRepresentation.__qualname__)

    def convert_assign(self, context, target, toStore):
        assert target.isReference

        toStore.convert_incref()
        target.convert_destroy()

        context.pushEffect(
            target.expr.store(toStore.nonref_expr)
        )

    def convert_copy_initialize(self, context, target, toStore):
        assert target.isReference

        toStore.convert_incref()

        context.pushEffect(
            target.expr.store(toStore.nonref_expr)
        )

    def convert_destroy(self, context, instance):
        context.pushEffect(
            runtime_functions.decref_pyobj.call(instance.nonref_expr)
        )

    def convert_attribute(self, context, instance, attr):
        assert isinstance(attr, str)
        return context.push(
            self,
            lambda targetSlot:
                targetSlot.expr.store(
                    runtime_functions.getattr_pyobj.call(
                        instance.nonref_expr,
                        native_ast.const_utf8_cstr(attr)
                    )
                )
        )

    def convert_to_type_with_target(self, context, e, targetVal, explicit):
        target_type = targetVal.expr_type

        if not explicit:
            return super().convert_to_type_with_target(context, e, targetVal, explicit)

        arith_calls = {
            Int64: runtime_functions.pyobj_to_int64,
            Int32: runtime_functions.pyobj_to_int32,
            Int16: runtime_functions.pyobj_to_int16,
            Int8: runtime_functions.pyobj_to_int8,
            UInt64: runtime_functions.pyobj_to_uint64,
            UInt32: runtime_functions.pyobj_to_uint32,
            UInt16: runtime_functions.pyobj_to_uint16,
            UInt8: runtime_functions.pyobj_to_uint8,
            Float64: runtime_functions.pyobj_to_float64,
            Float32: runtime_functions.pyobj_to_float32,
            Bool: runtime_functions.pyobj_to_bool
        }

        t = target_type.typeRepresentation
        if t in arith_calls:
            context.pushEffect(
                targetVal.expr.store(
                    arith_calls[t].call(e.nonref_expr)
                )
            )
            return context.constant(True)

        tp = context.getTypePointer(t)

        if tp:
            context.pushEffect(
                runtime_functions.pyobj_to_typed.call(
                    e.nonref_expr.cast(VoidPtr),
                    targetVal.expr.cast(VoidPtr),
                    tp
                )
            )
            return context.constant(True)

        return super().convert_to_type_with_target(context, e, targetVal, explicit)

    def convert_to_self_with_target(self, context, targetVal, sourceVal, explicit):
        if not explicit:
            return super().convert_to_self_with_target(context, targetVal, sourceVal, explicit)

        arith_calls = {
            Int64: runtime_functions.int64_to_pyobj,
            Int32: runtime_functions.int32_to_pyobj,
            Int16: runtime_functions.int16_to_pyobj,
            Int8: runtime_functions.int8_to_pyobj,
            UInt64: runtime_functions.uint64_to_pyobj,
            UInt32: runtime_functions.uint32_to_pyobj,
            UInt16: runtime_functions.uint16_to_pyobj,
            UInt8: runtime_functions.uint8_to_pyobj,
            Float64: runtime_functions.float64_to_pyobj,
            Float32: runtime_functions.float32_to_pyobj,
        }

        t = sourceVal.expr_type.typeRepresentation
        if t in arith_calls:
            context.pushEffect(
                targetVal.expr.store(
                    arith_calls[t].call(sourceVal.nonref_expr)
                )
            )
            return context.constant(True)

        tp = context.getTypePointer(t)

        if tp:
            if not sourceVal.isReference:
                sourceVal = context.push(sourceVal.expr_type, lambda x: x.convert_copy_initialize(sourceVal))

            context.pushEffect(
                targetVal.expr.store(
                    runtime_functions.to_pyobj.call(sourceVal.expr.cast(VoidPtr), tp)
                )
            )
            return context.constant(True)

        return super().convert_to_self_with_target(context, targetVal, sourceVal, explicit)
