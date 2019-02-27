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
from typed_python import Int64
from nativepython.typed_expression import TypedExpression

import nativepython.native_ast as native_ast
import nativepython.type_wrappers.runtime_functions as runtime_functions


class PythonObjectOfTypeWrapper(Wrapper):
    is_pod = False
    is_empty = False
    is_pass_by_ref = True

    def __init__(self, pytype):
        super().__init__(pytype)

    def getNativeLayoutType(self):
        return native_ast.Type.Void().pointer()

    def convert_call(self, context, expr, args, kwargs):
        raise NotImplementedError()

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

    def convert_to_type(self, context, expr, target_type):
        if target_type.typeRepresentation == Int64:
            return context.pushPod(
                target_type,
                runtime_functions.pyobj_to_int.call(expr.nonref_expr)
            )

        return super().convert_to_type(context, expr, target_type)

    def convert_to_self(self, context, expr):
        if expr.expr_type.typeRepresentation == Int64:
            return context.push(
                self,
                lambda targetSlot:
                    targetSlot.expr.store(
                        runtime_functions.int_to_pyobj.call(expr.nonref_expr)
                    )
            )

        return super().convert_to_self(context, expr)
