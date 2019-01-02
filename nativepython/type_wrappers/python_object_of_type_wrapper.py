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
from typed_python import NoneType, Int64
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

    def convert_call(self, context, expr, args):
        assert False

    def convert_incref(self, context, expr):
        return TypedExpression(
            context,
            native_ast.Expression.Call(
                target=runtime_functions.incref_pyobj,
                args=(expr.nonref_expr,)
                ) >> expr.expr,
            self,
            expr.isReference
            )

    def convert_assign(self, context, target, toStore):
        assert target.isReference

        return (
            toStore.convert_incref() >> 
            target.convert_destroy() >> 
            context.NoneExpr(
                target.expr.store(toStore.nonref_expr)
                )
            )

    def convert_copy_initialize(self, context, target, toStore):
        assert target.isReference

        return (
            toStore.convert_incref() >> 
             context.NoneExpr(
                target.expr.store(toStore.nonref_expr)
                )
            )

    def convert_destroy(self, context, instance):
        return context.NoneExpr(
            native_ast.Expression.Call(
                target=runtime_functions.decref_pyobj,
                args=(instance.nonref_expr,)
                )
            )


    def convert_attribute(self, context, instance, attr):
        assert isinstance(attr, str)
        return context.ValueExpr(
            native_ast.Expression.Call(
                target=runtime_functions.getattr_pyobj,
                args=(instance.nonref_expr, native_ast.const_utf8_cstr(attr))
                ),
            self
            )

    def convert_to_type(self, context, expr, target_type):
        if target_type.typeRepresentation == Int64():
            return context.ValueExpr(
                native_ast.Expression.Call(
                    target=runtime_functions.pyobj_to_int,
                    args=(expr.nonref_expr,)
                    ),
                target_type
                )

        return super().convert_to_type(context, expr, target_type)

    def convert_to_self(self, context, expr):
        if expr.expr_type.typeRepresentation == Int64():
            return context.ValueExpr(
                native_ast.Expression.Call(
                    target=runtime_functions.int_to_pyobj,
                    args=(expr.nonref_expr,)
                    ),
                self
                )

        return super().convert_to_self(context, expr)


