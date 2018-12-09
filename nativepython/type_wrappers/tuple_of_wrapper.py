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
from nativepython.typed_expression import TypedExpression
from nativepython.type_wrappers.exceptions import generateThrowException
import nativepython.type_wrappers.runtime_functions as runtime_functions

from typed_python import NoneType, Int64

import nativepython.native_ast as native_ast
import nativepython

class TupleOfWrapper(Wrapper):
    is_pod = False
    is_empty = False
    is_pass_by_ref = True

    def __init__(self, t):
        assert hasattr(t, '__typed_python_category__')
        super().__init__(t)

        typeWrapper = nativepython.python_object_representation.typedPythonTypeToTypeWrapper

        self.underlyingWrapperType = typeWrapper(t.ElementType)

        self.layoutType = native_ast.Type.Struct(element_types=(
            ('refcount', native_ast.Int64),
            ('hash_cache', native_ast.Int32),
            ('count', native_ast.Int32),
            ('data', native_ast.UInt8)
            ), name='TupleOfLayout').pointer()

    def getNativeLayoutType(self):
        return self.layoutType

    def convert_bin_op(self, context, left, op, right):
        raise ConversionException("Not convertible")

    def convert_getitem(self, context, expr, item):
        expr = expr.ensureNonReference()

        return TypedExpression(
            native_ast.Expression.Branch(
                cond=((item >= 0) & (item < self.convert_len(context, expr))).nonref_expr,
                true=expr.expr.ElementPtrIntegers(0,3).cast(
                    self.underlyingWrapperType.getNativeLayoutType().pointer()
                    ).elemPtr(item.toInt64(context).nonref_expr).load(),
                false=generateThrowException(context, IndexError("tuple index out of range"))
                ),
            self.underlyingWrapperType,
            False
            )

    def convert_incref(self, context, expr):
        assert expr.isReference

        return TypedExpression(
            expr.expr.load().ElementPtrIntegers(0,0).store(
                    native_ast.Expression.Binop(
                        l=expr.expr.load().ElementPtrIntegers(0,0).load(),
                        op=native_ast.BinaryOp.Add(),
                        r=native_ast.const_int_expr(1)
                    )
                ) >> expr.expr,
            self,
            True
            )

    def convert_initialize_copy(self, context, expr, other):
        other = other.ensureNonReference()
        assert expr.isReference

        return TypedExpression.NoneExpr(
            native_ast.Expression.Branch(
                cond=other.expr,
                false=expr.expr.store(other.expr),
                true=
                    expr.expr.store(other.expr) >>
                    expr.expr.load().ElementPtrIntegers(0,0).store(
                        native_ast.Expression.Binop(
                            l=expr.expr.load().ElementPtrIntegers(0,0).load(),
                            op=native_ast.BinaryOp.Add(),
                            r=native_ast.const_int_expr(1)
                        )
                    )
                )
            )

    def convert_destroy(self, context, target):
        target = target.ensureNonReference()

        assert self.underlyingWrapperType.is_pod, "we don't know how to call destructors yet"

        return TypedExpression.NoneExpr(
            native_ast.Expression.Branch(
                cond=target.expr,
                false=native_ast.nullExpr,
                true=
                    target.expr.ElementPtrIntegers(0,0).store(
                        native_ast.Expression.Binop(
                            l=target.expr.ElementPtrIntegers(0,0).load(),
                            op=native_ast.BinaryOp.Sub(),
                            r=native_ast.const_int_expr(1)
                        )
                    ) >>
                    native_ast.Expression.Branch(
                        cond=target.expr.ElementPtrIntegers(0,0).load(),
                        true=native_ast.nullExpr,
                        false=native_ast.callFree(target.expr)
                        )
                )
            )

    def convert_len(self, context, expr):
        return TypedExpression(
            native_ast.Expression.Branch(
                cond=expr.expr,
                false=native_ast.const_int_expr(0),
                true=expr.nonref_expr.ElementPtrIntegers(0,2).load().cast(native_ast.Int64)
                ),
            Int64(),
            False
            )



