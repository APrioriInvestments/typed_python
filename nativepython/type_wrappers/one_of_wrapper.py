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

from typed_python import NoneType, Int64, _types

import nativepython.native_ast as native_ast
import nativepython

typeWrapper = lambda t: nativepython.python_object_representation.typedPythonTypeToTypeWrapper(t)

class OneOfWrapper(Wrapper):
    is_empty = False
    is_pass_by_ref = True

    def __init__(self, t):
        assert hasattr(t, '__typed_python_category__')
        super().__init__(t)

        excessBytes = _types.bytecount(t)-1

        self.layoutType = native_ast.Type.Struct(
            element_types=(
                ('which', native_ast.UInt8),
                ('data', native_ast.Type.Array(element_type=native_ast.UInt8, count=excessBytes))
                ),
            name='OneOfLayout'
            )

        self._is_pod = all(typeWrapper(possibility).is_pod for possibility in t.Types)

    @property
    def is_pod(self):
        return self._is_pod

    def getNativeLayoutType(self):
        return self.layoutType

    def convert_bin_op(self, context, left, op, right):
        raise ConversionException("Not convertible")

    def convert_incref(self, context, expr):
        if self._is_pod:
            return context.NoneExpr()

        raise NotImplementedError()

    def convert_assign(self, context, expr, other):
        assert expr.isReference

        if self.is_pod:
            return context.NoneExpr(expr.expr.store(other.nonref_expr))
        else:
            temp = context.allocate_temporary(self)

            return context.NoneExpr(
                temp.expr.store(expr.load()) >> 
                expr.convert_copy_initialize(other).expr >> 
                temp.convert_destroy().expr
                )            

    def refAs(self, context, expr, which):
        tw = typeWrapper(self.typeRepresentation.Types[which])

        return context.RefExpr(
            expr.expr.ElementPtrIntegers(0,1).cast(tw.getNativeLayoutType().pointer()),
            tw
            )

    def convert_copy_initialize(self, context, expr, other):
        assert expr.isReference

        if self.is_pod:
            return context.NoneExpr(expr.expr.store(other.nonref_expr))
        else:
            outputExpr = native_ast.nullExpr
            for ix, t in enumerate(self.typeRepresentation.Types):
                tWrapper = typeWrapper(t)

                copy = self.refAs(context, expr, ix).convert_copy_initialize(self.refAs(context,other,ix))

                outputExpr = outputExpr >> native_ast.Expression.Branch(
                    cond=other.expr.ElementPtrIntegers(0,0).load().eq(native_ast.const_uint8_expr(ix)),
                    true=copy.expr >> expr.expr.ElementPtrIntegers(0,0).store(other.expr.ElementPtrIntegers(0,0).load()),
                    false=outputExpr
                    )

            return context.NoneExpr(outputExpr)

    def convert_destroy(self, context, expr):
        if self.is_pod:
            return context.NoneExpr()
        else:
            outputExpr = native_ast.nullExpr
            for ix, t in enumerate(self.typeRepresentation.Types):
                tWrapper = typeWrapper(t)

                if not tWrapper.is_pod:
                    destroy = self.refAs(context, expr, ix).convert_destroy()
                    outputExpr = outputExpr >> native_ast.Expression.Branch(
                        cond=expr.expr.ElementPtrIntegers(0,0).load().eq(native_ast.const_uint8_expr(ix)),
                        true=destroy.expr,
                        false=outputExpr
                        )

            return context.NoneExpr(outputExpr)

    def convert_to_type(self, context, expr, otherType):
        if otherType.typeRepresentation in self.typeRepresentation.Types:
            assert expr.isReference

            which = tuple(self.typeRepresentation.Types).index(otherType.typeRepresentation)

            result = context.allocate_temporary(otherType)

            return context.RefExpr(
                native_ast.Expression.Branch(
                    cond=expr.expr.ElementPtrIntegers(0,0).load().eq(native_ast.const_uint8_expr(which)),
                    true=result.convert_copy_initialize(self.refAs(context, expr, which)).expr
                        >> context.activates_temporary(result)
                        >> result.expr,
                    false=generateThrowException(context, Exception("Can't convert"))
                    ),
                otherType
                )

        return super().convert_to_type(context, expr, otherType)

    def convert_to_self(self, context, otherExpr):
        if otherExpr.expr_type.typeRepresentation in self.typeRepresentation.Types:
            which = tuple(self.typeRepresentation.Types).index(otherExpr.expr_type.typeRepresentation)

            result = context.allocate_temporary(self)

            return context.RefExpr(
                result.expr.ElementPtrIntegers(0,0).store(native_ast.const_uint8_expr(which)) 
                    >> self.refAs(context, result, which).convert_copy_initialize(otherExpr).expr
                    >> context.activates_temporary(result)
                    >> result.expr,
                result.expr_type
                )

        return super().convert_to_self(context, otherExpr)
