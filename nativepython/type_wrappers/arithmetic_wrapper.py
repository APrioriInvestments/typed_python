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

import typed_python.python_ast as python_ast

import nativepython.type_wrappers.runtime_functions as runtime_functions
from nativepython.type_wrappers.wrapper import Wrapper
from nativepython.conversion_exception import ConversionException
from nativepython.typed_expression import TypedExpression
from nativepython.type_wrappers.exceptions import generateThrowException
import nativepython.native_ast as native_ast

from typed_python import *

pyOpToNative = {
    python_ast.BinaryOp.Add(): native_ast.BinaryOp.Add(),
    python_ast.BinaryOp.Sub(): native_ast.BinaryOp.Sub(),
    python_ast.BinaryOp.Mult(): native_ast.BinaryOp.Mul(),
    python_ast.BinaryOp.Div(): native_ast.BinaryOp.Div(),
    python_ast.BinaryOp.Mod(): native_ast.BinaryOp.Mod(),
    python_ast.BinaryOp.LShift(): native_ast.BinaryOp.LShift(),
    python_ast.BinaryOp.RShift(): native_ast.BinaryOp.RShift(),
    python_ast.BinaryOp.BitOr(): native_ast.BinaryOp.BitOr(),
    python_ast.BinaryOp.BitXor(): native_ast.BinaryOp.BitXor(),
    python_ast.BinaryOp.BitAnd(): native_ast.BinaryOp.BitAnd()
    }

pyCompOp = {
    python_ast.ComparisonOp.Eq(): native_ast.BinaryOp.Eq(),
    python_ast.ComparisonOp.NotEq(): native_ast.BinaryOp.NotEq(),
    python_ast.ComparisonOp.Lt(): native_ast.BinaryOp.Lt(),
    python_ast.ComparisonOp.LtE(): native_ast.BinaryOp.LtE(),
    python_ast.ComparisonOp.Gt(): native_ast.BinaryOp.Gt(),
    python_ast.ComparisonOp.GtE(): native_ast.BinaryOp.GtE()
    }

class ArithmeticTypeWrapper(Wrapper):
    is_pod = True
    is_pass_by_ref = False

    def convert_assign(self, context, target, toStore):
        assert target.isReference
        return TypedExpression.NoneExpr(target.expr.store(toStore.nonref_expr))

    def convert_initialize_copy(self, context, target, toStore):
        assert target.isReference
        return TypedExpression.NoneExpr(target.expr.store(toStore.nonref_expr))

    def convert_destroy(self, context, instance):
        return TypedExpression.NoneExpr()

class Int64Wrapper(ArithmeticTypeWrapper):
    def __init__(self):
        super().__init__(Int64())

    def getNativeLayoutType(self):
        return native_ast.Type.Int(bits=64,signed=True)

    def toFloat64(self, context, e):
        return TypedExpression(
            native_ast.Expression.Cast(
                left=e.nonref_expr,
                to_type=native_ast.Type.Float(bits=64)
                ),
            Float64Wrapper(),
            False
            )

    def sugar_comparison(self, left, right, which):
        if isinstance(right, int):
            return TypedExpression(
                native_ast.Expression.Binop(
                    l=left.nonref_expr,
                    r=native_ast.const_int_expr(right),
                    op=getattr(native_ast.BinaryOp, which)()
                    ),
                BoolWrapper(),
                False
                )
        elif isinstance(right, TypedExpression) and isinstance(right.expr_type, Int64Wrapper):
            return TypedExpression(
                native_ast.Expression.Binop(
                    l=left.nonref_expr,
                    r=right.nonref_expr,
                    op=getattr(native_ast.BinaryOp, which)()
                    ),
                BoolWrapper(),
                False
                )

        raise TypeError("Can't provide a syntactic-sugar for op %s and types %s and %s" % (op, left.expr_type, right.expr_type))

    def toInt64(self, context, e):
        return e

    def convert_bin_op(self, context, left, op, right):
        left = left.ensureNonReference()
        right = right.ensureNonReference()

        if op.matches.Div:
            if right.expr_type == self:
                return left.toFloat64(context).convert_bin_op(context, op, right)

        if right.expr_type == left.expr_type:
            if op.matches.Mod:
                return TypedExpression(
                    native_ast.Expression.Branch(
                        cond=right.expr,
                        true=native_ast.Expression.Call(
                            target=runtime_functions.mod_int64_int64,
                            args=(left.expr, right.expr,)
                            ),
                        false=generateThrowException(context, ZeroDivisionError())
                        ),
                    self,
                    False
                    )
            if op.matches.Pow:
                return TypedExpression(
                    native_ast.Expression.Call(
                        target=runtime_functions.pow_int64_int64,
                        args=(left.expr, right.expr,)
                        ),
                    self,
                    False
                    )
            if op.matches.LShift or op.matches.RShift:
                return TypedExpression(
                    native_ast.Expression.Branch(
                        cond=(right >= 0).nonref_expr,
                        true=native_ast.Expression.Binop(
                            l=left.expr,
                            r=right.expr,
                            op=pyOpToNative[op]
                            ),
                        false=generateThrowException(context, ValueError("negative shift count"))
                        ),
                    self,
                    False
                    )

            if op in pyOpToNative:
                return TypedExpression(
                    native_ast.Expression.Binop(
                        l=left.expr,
                        r=right.expr,
                        op=pyOpToNative[op]
                        ),
                    self,
                    False
                    )
            if op in pyCompOp:
                return TypedExpression(
                    native_ast.Expression.Binop(
                        l=left.expr,
                        r=right.expr,
                        op=pyCompOp[op]
                        ),
                    BoolWrapper(),
                    False
                    )

        if isinstance(right.expr_type, Float64Wrapper):
            return left.toFloat64(context).convert_bin_op(context, op, right)

        raise ConversionException("Not convertible: %s of type %s on %s/%s" % (op, type(op), left.expr_type, right.expr_type))

class BoolWrapper(ArithmeticTypeWrapper):
    def __init__(self):
        super().__init__(Bool())

    def getNativeLayoutType(self):
        return native_ast.Type.Int(bits=1,signed=False)

    def sugar_operator(self, left, right, opname):
        if isinstance(right, TypedExpression) and isinstance(right.expr_type, BoolWrapper):
            return TypedExpression(
                native_ast.Expression.Binop(
                    l=left.nonref_expr,
                    r=right.nonref_expr,
                    op=getattr(native_ast.BinaryOp, opname)()
                    ),
                BoolWrapper(),
                False
                )

        raise TypeError("Can't provide a syntactic-sugar for op %s and types %s and %s" % (op, left.expr_type, right.expr_type))




    def toFloat64(self, context, e):
        return TypedExpression(
            native_ast.Expression.Cast(
                left=e.nonref_expr,
                to_type=native_ast.Type.Float(bits=64)
                ),
            Float64Wrapper(),
            False
            )

    def toInt64(self, context, e):
        return TypedExpression(
            native_ast.Expression.Cast(
                left=e.nonref_expr,
                to_type=native_ast.Type.Int(bits=64, signed=True)
                ),
            Int64Wrapper(),
            False
            )

    def convert_bin_op(self, context, left, op, right):
        raise ConversionException("Not convertible: %s of type %s on %s/%s" % (op, type(op), left.expr_type, right.expr_type))

    def toBool(self, context, expr):
        return expr

class Float64Wrapper(ArithmeticTypeWrapper):
    def __init__(self):
        super().__init__(Float64())

    def getNativeLayoutType(self):
        return native_ast.Type.Float(bits=64)

    def toFloat64(self, context, e):
        return e

    def toBool(self, context, e):
        return e != 0.0

    def sugar_comparison(self, left, right, which):
        if isinstance(right, float):
            return TypedExpression(
                native_ast.Expression.Binop(
                    l=left.nonref_expr,
                    r=native_ast.const_float_expr(right),
                    op=getattr(native_ast.BinaryOp, which)()
                    ),
                BoolWrapper(),
                False
                )
        elif isinstance(right, TypedExpression) and isinstance(right.expr_type, Float64Wrapper):
            return TypedExpression(
                native_ast.Expression.Binop(
                    l=left.nonref_expr,
                    r=right.nonref_expr,
                    op=getattr(native_ast.BinaryOp, which)()
                    ),
                BoolWrapper(),
                False
                )

        raise TypeError("Can't provide a syntactic-sugar for op %s and types %s and %s" % (op, left.expr_type, right.expr_type))

    def toInt64(self, context, e):
        return TypedExpression(
            native_ast.Expression.Cast(
                left=e.nonref_expr,
                to_type=native_ast.Type.Int(bits=64, signed=True)
                ),
            Int64Wrapper(),
            False
            )

    def convert_bin_op(self, context, left, op, right):
        if isinstance(right.expr_type, Int64Wrapper):
            right = right.toFloat64(context)

        if right.expr_type == left.expr_type:
            if op.matches.Mod or op.matches.Div:
                return TypedExpression(
                    native_ast.Expression.Branch(
                        cond=right.toBool(context).nonref_expr,
                        true=native_ast.Expression.Binop(
                            l=left.nonref_expr,
                            r=right.nonref_expr,
                            op=pyOpToNative[op]
                            ),
                        false=generateThrowException(context, ZeroDivisionError())
                        ),
                    self,
                    False
                    )

            if op in pyOpToNative:
                return TypedExpression(
                    native_ast.Expression.Binop(
                        l=left.nonref_expr,
                        r=right.nonref_expr,
                        op=pyOpToNative[op]
                        ),
                    self,
                    False
                    )
            if op in pyCompOp:
                return TypedExpression(
                    native_ast.Expression.Binop(
                        l=left.nonref_expr,
                        r=right.nonref_expr,
                        op=pyCompOp[op]
                        ),
                    BoolWrapper(),
                    False
                    )

        raise ConversionException("Not convertible: %s of type %s on %s/%s" % (op, type(op), left.expr_type, right.expr_type))



