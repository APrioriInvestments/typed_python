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
from nativepython.type_wrappers.exceptions import generateThrowException
import nativepython.native_ast as native_ast
import nativepython

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

pyOpNotForFloat = {
    python_ast.BinaryOp.LShift(),
    python_ast.BinaryOp.RShift(),
    python_ast.BinaryOp.BitOr(),
    python_ast.BinaryOp.BitXor(),
    python_ast.BinaryOp.BitAnd()
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

    def convert_unary_op(self, context, instance, op):
        if op.matches.USub:
            return context.pushPod(self, instance.nonref_expr.negate())

        return super().convert_unary_op(context, instance, op)


class Int64Wrapper(ArithmeticTypeWrapper):
    def __init__(self):
        super().__init__(Int64)

    def getNativeLayoutType(self):
        return native_ast.Type.Int(bits=64, signed=True)

    def toFloat64(self, context, e):
        return context.pushPod(
            float,
            native_ast.Expression.Cast(
                left=e.nonref_expr,
                to_type=native_ast.Type.Float(bits=64)
            )
        )

    def convert_to_type(self, context, e, target_type):
        if target_type.typeRepresentation == Float64:
            return context.pushPod(
                float,
                native_ast.Expression.Cast(
                    left=e.nonref_expr,
                    to_type=native_ast.Type.Float(bits=64)
                )
            )
        elif target_type.typeRepresentation == Int64:
            return e
        elif target_type.typeRepresentation == Bool:
            return e != 0

        return super().convert_to_type(context, e, target_type)

    def convert_bin_op(self, context, left, op, right):
        if op.matches.Div:
            if right.expr_type == self:
                return left.toFloat64().convert_bin_op(op, right)

        if right.expr_type == left.expr_type:
            if op.matches.Mod:
                return context.pushPod(
                    self,
                    native_ast.Expression.Branch(
                        cond=right.nonref_expr,
                        true=runtime_functions.mod_int64_int64.call(left.nonref_expr, right.nonref_expr),
                        false=generateThrowException(context, ZeroDivisionError())
                    )
                )
            if op.matches.Pow:
                return context.pushPod(
                    self,
                    runtime_functions.pow_int64_int64.call(left.nonref_expr, right.nonref_expr)
                )
            if op.matches.LShift or op.matches.RShift:
                return context.pushPod(
                    self,
                    native_ast.Expression.Branch(
                        cond=(right >= 0).nonref_expr,
                        true=native_ast.Expression.Binop(
                            left=left.nonref_expr,
                            right=right.nonref_expr,
                            op=pyOpToNative[op]
                        ),
                        false=generateThrowException(context, ValueError("negative shift count"))
                    )
                )
            if op.matches.FloorDiv:
                res = left.toFloat64()
                if res is None:
                    return None
                res = res.convert_bin_op(python_ast.BinaryOp.Div(), right)
                if res is None:
                    return None
                return res.toInt64()

            if op in pyOpToNative:
                return context.pushPod(
                    self,
                    native_ast.Expression.Binop(
                        left=left.nonref_expr,
                        right=right.nonref_expr,
                        op=pyOpToNative[op]
                    )
                )
            if op in pyCompOp:
                return context.pushPod(
                    bool,
                    native_ast.Expression.Binop(
                        left=left.nonref_expr,
                        right=right.nonref_expr,
                        op=pyCompOp[op]
                    )
                )

        if isinstance(right.expr_type, Float64Wrapper):
            return left.toFloat64().convert_bin_op(op, right)

        return super().convert_bin_op(context, left, op, right)


class BoolWrapper(ArithmeticTypeWrapper):
    def __init__(self):
        super().__init__(Bool)

    def getNativeLayoutType(self):
        return native_ast.Type.Int(bits=1, signed=False)

    def convert_to_type(self, context, e, target_type):
        if target_type.typeRepresentation == Float64:
            return context.pushPod(
                float,
                native_ast.Expression.Cast(
                    left=e.nonref_expr,
                    to_type=native_ast.Type.Float(bits=64)
                )
            )
        elif target_type.typeRepresentation == Int64:
            return context.pushPod(
                int,
                native_ast.Expression.Cast(
                    left=e.nonref_expr,
                    to_type=native_ast.Type.Int(bits=64, signed=True)
                )
            )
        elif target_type.typeRepresentation == Bool:
            return e

        return super().convert_to_type(context, e, target_type)

    def convert_bin_op(self, context, left, op, right):
        if right.expr_type == left.expr_type:
            if op.matches.BitOr or op.matches.BitAnd or op.matches.BitXor:
                return context.pushPod(
                    self,
                    native_ast.Expression.Binop(
                        left=left.nonref_expr,
                        right=right.nonref_expr,
                        op=pyOpToNative[op]
                    )
                )

        return super().convert_bin_op(context, left, op, right)


class Float64Wrapper(ArithmeticTypeWrapper):
    def __init__(self):
        super().__init__(Float64)

    def getNativeLayoutType(self):
        return native_ast.Type.Float(bits=64)

    def convert_to_type(self, context, e, target_type):
        if target_type.typeRepresentation == Float64:
            return e
        elif target_type.typeRepresentation == Int64:
            return context.pushPod(
                int,
                native_ast.Expression.Cast(
                    left=e.nonref_expr,
                    to_type=native_ast.Type.Int(bits=64, signed=True)
                )
            )
        elif target_type.typeRepresentation == Bool:
            return e != 0.0

        return super().convert_to_type(context, e, target_type)

    def convert_bin_op(self, context, left, op, right):
        if isinstance(right.expr_type, Int64Wrapper):
            right = right.toFloat64()

        if right.expr_type == left.expr_type:
            if op.matches.Mod:
                return context.pushPod(
                    self,
                    native_ast.Expression.Branch(
                        cond=right.nonref_expr,
                        true=runtime_functions.mod_float64_float64.call(left.nonref_expr, right.nonref_expr),
                        false=generateThrowException(context, ZeroDivisionError())
                    )
                )
            if op.matches.Div:
                return context.pushPod(
                    self,
                    native_ast.Expression.Branch(
                        cond=right.nonref_expr,
                        true=native_ast.Expression.Binop(
                            left=left.nonref_expr,
                            right=right.nonref_expr,
                            op=pyOpToNative[op]
                        ),
                        false=generateThrowException(context, ZeroDivisionError())
                    )
                )
            if op.matches.Pow:
                return context.pushPod(
                    self,
                    runtime_functions.pow_float64_float64.call(
                        left.nonref_expr, right.nonref_expr
                    )
                )

            if op in pyOpToNative and op not in pyOpNotForFloat:
                return context.pushPod(
                    self,
                    native_ast.Expression.Binop(
                        left=left.nonref_expr,
                        right=right.nonref_expr,
                        op=pyOpToNative[op]
                    )
                )

            if op in pyCompOp:
                return context.pushPod(
                    bool,
                    native_ast.Expression.Binop(
                        left=left.nonref_expr,
                        right=right.nonref_expr,
                        op=pyCompOp[op]
                    )
                )

        return super().convert_bin_op(context, left, op, right)
