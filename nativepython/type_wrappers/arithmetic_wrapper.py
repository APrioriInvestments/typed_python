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

import typed_python.python_ast as python_ast
from typed_python.type_promotion import computeArithmeticBinaryResultType

import nativepython.type_wrappers.runtime_functions as runtime_functions
from nativepython.type_wrappers.wrapper import Wrapper
from nativepython.type_wrappers.exceptions import generateThrowException
import nativepython.native_ast as native_ast
import nativepython

from typed_python import (
    Float32, Float64, Int64, Bool, String, Int8, UInt8, Int16, UInt16, Int32, UInt32, UInt64
)

pyOpToNative = {
    python_ast.BinaryOp.Add(): native_ast.BinaryOp.Add(),
    python_ast.BinaryOp.Sub(): native_ast.BinaryOp.Sub(),
    python_ast.BinaryOp.Mult(): native_ast.BinaryOp.Mul(),
    python_ast.BinaryOp.Div(): native_ast.BinaryOp.Div(),
    python_ast.BinaryOp.FloorDiv(): native_ast.BinaryOp.FloorDiv(),
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
    is_arithmetic = True

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

        if op.matches.Not:
            return context.pushPod(bool, instance.nonref_expr.cast(native_ast.Bool).logical_not())

        return super().convert_unary_op(context, instance, op)


def toWrapper(T):
    if T is Bool:
        return BoolWrapper()
    if T.IsInteger:
        return IntWrapper(T)
    return FloatWrapper(T)


def toFloatType(T1):
    """Convert an int or float type to the enclosing float type."""
    if not T1.IsFloat:
        if T1.Bits <= 32:
            return Float32
        else:
            return Float64
    return T1


class IntWrapper(ArithmeticTypeWrapper):
    def __init__(self, T):
        super().__init__(T)

    def getNativeLayoutType(self):
        T = self.typeRepresentation

        return native_ast.Type.Int(bits=T.Bits, signed=T.IsSignedInt)

    def convert_hash(self, context, expr):
        if self.typeRepresentation == Int64:
            return context.pushPod(Int32, runtime_functions.hash_int64.call(expr.nonref_expr))

        if self.typeRepresentation == UInt64:
            return context.pushPod(Int32, runtime_functions.hash_uint64.call(expr.nonref_expr))

        return expr.convert_to_type(Int32)

    def convert_to_type_with_target(self, context, e, targetVal, explicit):
        assert targetVal.isReference

        target_type = targetVal.expr_type

        if not explicit:
            return super().convert_to_type_with_target(context, e, targetVal, explicit)

        if target_type.typeRepresentation in (Float64, Float32):
            context.pushEffect(
                targetVal.expr.store(
                    native_ast.Expression.Cast(
                        left=e.nonref_expr,
                        to_type=native_ast.Type.Float(bits=target_type.typeRepresentation.Bits)
                    )
                )
            )
            return context.constant(True)

        if target_type.typeRepresentation == Bool and explicit:
            targetVal.convert_copy_initialize(e != 0)
            return context.constant(True)

        if isinstance(target_type, IntWrapper):
            context.pushEffect(
                targetVal.expr.store(
                    native_ast.Expression.Cast(
                        left=e.nonref_expr,
                        to_type=native_ast.Type.Int(
                            bits=target_type.typeRepresentation.Bits,
                            signed=target_type.typeRepresentation.IsSignedInt
                        )
                    )
                )
            )
            return context.constant(True)

        if target_type.typeRepresentation == String:
            if self.typeRepresentation == Int64:
                strForm = context.push(
                    str,
                    lambda strRef: strRef.expr.store(
                        runtime_functions.int64_to_string.call(e.nonref_expr).cast(strRef.expr_type.layoutType)
                    )
                )
            elif self.typeRepresentation == UInt64:
                strForm = context.push(
                    str,
                    lambda strRef: strRef.expr.store(
                        runtime_functions.uint64_to_string.call(e.nonref_expr).cast(strRef.expr_type.layoutType)
                    )
                )
            else:
                suffix = {
                    Int32: 'i32',
                    UInt32: 'u32',
                    Int16: 'i16',
                    UInt16: 'u16',
                    Int8: 'i8',
                    UInt8: 'u8'
                }[self.typeRepresentation]

                strForm = e.convert_to_type(Int64).convert_to_type(str) + context.constant(suffix)

            targetVal.convert_copy_initialize(strForm)
            return context.constant(True)

        return super().convert_to_type_with_target(context, e, targetVal, explicit)

    def convert_abs(self, context, expr):
        if self.typeRepresentation.IsSignedInt:
            return context.pushPod(
                self,
                native_ast.Expression.Branch(
                    cond=(expr > 0).nonref_expr,
                    true=expr.nonref_expr,
                    false=expr.nonref_expr.negate()
                )
            )
        else:
            return context.pushPod(self, expr.nonref_expr)

    def convert_unary_op(self, context, left, op):
        if op.matches.Not:
            return context.pushPod(self, left.nonref_expr.logical_not())
        if op.matches.Invert:
            return context.pushPod(self, left.nonref_expr.bitwise_not())
        if op.matches.USub:
            return context.pushPod(self, left.nonref_expr.negate())
        if op.matches.UAdd:
            return context.pushPod(self, left.nonref_expr)

        return super().convert_unary_op(context, left, op)

    def convert_bin_op(self, context, left, op, right):
        if op.matches.Div:
            T = toWrapper(
                computeArithmeticBinaryResultType(
                    computeArithmeticBinaryResultType(
                        left.expr_type.typeRepresentation,
                        right.expr_type.typeRepresentation
                    ),
                    Float32
                )
            )
            return left.convert_to_type(T).convert_bin_op(op, right.convert_to_type(T))

        if right.expr_type != self:
            if isinstance(right.expr_type, ArithmeticTypeWrapper):
                if op.matches.Pow:
                    promoteType = toWrapper(
                        computeArithmeticBinaryResultType(
                            computeArithmeticBinaryResultType(
                                left.expr_type.typeRepresentation,
                                right.expr_type.typeRepresentation
                            ),
                            UInt64
                        )
                    )
                else:
                    promoteType = toWrapper(
                        computeArithmeticBinaryResultType(
                            self.typeRepresentation,
                            right.expr_type.typeRepresentation
                        )
                    )

                return left.convert_to_type(promoteType).convert_bin_op(op, right.convert_to_type(promoteType))

            return super().convert_bin_op(context, left, op, right)

        if op.matches.Mod:
            if left.expr_type.typeRepresentation.IsSignedInt:
                return context.pushPod(
                    int,
                    native_ast.Expression.Branch(
                        cond=right.nonref_expr,
                        true=runtime_functions.mod_int64_int64.call(
                            left.toInt64().nonref_expr,
                            right.toInt64().nonref_expr
                        ),
                        false=generateThrowException(context, ZeroDivisionError())
                    )
                ).convert_to_type(self)
            # unsigned int
            return context.pushPod(
                int,
                native_ast.Expression.Branch(
                    cond=right.nonref_expr,
                    true=runtime_functions.mod_uint64_uint64.call(
                        left.toUInt64().nonref_expr,
                        right.toUInt64().nonref_expr
                    ),
                    false=generateThrowException(context, ZeroDivisionError())
                )
            ).convert_to_type(self)
        if op.matches.Pow:
            if left.expr_type.typeRepresentation.IsSignedInt:
                return context.pushPod(
                    float,
                    runtime_functions.pow_int64_int64.call(left.toInt64().nonref_expr, right.toInt64().nonref_expr)
                ).toFloat64()
            # unsigned int
            return context.pushPod(
                float,
                runtime_functions.pow_uint64_uint64.call(left.toUInt64().nonref_expr, right.toUInt64().nonref_expr)
            ).toFloat64()
        if op.matches.LShift:
            if left.expr_type.typeRepresentation.IsSignedInt:
                return context.pushPod(
                    int,
                    runtime_functions.lshift_int64_int64.call(left.toInt64().nonref_expr, right.toInt64().nonref_expr)
                ).convert_to_type(self)
            # unsigned int
            return context.pushPod(
                int,
                runtime_functions.lshift_uint64_uint64.call(left.toUInt64().nonref_expr, right.toUInt64().nonref_expr)
            ).convert_to_type(self)
        if op.matches.RShift:
            if left.expr_type.typeRepresentation.IsSignedInt:
                return context.pushPod(
                    int,
                    runtime_functions.rshift_int64_int64.call(left.toInt64().nonref_expr, right.toInt64().nonref_expr)
                ).convert_to_type(self)
            # unsigned int
            return context.pushPod(
                int,
                runtime_functions.rshift_uint64_uint64.call(left.toUInt64().nonref_expr, right.toUInt64().nonref_expr)
            ).convert_to_type(self)
        if op.matches.FloorDiv:
            if right.expr_type.typeRepresentation.IsSignedInt:
                return context.pushPod(
                    int,
                    runtime_functions.floordiv_int64_int64.call(left.toInt64().nonref_expr, right.toInt64().nonref_expr)
                ).convert_to_type(self)
            # unsigned int
            return context.pushPod(
                self,
                native_ast.Expression.Branch(
                    cond=right.nonref_expr,
                    true=native_ast.Expression.Binop(
                        left=left.nonref_expr,
                        right=right.nonref_expr,
                        op=native_ast.BinaryOp.Div()
                    ),
                    false=generateThrowException(context, ZeroDivisionError())
                )
            )
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

        # we must have a bad binary operator
        return super().convert_bin_op(context, left, op, right)


class BoolWrapper(ArithmeticTypeWrapper):
    def __init__(self):
        super().__init__(Bool)

    def getNativeLayoutType(self):
        return native_ast.Type.Int(bits=1, signed=False)

    def convert_hash(self, context, expr):
        return expr.convert_to_type(Int32)

    def convert_to_type_with_target(self, context, e, targetVal, explicit):
        target_type = targetVal.expr_type

        if not explicit:
            return super().convert_to_type_with_target(context, e, targetVal, explicit)

        if target_type.typeRepresentation in (Float64, Float32):
            context.pushEffect(
                targetVal.expr.store(
                    native_ast.Expression.Cast(
                        left=e.nonref_expr,
                        to_type=native_ast.Type.Float(bits=target_type.typeRepresentation.Bits)
                    )
                )
            )
            return context.constant(True)

        elif isinstance(target_type, IntWrapper):
            context.pushEffect(
                targetVal.expr.store(
                    native_ast.Expression.Cast(
                        left=e.nonref_expr,
                        to_type=native_ast.Type.Int(
                            bits=target_type.typeRepresentation.Bits,
                            signed=target_type.typeRepresentation.IsSignedInt
                        )
                    )
                )
            )
            return context.constant(True)

        # elif target_type.typeRepresentation == String:
        #     #if not e.isReference:
        #     #    e = context.push(e.expr_type, lambda x: x.convert_copy_initialize(e))
        #     strForm = context.pushPod(
        #         String,
        #         lambda strRef: strRef.expr.store(
        #             runtime_functions.int64_to_string.call(e.nonref_expr).cast(strRef.expr_type.layoutType)
        #         )
        #     )
        #     targetVal.convert_copy_initialize(strForm)
        #     #targetVal.convert_copy_initialize(context.constant('TRUE'))
        #     # context.pushEffect(
        #         # native_ast.Expression.Branch(
        #         #     cond=e.nonref_expr,
        #         #     true=targetVal.expr.store(context.constant('TRUE').nonref_expr),
        #         #     false=targetVal.expr.store(context.constant('FALSE').nonref_expr)
        #         # targetVal.expr.store(
        #         #     native_ast.Expression.Branch(
        #         #         cond=e.nonref_expr,
        #         #         true=context.constant('TRUE').nonref_expr),
        #         #         false=context.constant('FALSE').nonref_expr
        #         #     )
        #     #     targetVal.expr.store(context.constant('TRUE').nonref_expr)
        #     # )
        #     return context.constant(True)

        return super().convert_to_type_with_target(context, e, targetVal, explicit)

    def convert_unary_op(self, context, left, op):
        if op.matches.Not:
            return context.pushPod(self, left.nonref_expr.logical_not())

        return super().convert_unary_op(context, left, op)

    def convert_bin_op(self, context, left, op, right):
        if op.matches.Is and right.expr_type == self:
            op = python_ast.ComparisonOp.Eq()

        if op.matches.IsNot and right.expr_type == self:
            op = python_ast.ComparisonOp.NotEq()

        if op.matches.Div:
            T = toWrapper(
                computeArithmeticBinaryResultType(
                    computeArithmeticBinaryResultType(
                        left.expr_type.typeRepresentation,
                        right.expr_type.typeRepresentation
                    ),
                    Float32
                )
            )
            return left.convert_to_type(T).convert_bin_op(op, right.convert_to_type(T))

        if right.expr_type != self:
            if isinstance(right.expr_type, ArithmeticTypeWrapper):
                promoteType = toWrapper(
                    computeArithmeticBinaryResultType(
                        self.typeRepresentation,
                        right.expr_type.typeRepresentation
                    )
                )

                return left.convert_to_type(promoteType).convert_bin_op(op, right.convert_to_type(promoteType))

            return super().convert_bin_op(context, left, op, right)

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


class FloatWrapper(ArithmeticTypeWrapper):
    def __init__(self, T):
        super().__init__(T)

    def getNativeLayoutType(self):
        return native_ast.Type.Float(bits=self.typeRepresentation.Bits)

    def convert_hash(self, context, expr):
        if self.typeRepresentation == Float32:
            return context.pushPod(Int32, runtime_functions.hash_float32.call(expr.nonref_expr))
        if self.typeRepresentation == Float64:
            return context.pushPod(Int32, runtime_functions.hash_float64.call(expr.nonref_expr))

        assert False

    def convert_to_type_with_target(self, context, e, targetVal, explicit):
        target_type = targetVal.expr_type

        if not explicit:
            return super().convert_to_type_with_target(context, e, targetVal, explicit)

        if target_type.typeRepresentation in (Float32, Float64):
            context.pushEffect(
                targetVal.expr.store(
                    native_ast.Expression.Cast(
                        left=e.nonref_expr,
                        to_type=native_ast.Type.Float(bits=target_type.typeRepresentation.Bits)
                    )
                )
            )
            return context.constant(True)

        if target_type.typeRepresentation == String:
            if self.typeRepresentation == Float64:
                func = runtime_functions.float64_to_string
            else:
                func = runtime_functions.float32_to_string

            context.pushEffect(
                targetVal.expr.store(func.call(e.nonref_expr).cast(target_type.layoutType))
            )
            return context.constant(True)

        if target_type.typeRepresentation == Int64:
            context.pushEffect(
                targetVal.expr.store(
                    native_ast.Expression.Cast(
                        left=e.nonref_expr,
                        to_type=native_ast.Type.Int(bits=64, signed=True)
                    )
                )
            )
            return context.constant(True)

        if target_type.typeRepresentation == Bool:
            targetVal.convert_copy_initialize(e != 0.0)
            return context.constant(True)

        return super().convert_to_type_with_target(context, e, targetVal, explicit)

    def convert_abs(self, context, expr):
        return context.pushPod(
            self,
            native_ast.Expression.Branch(
                cond=(expr > 0).nonref_expr,
                true=expr.nonref_expr,
                false=expr.nonref_expr.negate()
            )
        )

    def convert_unary_op(self, context, left, op):
        if op.matches.Not:
            return context.pushPod(self, left.nonref_expr.logical_not())
        if op.matches.USub:
            return context.pushPod(self, left.nonref_expr.negate())
        if op.matches.UAdd:
            return context.pushPod(self, left.nonref_expr)

        return super().convert_unary_op(context, left, op)

    def convert_bin_op(self, context, left, op, right):
        if right.expr_type != self:
            if isinstance(right.expr_type, ArithmeticTypeWrapper):
                if op.matches.Pow:
                    promoteType = toWrapper(Float64)
                else:
                    promoteType = toWrapper(
                        computeArithmeticBinaryResultType(
                            self.typeRepresentation,
                            right.expr_type.typeRepresentation
                        )
                    )
                return left.convert_to_type(promoteType).convert_bin_op(op, right.convert_to_type(promoteType))
            return super().convert_bin_op(context, left, op, right)

        if op.matches.Mod:
            # TODO: might define mod_float32_float32 instead of doing these conversions
            if left.expr_type.typeRepresentation == Float32:
                return left.toFloat64().convert_bin_op(
                    op, right.toFloat64()).convert_to_type(toWrapper(Float32))
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
                float,
                runtime_functions.pow_float64_float64.call(left.toFloat64().nonref_expr, right.toFloat64().nonref_expr)
            ).toFloat64()
        if op.matches.FloorDiv:
            return context.pushPod(
                float,
                runtime_functions.floordiv_float64_float64.call(left.toFloat64().nonref_expr, right.toFloat64().nonref_expr)
            ).convert_to_type(self)

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
