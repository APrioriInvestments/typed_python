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

import typed_python.python_ast as python_ast
from typed_python.compiler.conversion_level import ConversionLevel
from typed_python.type_promotion import computeArithmeticBinaryResultType, bitness, signedness, floatness, isSignedInt
from typed_python import _types
import typed_python.compiler.type_wrappers.runtime_functions as runtime_functions
from typed_python.compiler.type_wrappers.wrapper import Wrapper
import typed_python.compiler.native_ast as native_ast
import typed_python.compiler
from math import trunc, floor, ceil

from typed_python import (
    Float32, Int8, UInt8, Int16, UInt16, Int32, UInt32, UInt64
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

    def isIterable(self):
        return False

    def convert_default_initialize(self, context, target):
        self.convert_copy_initialize(
            context,
            target,
            typed_python.compiler.python_object_representation.pythonObjectRepresentation(
                context,
                self.typeRepresentation()
            )
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

    def convert_index_cast(self, context, expr):
        if expr.expr_type.typeRepresentation is float:
            return super().convert_index_cast(context, expr)

        return expr.toInt64()

    def convert_math_float_cast(self, context, e):
        return e.toFloat64()

    def convert_unary_op(self, context, instance, op):
        if op.matches.USub:
            return context.pushPod(self, instance.nonref_expr.negate())

        if op.matches.Not:
            return context.pushPod(bool, instance.nonref_expr.cast(native_ast.Bool).logical_not())

        return super().convert_unary_op(context, instance, op)

    def _can_convert_to_type(self, otherType, conversionLevel):
        if self == otherType:
            return True

        if conversionLevel.isNewOrHigher() and otherType.typeRepresentation is str:
            return True

        if not isinstance(otherType, ArithmeticTypeWrapper):
            return False

        if not isValidConversion(
            self.typeRepresentation,
            otherType.typeRepresentation,
            conversionLevel
        ):
            return False

        if isinstance(self, FloatWrapper) and isinstance(otherType, IntWrapper) and otherType.typeRepresentation is not bool:
            return "Maybe"

        return True

    def _can_convert_from_type(self, otherType, conversionLevel):
        return False

    def convert_to_type_constant(self, context, expr, target_type, level: ConversionLevel):
        """Given that 'expr' is a constant expression, attempt to convert it directly.

        This function should return None if it can't convert it to a constant, otherwise
        a typed expression with the constant.
        """
        if target_type.typeRepresentation in (str, int, float, bool, UInt64, UInt32, UInt16, UInt8, Int8, Int16, Int32):
            if isValidConversion(target_type.typeRepresentation, self.typeRepresentation, level):
                try:
                    return context.constant(target_type.typeRepresentation(expr.constantValue))
                except Exception as e:
                    context.pushException(type(e), *e.args)
                    return "FAILURE"

    def convert_type_call(self, context, typeInst, args, kwargs):
        if len(args) == 0 and not kwargs:
            return context.push(self, lambda x: x.convert_default_initialize())

        if len(args) == 1 and not kwargs:
            return args[0].convert_to_type(self, ConversionLevel.New)

        return super().convert_type_call(context, typeInst, args, kwargs)


def toWrapper(T):
    if T is bool:
        return BoolWrapper()
    if not floatness(T):
        return IntWrapper(T)
    return FloatWrapper(T)


def toFloatType(T1):
    """Convert an int or float type to the enclosing float type."""
    if not T1.IsFloat:
        if bitness(T1) <= 32:
            return Float32
        else:
            return float
    return T1


def isValidConversion(fromType, toType, conversionLevel):
    if (fromType is str or toType is str) and conversionLevel >= ConversionLevel.New:
        return True

    return _types.isValidArithmeticConversion(fromType, toType, conversionLevel.LEVEL)


class IntWrapper(ArithmeticTypeWrapper):
    def __init__(self, T):
        super().__init__(T)

    def getNativeLayoutType(self):
        T = self.typeRepresentation

        return native_ast.Type.Int(bits=bitness(T), signed=signedness(T))

    def convert_hash(self, context, expr):
        if self.typeRepresentation == int:
            return context.pushPod(Int32, runtime_functions.hash_int64.call(expr.nonref_expr))

        if self.typeRepresentation == UInt64:
            return context.pushPod(Int32, runtime_functions.hash_uint64.call(expr.nonref_expr))

        return expr.convert_to_type(Int32, ConversionLevel.Implicit)

    def convert_to_type_with_target(self, context, instance, targetVal, conversionLevel, mayThrowOnFailure=False):
        assert targetVal.isReference

        target_type = targetVal.expr_type

        if not isValidConversion(
            self.typeRepresentation,
            targetVal.expr_type.typeRepresentation,
            conversionLevel
        ):
            return super().convert_to_type_with_target(context, instance, targetVal, conversionLevel, mayThrowOnFailure)

        if isinstance(target_type, FloatWrapper):
            context.pushEffect(
                targetVal.expr.store(
                    native_ast.Expression.Cast(
                        left=instance.nonref_expr,
                        to_type=native_ast.Type.Float(bits=bitness(target_type.typeRepresentation))
                    )
                )
            )
            return context.constant(True)

        if isinstance(target_type, IntWrapper):
            context.pushEffect(
                targetVal.expr.store(
                    native_ast.Expression.Cast(
                        left=instance.nonref_expr,
                        to_type=native_ast.Type.Int(
                            bits=bitness(target_type.typeRepresentation),
                            signed=signedness(target_type.typeRepresentation)
                        )
                    )
                )
            )
            return context.constant(True)

        if isinstance(target_type, BoolWrapper):
            context.pushEffect(
                targetVal.expr.store(
                    instance.nonref_expr.neq(instance.expr_type.getNativeLayoutType().zero())
                )
            )
            return context.constant(True)

        if target_type.typeRepresentation is str:
            if self.typeRepresentation == int:
                context.pushEffect(
                    targetVal.expr.store(
                        runtime_functions.int64_to_string.call(instance.nonref_expr).cast(targetVal.expr_type.layoutType)
                    )
                )
                return context.constant(True)
            elif self.typeRepresentation == UInt64:
                context.pushEffect(
                    targetVal.expr.store(
                        runtime_functions.uint64_to_string.call(instance.nonref_expr).cast(targetVal.expr_type.layoutType)
                    )
                )
                return context.constant(True)
            else:
                suffix = {
                    Int32: 'i32',
                    UInt32: 'u32',
                    Int16: 'i16',
                    UInt16: 'u16',
                    Int8: 'i8',
                    UInt8: 'u8'
                }[self.typeRepresentation]

                targetVal.convert_copy_initialize(
                    instance.convert_to_type(
                        int,
                        ConversionLevel.Implicit
                    ).convert_to_type(str, ConversionLevel.New) + context.constant(suffix)
                )
                return context.constant(True)

        return super().convert_to_type_with_target(context, instance, targetVal, conversionLevel, mayThrowOnFailure)

    def convert_abs(self, context, expr):
        if isSignedInt(self.typeRepresentation):
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

    def convert_builtin(self, f, context, expr, a1=None):
        if f is chr and a1 is None:
            if expr.isConstant:
                return context.constant(chr(expr.constantValue))

            return context.push(
                str,
                lambda strRef: strRef.expr.store(
                    runtime_functions.string_chr_int64.call(
                        expr.toInt64().nonref_expr
                    ).cast(strRef.expr_type.layoutType)
                )
            )

        if f is round:
            if a1 is None:
                return context.pushPod(
                    float,
                    runtime_functions.round_float64.call(expr.toFloat64().nonref_expr, context.constant(0))
                ).convert_to_type(self, ConversionLevel.Implicit)
            else:
                return context.pushPod(
                    float,
                    runtime_functions.round_float64.call(expr.toFloat64().nonref_expr, a1.toInt64().nonref_expr)
                ).convert_to_type(self, ConversionLevel.Implicit)

        if f in [trunc, floor, ceil]:
            return context.pushPod(self, expr.nonref_expr)

        return super().convert_builtin(f, context, expr, a1)

    def convert_unary_op(self, context, left, op):
        if op.matches.Not:
            if left.isConstant:
                return context.constant(not left.constantValue)
            return context.pushPod(bool, left.nonref_expr.logical_not())
        if op.matches.Invert:
            if left.isConstant:
                return context.constant(~left.constantValue)
            return context.pushPod(self, left.nonref_expr.bitwise_not())
        if op.matches.USub:
            if left.isConstant:
                return context.constant(-left.constantValue)
            return context.pushPod(self, left.nonref_expr.negate())
        if op.matches.UAdd:
            if left.isConstant:
                return context.constant(left.constantValue)
            return context.pushPod(self, left.nonref_expr)

        return super().convert_unary_op(context, left, op)

    def convert_bin_op(self, context, left, op, right, inplace):
        if left.isConstant and right.isConstant and isinstance(right.constantValue, int):
            if op.matches.Add and not inplace:
                return context.constant(left.constantValue + right.constantValue)

            if op.matches.Sub and not inplace:
                return context.constant(left.constantValue - right.constantValue)

            if op.matches.Mul and not inplace:
                return context.constant(left.constantValue * right.constantValue)

            if op.matches.FloorDiv and not inplace:
                return context.constant(left.constantValue // right.constantValue)

        if op.matches.Div and isinstance(right.expr_type, ArithmeticTypeWrapper):
            T = toWrapper(
                computeArithmeticBinaryResultType(
                    computeArithmeticBinaryResultType(
                        left.expr_type.typeRepresentation,
                        right.expr_type.typeRepresentation
                    ),
                    Float32
                )
            )
            return left.convert_to_type(
                T,
                ConversionLevel.Implicit
            ).convert_bin_op(
                op,
                right.convert_to_type(T, ConversionLevel.Implicit)
            )

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

                return left.convert_to_type(
                    promoteType, ConversionLevel.Implicit
                ).convert_bin_op(
                    op,
                    right.convert_to_type(promoteType, ConversionLevel.Implicit)
                )

            return super().convert_bin_op(context, left, op, right, inplace)

        if op.matches.Mod:
            with context.ifelse(right.nonref_expr) as (ifTrue, ifFalse):
                with ifFalse:
                    context.pushException(ZeroDivisionError, "division by zero")

            if isSignedInt(left.expr_type.typeRepresentation):
                return context.pushPod(
                    int,
                    runtime_functions.mod_int64_int64.call(
                        left.toInt64().nonref_expr,
                        right.toInt64().nonref_expr
                    )
                ).convert_to_type(self, ConversionLevel.Implicit)

            # unsigned int
            return context.pushPod(
                int,
                runtime_functions.mod_uint64_uint64.call(
                    left.toUInt64().nonref_expr,
                    right.toUInt64().nonref_expr
                )
            ).convert_to_type(self, ConversionLevel.Implicit)
        if op.matches.Pow:
            if isSignedInt(left.expr_type.typeRepresentation):
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
            if isSignedInt(left.expr_type.typeRepresentation):
                return context.pushPod(
                    int,
                    runtime_functions.lshift_int64_int64.call(left.toInt64().nonref_expr, right.toInt64().nonref_expr)
                ).convert_to_type(self, ConversionLevel.Implicit)
            # unsigned int
            return context.pushPod(
                int,
                runtime_functions.lshift_uint64_uint64.call(left.toUInt64().nonref_expr, right.toUInt64().nonref_expr)
            ).convert_to_type(self, ConversionLevel.Implicit)
        if op.matches.RShift:
            if isSignedInt(left.expr_type.typeRepresentation):
                return context.pushPod(
                    int,
                    runtime_functions.rshift_int64_int64.call(left.toInt64().nonref_expr, right.toInt64().nonref_expr)
                ).convert_to_type(self, ConversionLevel.Implicit)
            # unsigned int
            return context.pushPod(
                int,
                runtime_functions.rshift_uint64_uint64.call(left.toUInt64().nonref_expr, right.toUInt64().nonref_expr)
            ).convert_to_type(self, ConversionLevel.Implicit)
        if op.matches.FloorDiv:
            if isSignedInt(right.expr_type.typeRepresentation):
                return context.pushPod(
                    int,
                    runtime_functions.floordiv_int64_int64.call(left.toInt64().nonref_expr, right.toInt64().nonref_expr)
                ).convert_to_type(self, ConversionLevel.Implicit)
            # unsigned int
            with context.ifelse(right.nonref_expr) as (ifTrue, ifFalse):
                with ifFalse:
                    context.pushException(ZeroDivisionError, "division by zero")

            return context.pushPod(
                self,
                native_ast.Expression.Binop(
                    left=left.nonref_expr,
                    right=right.nonref_expr,
                    op=native_ast.BinaryOp.Div()
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
        return super().convert_bin_op(context, left, op, right, inplace)


class BoolWrapper(ArithmeticTypeWrapper):
    def __init__(self):
        super().__init__(bool)

    def getNativeLayoutType(self):
        return native_ast.Type.Int(bits=1, signed=False)

    def convert_hash(self, context, expr):
        return expr.convert_to_type(Int32, ConversionLevel.Implicit)

    def convert_to_type_with_target(self, context, instance, targetVal, conversionLevel, mayThrowOnFailure=False):
        target_type = targetVal.expr_type

        if not isValidConversion(
            self.typeRepresentation,
            targetVal.expr_type.typeRepresentation,
            conversionLevel
        ):
            return super().convert_to_type_with_target(context, instance, targetVal, conversionLevel, mayThrowOnFailure)

        if isinstance(target_type, FloatWrapper):
            context.pushEffect(
                targetVal.expr.store(
                    native_ast.Expression.Cast(
                        left=instance.nonref_expr,
                        to_type=native_ast.Type.Float(bits=bitness(target_type.typeRepresentation))
                    )
                )
            )
            return context.constant(True)

        elif isinstance(target_type, IntWrapper):
            context.pushEffect(
                targetVal.expr.store(
                    native_ast.Expression.Cast(
                        left=instance.nonref_expr,
                        to_type=native_ast.Type.Int(
                            bits=bitness(target_type.typeRepresentation),
                            signed=signedness(target_type.typeRepresentation)
                        )
                    )
                )
            )
            return context.constant(True)

        elif target_type.typeRepresentation is str:
            context.pushEffect(
                targetVal.expr.store(
                    runtime_functions.bool_to_string.call(instance.nonref_expr).cast(
                        targetVal.expr_type.layoutType
                    )
                )
            )
            return context.constant(True)

        return super().convert_to_type_with_target(context, instance, targetVal, conversionLevel, mayThrowOnFailure)

    def convert_builtin(self, f, context, expr, a1=None):
        if f is round and a1 is not None:
            return context.pushPod(
                self,
                native_ast.Expression.Binop(
                    left=expr.nonref_expr,
                    right=a1.nonref_expr.gte(0),
                    op=native_ast.BinaryOp.BitAnd()
                )
            )
        if f in [round, trunc, floor, ceil]:
            return context.pushPod(self, expr.nonref_expr)

        return super().convert_builtin(f, context, expr, a1)

    def convert_unary_op(self, context, left, op):
        if op.matches.Not:
            if left.isConstant:
                return context.constant(not left.constantValue)
            return context.pushPod(bool, left.nonref_expr.logical_not())

        return super().convert_unary_op(context, left, op)

    def convert_bin_op(self, context, left, op, right, inplace):
        if op.matches.Is and right.expr_type == self:
            op = python_ast.ComparisonOp.Eq()

        if op.matches.IsNot and right.expr_type == self:
            op = python_ast.ComparisonOp.NotEq()

        if op.matches.Div and isinstance(right, ArithmeticTypeWrapper):
            T = toWrapper(
                computeArithmeticBinaryResultType(
                    computeArithmeticBinaryResultType(
                        left.expr_type.typeRepresentation,
                        right.expr_type.typeRepresentation
                    ),
                    Float32
                )
            )
            return left.convert_to_type(
                T, ConversionLevel.Implicit
            ).convert_bin_op(op, right.convert_to_type(T, ConversionLevel.Implicit))

        if right.expr_type != self:
            if isinstance(right.expr_type, ArithmeticTypeWrapper):
                promoteType = toWrapper(
                    computeArithmeticBinaryResultType(
                        self.typeRepresentation,
                        right.expr_type.typeRepresentation
                    )
                )

                return left.convert_to_type(
                    promoteType, ConversionLevel.Implicit
                ).convert_bin_op(op, right.convert_to_type(promoteType, ConversionLevel.Implicit))

            return super().convert_bin_op(context, left, op, right, inplace)

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

        if op in pyOpToNative or op.matches.Pow:
            return left.convert_to_type(int, ConversionLevel.Implicit).convert_bin_op(op, right, inplace)

        if op in pyCompOp:
            return context.pushPod(
                bool,
                native_ast.Expression.Binop(
                    left=left.nonref_expr,
                    right=right.nonref_expr,
                    op=pyCompOp[op]
                )
            )

        return super().convert_bin_op(context, left, op, right, inplace)


class FloatWrapper(ArithmeticTypeWrapper):
    def __init__(self, T):
        super().__init__(T)

    def getNativeLayoutType(self):
        return native_ast.Type.Float(bits=bitness(self.typeRepresentation))

    def convert_hash(self, context, expr):
        if self.typeRepresentation == Float32:
            return context.pushPod(Int32, runtime_functions.hash_float32.call(expr.nonref_expr))
        if self.typeRepresentation == float:
            return context.pushPod(Int32, runtime_functions.hash_float64.call(expr.nonref_expr))

        assert False

    def convert_to_type_with_target(self, context, instance, targetVal, conversionLevel, mayThrowOnFailure=False):
        target_type = targetVal.expr_type

        if not isValidConversion(
            self.typeRepresentation,
            targetVal.expr_type.typeRepresentation,
            conversionLevel
        ):
            return super().convert_to_type_with_target(context, instance, targetVal, conversionLevel, mayThrowOnFailure)

        if isinstance(target_type, FloatWrapper):
            context.pushEffect(
                targetVal.expr.store(
                    native_ast.Expression.Cast(
                        left=instance.nonref_expr,
                        to_type=native_ast.Type.Float(bits=bitness(target_type.typeRepresentation))
                    )
                )
            )
            return context.constant(True)

        if isinstance(target_type, BoolWrapper):
            context.pushEffect(
                targetVal.expr.store(
                    instance.nonref_expr.neq(instance.expr_type.getNativeLayoutType().zero())
                )
            )
            return context.constant(True)

        if isinstance(target_type, IntWrapper):
            if self.typeRepresentation == float:
                func = runtime_functions.float64_to_int
            else:
                func = runtime_functions.float32_to_int

            return context.pushPod(
                bool,
                func.call(
                    targetVal.expr.cast(native_ast.VoidPtr),
                    instance.nonref_expr,
                    native_ast.const_bool_expr(mayThrowOnFailure),
                    context.getTypePointer(targetVal.expr_type.typeRepresentation)
                )
            )

        if target_type.typeRepresentation is str:
            if self.typeRepresentation == float:
                func = runtime_functions.float64_to_string
            else:
                func = runtime_functions.float32_to_string

            context.pushEffect(
                targetVal.expr.store(func.call(instance.nonref_expr).cast(targetVal.expr_type.layoutType))
            )
            return context.constant(True)

        return super().convert_to_type_with_target(context, instance, targetVal, conversionLevel, mayThrowOnFailure)

    def convert_abs(self, context, expr):
        return context.pushPod(
            self,
            native_ast.Expression.Branch(
                cond=(expr > 0).nonref_expr,
                true=expr.nonref_expr,
                false=expr.nonref_expr.negate()
            )
        )

    def convert_builtin(self, f, context, expr, a1=None):
        if f is round:
            if a1:
                return context.pushPod(
                    float,
                    runtime_functions.round_float64.call(expr.toFloat64().nonref_expr, a1.toInt64().nonref_expr)
                ).convert_to_type(self, ConversionLevel.Implicit)
            else:
                return context.pushPod(
                    float,
                    runtime_functions.round_float64.call(expr.toFloat64().nonref_expr, context.constant(0))
                ).convert_to_type(self, ConversionLevel.Implicit)
        if f is trunc:
            return context.pushPod(
                float,
                runtime_functions.trunc_float64.call(expr.toFloat64().nonref_expr)
            )
        if f is floor:
            return context.pushPod(
                float, runtime_functions.floor_float64.call(expr.toFloat64().nonref_expr)
            )
        if f is ceil:
            return context.pushPod(
                float, runtime_functions.ceil_float64.call(expr.toFloat64().nonref_expr)
            )

        return super().convert_builtin(f, context, expr, a1)

    def convert_unary_op(self, context, left, op):
        if op.matches.Not:
            if left.isConstant:
                return context.constant(not left.constantValue)
            return context.pushPod(bool, left.nonref_expr.eq(self.getNativeLayoutType().zero()))
        if op.matches.USub:
            if left.isConstant:
                return context.constant(-left.constantValue)
            return context.pushPod(self, left.nonref_expr.negate())
        if op.matches.UAdd:
            if left.isConstant:
                return context.constant(left.constantValue)
            return context.pushPod(self, left.nonref_expr)

        return super().convert_unary_op(context, left, op)

    def convert_bin_op(self, context, left, op, right, inplace):
        if right.expr_type != self:
            if isinstance(right.expr_type, ArithmeticTypeWrapper):
                if op.matches.Pow:
                    promoteType = toWrapper(float)
                else:
                    promoteType = toWrapper(
                        computeArithmeticBinaryResultType(
                            self.typeRepresentation,
                            right.expr_type.typeRepresentation
                        )
                    )
                return left.convert_to_type(
                    promoteType, ConversionLevel.Implicit
                ).convert_bin_op(op, right.convert_to_type(promoteType, ConversionLevel.Implicit))
            return super().convert_bin_op(context, left, op, right, inplace)

        if op.matches.Mod:
            # TODO: might define mod_float32_float32 instead of doing these conversions
            if left.expr_type.typeRepresentation == Float32:
                return left.toFloat64().convert_bin_op(
                    op, right.toFloat64()).convert_to_type(Float32, ConversionLevel.Implicit)

            with context.ifelse(right.nonref_expr) as (ifTrue, ifFalse):
                with ifFalse:
                    context.pushException(ZeroDivisionError, "division by zero")

            return context.pushPod(
                self,
                runtime_functions.mod_float64_float64.call(left.nonref_expr, right.nonref_expr)
            )

        if op.matches.Div:
            if left.isConstant and right.isConstant:
                # If we divide by 0, for compatibility, we want that to remain a runtime error, not a compile error.
                try:
                    return context.constant(left.constantValue / right.constantValue).convert_to_type(self, ConversionLevel.Implicit)
                except Exception:
                    pass
            with context.ifelse(right.nonref_expr) as (ifTrue, ifFalse):
                with ifFalse:
                    context.pushException(ZeroDivisionError, "division by zero")

            return context.pushPod(
                self,
                native_ast.Expression.Binop(
                    left=left.nonref_expr,
                    right=right.nonref_expr,
                    op=pyOpToNative[op]
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
            ).convert_to_type(self, ConversionLevel.Implicit)

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

        return super().convert_bin_op(context, left, op, right, inplace)
