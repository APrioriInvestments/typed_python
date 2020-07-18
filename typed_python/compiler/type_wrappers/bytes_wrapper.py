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

from typed_python import sha_hash
from typed_python.compiler.global_variable_definition import GlobalVariableMetadata
from typed_python.compiler.type_wrappers.refcounted_wrapper import RefcountedWrapper
from typed_python.compiler.type_wrappers.bound_method_wrapper import BoundMethodWrapper
import typed_python.compiler.type_wrappers.runtime_functions as runtime_functions
from typed_python.compiler.type_wrappers.typed_list_masquerading_as_list_wrapper import TypedListMasqueradingAsList

from typed_python import Int32, ListOf

import typed_python.compiler.native_ast as native_ast
import typed_python.compiler

from typed_python.compiler.native_ast import VoidPtr

typeWrapper = lambda t: typed_python.compiler.python_object_representation.typedPythonTypeToTypeWrapper(t)


class BytesWrapper(RefcountedWrapper):
    is_pod = False
    is_empty = False
    is_pass_by_ref = True

    def __init__(self):
        super().__init__(bytes)

        self.layoutType = native_ast.Type.Struct(element_types=(
            ('refcount', native_ast.Int64),
            ('data', native_ast.UInt8)
        ), name='BytesLayout').pointer()

    def getNativeLayoutType(self):
        return self.layoutType

    def convert_hash(self, context, expr):
        return context.pushPod(Int32, runtime_functions.hash_bytes.call(expr.nonref_expr.cast(VoidPtr)))

    def on_refcount_zero(self, context, instance):
        assert instance.isReference
        return runtime_functions.free.call(instance.nonref_expr.cast(native_ast.UInt8Ptr))

    def convert_builtin(self, f, context, expr, a1=None):
        if f is bytes and a1 is None:
            return expr
        return super().convert_builtin(f, context, expr, a1)

    def convert_bin_op(self, context, left, op, right, inplace):
        if right.expr_type == left.expr_type:
            if op.matches.Eq or op.matches.NotEq or op.matches.Lt or op.matches.LtE or op.matches.GtE or op.matches.Gt:
                if left.isConstant and right.isConstant:
                    if op.matches.Eq:
                        return context.constant(left.constantValue == right.constantValue)
                    if op.matches.NotEq:
                        return context.constant(left.constantValue != right.constantValue)
                    if op.matches.Lt:
                        return context.constant(left.constantValue < right.constantValue)
                    if op.matches.LtE:
                        return context.constant(left.constantValue <= right.constantValue)
                    if op.matches.Gt:
                        return context.constant(left.constantValue > right.constantValue)
                    if op.matches.GtE:
                        return context.constant(left.constantValue >= right.constantValue)

                cmp_res = context.pushPod(
                    int,
                    runtime_functions.bytes_cmp.call(
                        left.nonref_expr.cast(VoidPtr),
                        right.nonref_expr.cast(VoidPtr)
                    )
                )
                if op.matches.Eq:
                    return context.pushPod(
                        bool,
                        cmp_res.nonref_expr.eq(0)
                    )
                if op.matches.NotEq:
                    return context.pushPod(
                        bool,
                        cmp_res.nonref_expr.neq(0)
                    )
                if op.matches.Lt:
                    return context.pushPod(
                        bool,
                        cmp_res.nonref_expr.lt(0)
                    )
                if op.matches.LtE:
                    return context.pushPod(
                        bool,
                        cmp_res.nonref_expr.lte(0)
                    )
                if op.matches.Gt:
                    return context.pushPod(
                        bool,
                        cmp_res.nonref_expr.gt(0)
                    )
                if op.matches.GtE:
                    return context.pushPod(
                        bool,
                        cmp_res.nonref_expr.gte(0)
                    )

            if op.matches.Add:
                if left.isConstant and right.isConstant:
                    return context.constant(left.constantValue + right.constantValue)

                return context.push(
                    bytes,
                    lambda bytesRef: bytesRef.expr.store(
                        runtime_functions.bytes_concat.call(
                            left.nonref_expr.cast(VoidPtr),
                            right.nonref_expr.cast(VoidPtr)
                        ).cast(self.layoutType)
                    )
                )

        return super().convert_bin_op(context, left, op, right, inplace)

    def convert_getslice(self, context, expr, lower, upper, step):
        if step is not None:
            raise Exception("Slicing with a step isn't supported yet")

        if lower is None and upper is None:
            return self

        if lower is None and upper is not None:
            lower = context.constant(0)

        lower = lower.toInt64()
        if lower is None:
            return

        upper = upper.toInt64()
        if upper is None:
            return

        if expr.isConstant and lower.isConstant and upper.isConstant:
            return context.constant(expr.constantValue[lower.constantValue:upper.constantValue])

        return context.push(
            bytes,
            lambda bytesRef: bytesRef.expr.store(
                runtime_functions.bytes_getslice_int64.call(
                    expr.nonref_expr.cast(native_ast.VoidPtr),
                    lower.nonref_expr,
                    upper.nonref_expr
                ).cast(self.layoutType)
            )
        )

    def convert_getitem(self, context, expr, item):
        item = item.toInt64()

        if item is None:
            return None

        if expr.isConstant and item.isConstant:
            return context.constant(expr.constantValue[item.constantValue])

        len_expr = self.convert_len(context, expr)

        with context.ifelse((item.nonref_expr.lt(len_expr.nonref_expr.negate()))
                            .bitor(item.nonref_expr.gte(len_expr.nonref_expr))) as (true, false):
            with true:
                context.pushException(IndexError, "index out of range")

        return context.pushPod(
            int,
            expr.nonref_expr.ElementPtrIntegers(0, 1).elemPtr(
                native_ast.Expression.Branch(
                    cond=item.nonref_expr.lt(native_ast.const_int_expr(0)),
                    false=item.nonref_expr,
                    true=item.nonref_expr.add(len_expr.nonref_expr)
                ).add(native_ast.const_int_expr(8))
            ).load().cast(native_ast.Int64)
        )

    def convert_attribute(self, context, instance, attr):
        if attr in ("split",):
            return instance.changeType(BoundMethodWrapper.Make(self, attr))

        return super().convert_attribute(context, instance, attr)

    def convert_method_call(self, context, instance, methodname, args, kwargs):
        if kwargs:
            return super().convert_method_call(context, instance, methodname, args, kwargs)

        if methodname == "split":
            if len(args) == 0:
                sepPtr = VoidPtr.zero()
                maxCount = native_ast.const_int_expr(-1)
            elif len(args) == 1 and args[0].expr_type.typeRepresentation == bytes:
                sepPtr = args[0].nonref_expr.cast(VoidPtr)
                maxCount = native_ast.const_int_expr(-1)
            elif len(args) == 2 and (
                args[0].expr_type.typeRepresentation == bytes
                and args[1].expr_type.typeRepresentation == int
            ):
                sepPtr = args[0].nonref_expr.cast(VoidPtr)
                maxCount = args[1].nonref_expr
            else:
                maxCount = None

            if maxCount is not None:
                return context.push(
                    TypedListMasqueradingAsList(ListOf(bytes)),
                    lambda outBytes: outBytes.expr.store(
                        runtime_functions.bytes_split.call(
                            instance.nonref_expr.cast(VoidPtr),
                            sepPtr,
                            maxCount
                        ).cast(outBytes.expr_type.getNativeLayoutType())
                    )
                )

        return context.pushException(AttributeError, methodname)

    def convert_len_native(self, expr):
        return native_ast.Expression.Branch(
            cond=expr,
            false=native_ast.const_int_expr(0),
            true=(
                expr.ElementPtrIntegers(0, 1).ElementPtrIntegers(4)
                .cast(native_ast.Int32.pointer()).load().cast(native_ast.Int64)
            )
        )

    def convert_len(self, context, expr):
        if expr.isConstant:
            return context.constant(len(expr.constantValue))

        return context.pushPod(int, self.convert_len_native(expr.nonref_expr))

    def constant(self, context, s):
        return typed_python.compiler.typed_expression.TypedExpression(
            context,
            native_ast.Expression.GlobalVariable(
                name='bytes_constant_' + sha_hash(s).hexdigest,
                type=native_ast.VoidPtr,
                metadata=GlobalVariableMetadata.BytesConstant(value=s)
            ).cast(self.layoutType.pointer()),
            self,
            True,
            constantValue=s
        )

    def can_cast_to_primitive(self, context, expr, primitiveType):
        return primitiveType in (bytes, float, int, bool)

    def convert_bool_cast(self, context, expr):
        if expr.isConstant:
            return context.constant(bool(expr.constantValue))

        return context.pushPod(bool, self.convert_len_native(expr.nonref_expr).neq(0))

    def convert_int_cast(self, context, expr):
        if expr.isConstant:
            try:
                return context.constant(int(expr.constantValue))
            except Exception as e:
                return context.pushException(type(e), *e.args)

        return context.pushPod(int, runtime_functions.bytes_to_int64.call(expr.nonref_expr.cast(VoidPtr)))

    def convert_float_cast(self, context, expr):
        if expr.isConstant:
            try:
                return context.constant(float(expr.constantValue))
            except Exception as e:
                return context.pushException(type(e), *e.args)

        return context.pushPod(float, runtime_functions.bytes_to_float64.call(expr.nonref_expr.cast(VoidPtr)))

    def convert_bytes_cast(self, context, expr):
        return expr
