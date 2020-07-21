#   Copyright 2017-2020 typed_python Authors
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
from typed_python.compiler.type_wrappers.wrapper import Wrapper
from typed_python.compiler.type_wrappers.refcounted_wrapper import RefcountedWrapper
from typed_python.compiler.type_wrappers.bound_method_wrapper import BoundMethodWrapper
import typed_python.compiler.type_wrappers.runtime_functions as runtime_functions
from typed_python.compiler.type_wrappers.typed_list_masquerading_as_list_wrapper import TypedListMasqueradingAsList

from typed_python import UInt8, Int32, ListOf

import typed_python.compiler.native_ast as native_ast
import typed_python.compiler

from typed_python.compiler.native_ast import VoidPtr

typeWrapper = lambda t: typed_python.compiler.python_object_representation.typedPythonTypeToTypeWrapper(t)


def bytes_isalnum(x):
    if len(x) == 0:
        return False
    for i in x:
        if i < ord('0') or (i > ord('9') and i < ord('A')) or (i > ord('Z') and i < ord('a')) or i > ord('z'):
            return False
    return True


def bytes_isalpha(x):
    if len(x) == 0:
        return False
    for i in x:
        if i < ord('A') or (i > ord('Z') and i < ord('a')) or i > ord('z'):
            return False
    return True


def bytes_isdigit(x):
    if len(x) == 0:
        return False
    for i in x:
        if i < ord('0') or i > ord('9'):
            return False
    return True


def bytes_islower(x):
    found_lower = False
    for i in x:
        if i >= ord('a') and i <= ord('z'):
            found_lower = True
        elif i >= ord('A') and i <= ord('Z'):
            return False
    return found_lower


def bytes_isspace(x):
    if len(x) == 0:
        return False
    for i in x:
        if i != ord(' ') and i != ord('\t') and i != ord('\n') and i != ord('\r') and i != 0x0b and i != ord('\f'):
            return False
    return True


def bytes_istitle(x):
    if len(x) == 0:
        return False
    last_cased = False
    found_one = False
    for i in x:
        upper = i >= ord('A') and i <= ord('Z')
        lower = i >= ord('a') and i <= ord('z')
        if upper and last_cased:
            return False
        if lower and not last_cased:
            return False
        last_cased = upper or lower
        if last_cased:
            found_one = True
    return found_one


def bytes_isupper(x):
    found_upper = False
    for i in x:
        if i >= ord('A') and i <= ord('Z'):
            found_upper = True
        elif i >= ord('a') and i <= ord('z'):
            return False
    return found_upper


class BytesWrapper(RefcountedWrapper):
    is_pod = False
    is_empty = False
    is_pass_by_ref = True

    def __init__(self):
        super().__init__(bytes)

        self.layoutType = native_ast.Type.Struct(element_types=(
            ('refcount', native_ast.Int64),
            ('hash_cache', native_ast.Int32),
            ('bytecount', native_ast.Int32),
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
            expr.nonref_expr.ElementPtrIntegers(0, 3).elemPtr(
                native_ast.Expression.Branch(
                    cond=item.nonref_expr.lt(native_ast.const_int_expr(0)),
                    false=item.nonref_expr,
                    true=item.nonref_expr.add(len_expr.nonref_expr)
                )
            ).load().cast(native_ast.Int64)
        )

    _bool_methods = dict(
        isalnum=bytes_isalnum,
        isalpha=bytes_isalpha,
        isdigit=bytes_isdigit,
        islower=bytes_islower,
        isspace=bytes_isspace,
        istitle=bytes_istitle,
        isupper=bytes_isupper,
    )

    _bytes_methods = dict()

    def convert_attribute(self, context, instance, attr):
        if (
                attr in ("find", "split", "join", 'strip', 'rstrip', 'lstrip', "startswith", "endswith", "replace", "__iter__")
                or attr in self._bytes_methods
                or attr in self._bool_methods
        ):
            return instance.changeType(BoundMethodWrapper.Make(self, attr))

        return super().convert_attribute(context, instance, attr)

    def convert_method_call(self, context, instance, methodname, args, kwargs):
        if kwargs:
            return super().convert_method_call(context, instance, methodname, args, kwargs)

        if methodname == "__iter__" and not args and not kwargs:
            res = context.push(
                _BytesIteratorWrapper,
                lambda instance:
                instance.expr.ElementPtrIntegers(0, 0).store(-1)
            )

            context.pushReference(
                self,
                res.expr.ElementPtrIntegers(0, 1)
            ).convert_copy_initialize(instance)

            return res

        if methodname in self._bool_methods and not args and not kwargs:
            return context.call_py_function(self._bool_methods[methodname], (instance,), {})

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

    def convert_getitem_unsafe(self, context, expr, item):
        return context.push(
            UInt8,
            lambda intRef: intRef.expr.store(
                expr.nonref_expr.ElementPtrIntegers(0, 3)
                    .elemPtr(item.toInt64().nonref_expr).load()
            )
        )

    def convert_len_native(self, expr):
        return native_ast.Expression.Branch(
            cond=expr,
            false=native_ast.const_int_expr(0),
            true=(
                expr.ElementPtrIntegers(0, 2).load().cast(native_ast.Int64)
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

    def get_iteration_expressions(self, context, expr):
        if expr.isConstant:
            return [context.constant(expr.constantValue[i]) for i in range(len(expr.constantValue))]
        else:
            return None


class BytesIteratorWrapper(Wrapper):
    is_pod = False
    is_empty = False
    is_pass_by_ref = True

    def __init__(self):
        super().__init__((bytes, "iterator"))

    def getNativeLayoutType(self):
        return native_ast.Type.Struct(
            element_types=(("pos", native_ast.Int64), ("bytes", typeWrapper(bytes).getNativeLayoutType())),
            name="bytes_iterator"
        )

    def convert_next(self, context, inst):
        context.pushEffect(
            inst.expr.ElementPtrIntegers(0, 0).store(
                inst.expr.ElementPtrIntegers(0, 0).load().add(1)
            )
        )
        self_len = self.refAs(context, inst, 1).convert_len()
        canContinue = context.pushPod(
            bool,
            inst.expr.ElementPtrIntegers(0, 0).load().lt(self_len.nonref_expr)
        )

        nextIx = context.pushReference(int, inst.expr.ElementPtrIntegers(0, 0))
        return self.iteratedItemForReference(context, inst, nextIx), canContinue

    def refAs(self, context, expr, which):
        assert expr.expr_type == self

        if which == 0:
            return context.pushReference(int, expr.expr.ElementPtrIntegers(0, 0))

        if which == 1:
            return context.pushReference(
                bytes,
                expr.expr
                    .ElementPtrIntegers(0, 1)
                    .cast(typeWrapper(bytes).getNativeLayoutType().pointer())
            )

    def iteratedItemForReference(self, context, expr, ixExpr):
        return typeWrapper(bytes).convert_getitem_unsafe(
            context,
            self.refAs(context, expr, 1),
            ixExpr
        ).heldToRef()

    def convert_assign(self, context, expr, other):
        assert expr.isReference

        for i in range(2):
            self.refAs(context, expr, i).convert_assign(self.refAs(context, other, i))

    def convert_copy_initialize(self, context, expr, other):
        for i in range(2):
            self.refAs(context, expr, i).convert_copy_initialize(self.refAs(context, other, i))

    def convert_destroy(self, context, expr):
        self.refAs(context, expr, 1).convert_destroy()


_BytesIteratorWrapper = BytesIteratorWrapper()
