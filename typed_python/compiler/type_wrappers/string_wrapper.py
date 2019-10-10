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

from typed_python.compiler.type_wrappers.refcounted_wrapper import RefcountedWrapper
from typed_python import Int64, Bool, String, NoneType, Int32

import typed_python.compiler.type_wrappers.runtime_functions as runtime_functions
from typed_python.compiler.type_wrappers.bound_compiled_method_wrapper import BoundCompiledMethodWrapper

import typed_python.compiler.native_ast as native_ast
import typed_python.compiler

from typed_python import ListOf

from typed_python.compiler.native_ast import VoidPtr

typeWrapper = lambda t: typed_python.compiler.python_object_representation.typedPythonTypeToTypeWrapper(t)


def strJoinIterable(sep, iterable):
    """Converts the iterable container to list of strings and call sep.join(iterable).

    If any of the values in the container is not string, an exception is thrown.

    :param sep: string to separate the items
    :param iterable: iterable container with strings only
    :return: string with joined values
    """
    items = ListOf(String)()

    for item in iterable:
        if isinstance(item, str):
            items.append(item)
        else:
            raise TypeError("expected str instance")
    return sep.join(items)


class StringWrapper(RefcountedWrapper):
    is_pod = False
    is_empty = False
    is_pass_by_ref = True

    def __init__(self):
        super().__init__(String)

        self.layoutType = native_ast.Type.Struct(element_types=(
            ('refcount', native_ast.Int64),
            ('data', native_ast.UInt8)
        ), name='StringLayout').pointer()

    def convert_type_call(self, context, typeInst, args, kwargs):
        if len(args) == 0 and not kwargs:
            return context.push(self, lambda x: x.convert_default_initialize())

        if len(args) == 1 and not kwargs:
            return args[0].convert_cast(self)

        return super().convert_type_call(context, typeInst, args, kwargs)

    def convert_hash(self, context, expr):
        return context.pushPod(Int32, runtime_functions.hash_string.call(expr.nonref_expr.cast(VoidPtr)))

    def getNativeLayoutType(self):
        return self.layoutType

    def convert_default_initialize(self, context, target):
        context.pushEffect(
            target.expr.store(
                self.layoutType.zero()
            )
        )

    def on_refcount_zero(self, context, instance):
        assert instance.isReference
        return runtime_functions.free.call(instance.nonref_expr.cast(native_ast.UInt8Ptr))

    def _can_convert_to_type(self, otherType, explicit):
        return otherType.typeRepresentation is Bool or otherType == self

    def _can_convert_from_type(self, otherType, explicit):
        return False

    def convert_bin_op(self, context, left, op, right, inplace):
        if right.expr_type == left.expr_type:
            if op.matches.Eq or op.matches.NotEq or op.matches.Lt or op.matches.LtE or op.matches.GtE or op.matches.Gt:
                if op.matches.Eq:
                    return context.pushPod(
                        bool,
                        runtime_functions.string_eq.call(
                            left.nonref_expr.cast(VoidPtr),
                            right.nonref_expr.cast(VoidPtr)
                        )
                    )
                if op.matches.NotEq:
                    return context.pushPod(
                        bool,
                        runtime_functions.string_eq.call(
                            left.nonref_expr.cast(VoidPtr),
                            right.nonref_expr.cast(VoidPtr)
                        ).logical_not()
                    )

                cmp_res = context.pushPod(
                    int,
                    runtime_functions.string_cmp.call(
                        left.nonref_expr.cast(VoidPtr),
                        right.nonref_expr.cast(VoidPtr)
                    )
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
                return context.push(
                    str,
                    lambda strRef: strRef.expr.store(
                        runtime_functions.string_concat.call(
                            left.nonref_expr.cast(VoidPtr),
                            right.nonref_expr.cast(VoidPtr)
                        ).cast(self.layoutType)
                    )
                )

        return super().convert_bin_op(context, left, op, right, inplace)

    def convert_getitem(self, context, expr, item):
        item = item.toInt64()

        len_expr = self.convert_len(context, expr)

        with context.ifelse((item.nonref_expr.lt(len_expr.nonref_expr.negate()))
                            .bitor(item.nonref_expr.gte(len_expr.nonref_expr))) as (true, false):
            with true:
                context.pushException(IndexError, "string index out of range")

        return context.push(
            str,
            lambda strRef: strRef.expr.store(
                runtime_functions.string_getitem_int64.call(
                    expr.nonref_expr.cast(native_ast.VoidPtr), item.nonref_expr
                ).cast(self.layoutType)
            )
        )

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
        return context.pushPod(int, self.convert_len_native(expr.nonref_expr))

    def constant(self, context, s):
        return context.push(
            str,
            lambda strRef: strRef.expr.store(
                runtime_functions.string_from_utf8_and_len.call(
                    native_ast.const_utf8_cstr(s),
                    native_ast.const_int_expr(len(s))
                ).cast(self.layoutType)
            )
        )

    _bool_methods = dict(
        isalpha=runtime_functions.string_isalpha,
        isalnum=runtime_functions.string_isalnum,
        isdecimal=runtime_functions.string_isdecimal,
        isdigit=runtime_functions.string_isdigit,
        islower=runtime_functions.string_islower,
        isnumeric=runtime_functions.string_isnumeric,
        isprintable=runtime_functions.string_isprintable,
        isspace=runtime_functions.string_isspace,
        istitle=runtime_functions.string_istitle,
        isupper=runtime_functions.string_isupper
    )

    _str_methods = dict(
        lower=runtime_functions.string_lower,
        upper=runtime_functions.string_upper,
    )

    def convert_attribute(self, context, instance, attr):
        if attr in ("find", "split", "join", 'strip', 'rstrip', 'lstrip') or attr in self._str_methods or attr in self._bool_methods:
            return instance.changeType(BoundCompiledMethodWrapper(self, attr))

        return super().convert_attribute(context, instance, attr)

    def convert_method_call(self, context, instance, methodname, args, kwargs):
        if kwargs:
            return super().convert_method_call(context, instance, methodname, args, kwargs)

        if methodname in ['strip', 'lstrip', 'rstrip']:
            fromLeft = methodname in ['strip', 'lstrip']
            fromRight = methodname in ['strip', 'rstrip']
            if len(args) == 0:
                return context.push(
                    str,
                    lambda strRef: strRef.expr.store(
                        runtime_functions.string_strip.call(
                            instance.nonref_expr.cast(VoidPtr),
                            native_ast.const_bool_expr(fromLeft),
                            native_ast.const_bool_expr(fromRight)
                        ).cast(self.layoutType)
                    )
                )

        elif methodname in self._str_methods:
            if len(args) == 0:
                return context.push(
                    str,
                    lambda strRef: strRef.expr.store(
                        self._str_methods[methodname].call(
                            instance.nonref_expr.cast(VoidPtr)
                        ).cast(self.layoutType)
                    )
                )
        elif methodname in self._bool_methods:
            if len(args) == 0:
                return context.push(
                    Bool,
                    lambda bRef: bRef.expr.store(
                        self._bool_methods[methodname].call(
                            instance.nonref_expr.cast(VoidPtr)
                        )
                    )
                )
        elif methodname == "find":
            if len(args) == 1:
                return context.push(
                    Int64,
                    lambda iRef: iRef.expr.store(
                        runtime_functions.string_find_2.call(
                            instance.nonref_expr.cast(VoidPtr),
                            args[0].nonref_expr.cast(VoidPtr)
                        )
                    )
                )
            elif len(args) == 2:
                return context.push(
                    Int64,
                    lambda iRef: iRef.expr.store(
                        runtime_functions.string_find_3.call(
                            instance.nonref_expr.cast(VoidPtr),
                            args[0].nonref_expr.cast(VoidPtr),
                            args[1].nonref_expr
                        )
                    )
                )
            elif len(args) == 3:
                return context.push(
                    Int64,
                    lambda iRef: iRef.expr.store(
                        runtime_functions.string_find.call(
                            instance.nonref_expr.cast(VoidPtr),
                            args[0].nonref_expr.cast(VoidPtr),
                            args[1].nonref_expr,
                            args[2].nonref_expr
                        )
                    )
                )
        elif methodname == "join":
            if len(args) == 1:
                # we need to pass the list of strings
                separator = instance
                itemsToJoin = args[0]

                if itemsToJoin.expr_type.typeRepresentation is ListOf(str):
                    return context.push(
                        str,
                        lambda outStr: runtime_functions.string_join.call(
                            outStr.expr.cast(VoidPtr),
                            separator.nonref_expr.cast(VoidPtr),
                            itemsToJoin.nonref_expr.cast(VoidPtr)
                        )
                    )
                else:
                    return context.call_py_function(strJoinIterable, (separator, itemsToJoin), {})
        elif methodname == "split":
            if len(args) == 2:
                return context.push(
                    NoneType,
                    lambda Ref: Ref.expr.store(
                        runtime_functions.string_split_2.call(
                            args[0].nonref_expr.cast(VoidPtr),
                            args[1].nonref_expr.cast(VoidPtr)
                        )
                    )
                )
            elif len(args) == 3 and args[2].expr_type.typeRepresentation == String:
                return context.push(
                    NoneType,
                    lambda Ref: Ref.expr.store(
                        runtime_functions.string_split_3.call(
                            args[0].nonref_expr.cast(VoidPtr),
                            args[1].nonref_expr.cast(VoidPtr),
                            args[2].nonref_expr.cast(VoidPtr)
                        )
                    )
                )
            elif len(args) == 3 and args[2].expr_type.typeRepresentation == Int64:
                return context.push(
                    NoneType,
                    lambda Ref: Ref.expr.store(
                        runtime_functions.string_split_3max.call(
                            args[0].nonref_expr.cast(VoidPtr),
                            args[1].nonref_expr.cast(VoidPtr),
                            args[2].nonref_expr
                        )
                    )
                )
            elif len(args) == 4:
                return context.push(
                    NoneType,
                    lambda Ref: Ref.expr.store(
                        runtime_functions.string_split.call(
                            args[0].nonref_expr.cast(VoidPtr),
                            args[1].nonref_expr.cast(VoidPtr),
                            args[2].nonref_expr.cast(VoidPtr),
                            args[3].nonref_expr
                        )
                    )
                )

        return super().convert_method_call(context, instance, methodname, args, kwargs)

    def convert_cast_to_self(self, context, instance):
        t = instance.expr_type.typeRepresentation
        tp = context.getTypePointer(t)
        if tp:
            if not instance.isReference:
                instance = context.pushMove(instance)

            return context.push(
                str,
                lambda newStr:
                newStr.expr.store(
                    runtime_functions.np_str.call(instance.expr.cast(VoidPtr), tp).cast(self.getNativeLayoutType())
                )
            )
            return context.constant(True)

        return super().convert_cast_to_self(context, instance)

    def convert_to_type_with_target(self, context, e, targetVal, explicit):
        if not explicit:
            return super().convert_to_type_with_target(context, e, targetVal, explicit)

        target_type = targetVal.expr_type

        if target_type.typeRepresentation == Bool:
            context.pushEffect(
                targetVal.expr.store(
                    self.convert_len_native(e.nonref_expr).neq(0)
                )
            )
            return context.constant(True)

        return super().convert_to_type_with_target(context, e, targetVal, explicit)
