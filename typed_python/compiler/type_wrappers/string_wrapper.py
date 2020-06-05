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

from typed_python.compiler.type_wrappers.refcounted_wrapper import RefcountedWrapper
from typed_python import Int32, Dict, Float32
from typed_python.type_promotion import isInteger
from typed_python.compiler.type_wrappers.list_of_wrapper import MasqueradingListOfWrapper
import typed_python.compiler.type_wrappers.runtime_functions as runtime_functions
from typed_python.compiler.type_wrappers.bound_method_wrapper import BoundMethodWrapper

import typed_python.compiler.native_ast as native_ast
import typed_python.compiler

from typed_python import ListOf

from typed_python.compiler.native_ast import VoidPtr

typeWrapper = lambda t: typed_python.compiler.python_object_representation.typedPythonTypeToTypeWrapper(t)


_memoizedStringTable = Dict(str, ListOf(str))()


def strJoinIterable(sep, iterable):
    """Converts the iterable container to list of strings and call sep.join(iterable).

    If any of the values in the container is not string, an exception is thrown.

    :param sep: string to separate the items
    :param iterable: iterable container with strings only
    :return: string with joined values
    """
    items = ListOf(str)()

    for item in iterable:
        if isinstance(item, str):
            items.append(item)
        else:
            raise TypeError("expected str instance")
    return sep.join(items)


def strStartswith(s, prefix):
    return s[:len(prefix)] == prefix


def strEndswith(s, suffix):
    if not suffix:
        return True

    return s[-len(suffix):] == suffix


def strReplace(s, old, new, maxCount):
    if maxCount == 0:
        return s

    accumulator = ListOf(str)()

    pos = 0
    seen = 0

    while True:
        if maxCount >= 0 and seen >= maxCount:
            nextLoc = -1
        else:
            nextLoc = s.find(old, pos)

        if nextLoc >= 0:
            accumulator.append(s[pos:nextLoc])

            if len(old):
                pos = nextLoc + len(old)
            else:
                pos += 1

            seen += 1
        else:
            accumulator.append(s[pos:])
            return new.join(accumulator)


class StringWrapper(RefcountedWrapper):
    is_pod = False
    is_empty = False
    is_pass_by_ref = True

    def __init__(self):
        super().__init__(str)

        self.layoutType = native_ast.Type.Struct(element_types=(
            ('refcount', native_ast.Int64),
            ('data', native_ast.UInt8)
        ), name='StringLayout').pointer()

    def convert_type_call(self, context, typeInst, args, kwargs):
        if len(args) == 0 and not kwargs:
            return context.push(self, lambda x: x.convert_default_initialize())

        if len(args) == 1 and not kwargs:
            return args[0].convert_str_cast()

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
        return otherType.typeRepresentation is bool or otherType == self

    def _can_convert_from_type(self, otherType, explicit):
        return False

    def convert_bin_op(self, context, left, op, right, inplace):
        if right.expr_type == left.expr_type:
            if op.matches.Eq or op.matches.NotEq or op.matches.Lt or op.matches.LtE or op.matches.GtE or op.matches.Gt:
                if op.matches.Eq:
                    if left.isConstant and right.isConstant:
                        return context.constant(left.constantValue == right.constantValue)

                    return context.pushPod(
                        bool,
                        runtime_functions.string_eq.call(
                            left.nonref_expr.cast(VoidPtr),
                            right.nonref_expr.cast(VoidPtr)
                        )
                    )
                if op.matches.NotEq:
                    if left.isConstant and right.isConstant:
                        return context.constant(left.constantValue != right.constantValue)

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
                    if left.isConstant and right.isConstant:
                        return context.constant(left.constantValue < right.constantValue)

                    return context.pushPod(
                        bool,
                        cmp_res.nonref_expr.lt(0)
                    )
                if op.matches.LtE:
                    if left.isConstant and right.isConstant:
                        return context.constant(left.constantValue <= right.constantValue)

                    return context.pushPod(
                        bool,
                        cmp_res.nonref_expr.lte(0)
                    )
                if op.matches.Gt:
                    if left.isConstant and right.isConstant:
                        return context.constant(left.constantValue > right.constantValue)

                    return context.pushPod(
                        bool,
                        cmp_res.nonref_expr.gt(0)
                    )
                if op.matches.GtE:
                    if left.isConstant and right.isConstant:
                        return context.constant(left.constantValue >= right.constantValue)

                    return context.pushPod(
                        bool,
                        cmp_res.nonref_expr.gte(0)
                    )

            if op.matches.In:
                if left.isConstant and right.isConstant:
                    return context.constant(left.constantValue in right.constantValue)

                return right.convert_method_call("find", (left,), {}) >= 0

            if op.matches.Add:
                if left.isConstant and right.isConstant:
                    return context.constant(left.constantValue + right.constantValue)

                return context.push(
                    str,
                    lambda strRef: strRef.expr.store(
                        runtime_functions.string_concat.call(
                            left.nonref_expr.cast(VoidPtr),
                            right.nonref_expr.cast(VoidPtr)
                        ).cast(self.layoutType)
                    )
                )

        # emulate a few specific error strings
        if op.matches.Add and left.expr_type.typeRepresentation is str \
                and right.expr_type.is_arithmetic:
            if isInteger(right.expr_type.typeRepresentation):
                return context.pushException(TypeError, "must be str, not int")
            elif right.expr_type.typeRepresentation in (float, Float32):
                return context.pushException(TypeError, "must be str, not float")
            elif right.expr_type.typeRepresentation is bool:
                return context.pushException(TypeError, "must be str, not bool")

        return super().convert_bin_op(context, left, op, right, inplace)

    def convert_builtin(self, f, context, expr, a1=None):
        if a1 is None and f is ord:
            return context.pushPod(
                int,
                runtime_functions.string_ord.call(
                    expr.nonref_expr.cast(native_ast.VoidPtr)
                )
            )

        return super().convert_builtin(f, context, expr, a1)

    def convert_getslice(self, context, expr, lower, upper, step):
        if step is not None:
            raise Exception("Slicing with a step isn't supported yet")

        if lower is None and upper is None:
            return self

        if lower is None and upper is not None:
            lower = context.constant(0)

        if lower is not None and upper is None:
            upper = expr.convert_len()

        lower = lower.toInt64()
        if lower is None:
            return

        upper = upper.toInt64()
        if upper is None:
            return

        if expr.isConstant and lower.isConstant:
            return context.constant(
                expr.constantValue[
                    lower.constantValue:upper.constantValue
                ]
            )

        return context.push(
            str,
            lambda strRef: strRef.expr.store(
                runtime_functions.string_getslice_int64.call(
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

        if len_expr is None:
            return None

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
        if expr.constantValue is not None:
            return context.constant(len(expr.constantValue))

        return context.pushPod(int, self.convert_len_native(expr.nonref_expr))

    def constant(self, context, s):
        if s not in _memoizedStringTable:
            _memoizedStringTable[s] = ListOf(str)([s])

        pointerVal = int(_memoizedStringTable[s].pointerUnsafe(0))

        return typed_python.compiler.typed_expression.TypedExpression(
            context,
            native_ast.const_uint64_expr(pointerVal).cast(self.layoutType.pointer()),
            self,
            True,
            constantValue=s
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
        if (
            attr in ("find", "split", "join", 'strip', 'rstrip', 'lstrip', "startswith", "endswith", "replace")
            or attr in self._str_methods
            or attr in self._bool_methods
        ):
            return instance.changeType(BoundMethodWrapper.Make(self, attr))

        return super().convert_attribute(context, instance, attr)

    def convert_method_call(self, context, instance, methodname, args, kwargs):
        if not (methodname in ("find", "split", "join", 'strip', 'rstrip', 'lstrip', "startswith", "endswith", "replace")
                or methodname in self._str_methods or methodname in self._bool_methods):
            return context.pushException(AttributeError, methodname)

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
                    bool,
                    lambda bRef: bRef.expr.store(
                        self._bool_methods[methodname].call(
                            instance.nonref_expr.cast(VoidPtr)
                        )
                    )
                )
        elif methodname == "startswith":
            if len(args) == 1:
                if args[0].expr_type != self:
                    context.pushException(
                        TypeError,
                        "startswith first arg must be str (tuple of str not supported yet)"
                    )
                    return

                return context.call_py_function(strStartswith, (instance, args[0]), {})

        elif methodname == "endswith":
            if len(args) == 1:
                if args[0].expr_type != self:
                    context.pushException(
                        TypeError,
                        "endswith first arg must be str (tuple of str not supported yet)"
                    )
                    return

                return context.call_py_function(strEndswith, (instance, args[0]), {})

        elif methodname == "replace":
            if len(args) in [2, 3]:
                for i in [0, 1]:
                    if args[i].expr_type != self:
                        context.pushException(
                            TypeError,
                            f"replace() argument {i + 1} must be str"
                        )
                        return

                if len(args) == 3 and args[2].expr_type.typeRepresentation != int:
                    context.pushException(
                        TypeError,
                        f"replace() argument 3 must be int, not {args[2].expr_type.typeRepresentation}"
                    )
                    return

                if len(args) == 2:
                    return context.call_py_function(strReplace, (instance, args[0], args[1], context.constant(-1)), {})
                else:
                    return context.call_py_function(strReplace, (instance, args[0], args[1], args[2]), {})

        elif methodname == "find":
            if len(args) == 1:
                return context.push(
                    int,
                    lambda iRef: iRef.expr.store(
                        runtime_functions.string_find_2.call(
                            instance.nonref_expr.cast(VoidPtr),
                            args[0].nonref_expr.cast(VoidPtr)
                        )
                    )
                )
            elif len(args) == 2:
                return context.push(
                    int,
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
                    int,
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
            if len(args) == 0:
                return context.push(
                    MasqueradingListOfWrapper(ListOf(str)),
                    lambda outStrings: outStrings.expr.store(
                        runtime_functions.string_split_2.call(
                            instance.nonref_expr.cast(VoidPtr)
                        ).cast(outStrings.expr_type.getNativeLayoutType())
                    )
                )
            elif len(args) == 1 and args[0].expr_type.typeRepresentation == str:
                return context.push(
                    MasqueradingListOfWrapper(ListOf(str)),
                    lambda outStrings: outStrings.expr.store(
                        runtime_functions.string_split_3.call(
                            instance.nonref_expr.cast(VoidPtr),
                            args[0].nonref_expr.cast(VoidPtr)
                        ).cast(outStrings.expr_type.getNativeLayoutType())
                    )
                )
            elif len(args) == 1 and args[0].expr_type.typeRepresentation == int:
                return context.push(
                    MasqueradingListOfWrapper(ListOf(str)),
                    lambda outStrings: outStrings.expr.store(
                        runtime_functions.string_split_3max.call(
                            instance.nonref_expr.cast(VoidPtr),
                            args[0].nonref_expr
                        ).cast(outStrings.expr_type.getNativeLayoutType())
                    )
                )
            elif len(args) == 2 and args[0].expr_type.typeRepresentation == str and args[1].expr_type.typeRepresentation == int:
                return context.push(
                    MasqueradingListOfWrapper(ListOf(str)),
                    lambda outStrings: outStrings.expr.store(
                        runtime_functions.string_split.call(
                            instance.nonref_expr.cast(VoidPtr),
                            args[0].nonref_expr.cast(VoidPtr),
                            args[1].nonref_expr
                        ).cast(outStrings.expr_type.getNativeLayoutType())
                    )
                )

        return super().convert_method_call(context, instance, methodname, args, kwargs)

    def can_cast_to_primitive(self, context, e, primitiveType):
        return primitiveType in (bool, int, float, str)

    def convert_bool_cast(self, context, expr):
        if expr.isConstant:
            try:
                return context.constant(bool(expr.constantValue))
            except Exception as e:
                return context.pushException(type(e), *e.args)

        return context.pushPod(bool, self.convert_len_native(expr.nonref_expr).neq(0))

    def convert_int_cast(self, context, expr):
        if expr.isConstant:
            try:
                return context.constant(int(expr.constantValue))
            except Exception as e:
                return context.pushException(type(e), *e.args)

        return context.pushPod(int, runtime_functions.str_to_int64.call(expr.nonref_expr.cast(VoidPtr)))

    def convert_str_cast(self, context, expr):
        return expr

    def convert_float_cast(self, context, expr):
        if expr.isConstant:
            try:
                return context.constant(float(expr.constantValue))
            except Exception as e:
                return context.pushException(type(e), *e.args)

        return context.pushPod(float, runtime_functions.str_to_float64.call(expr.nonref_expr.cast(VoidPtr)))
