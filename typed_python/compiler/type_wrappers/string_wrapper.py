#   Copyright 2017-2022 typed_python Authors
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

import sys
from typed_python import sha_hash
from typed_python.compiler.global_variable_definition import GlobalVariableMetadata
from typed_python.compiler.type_wrappers.wrapper import Wrapper
from typed_python.compiler.type_wrappers.refcounted_wrapper import RefcountedWrapper
from typed_python.type_promotion import isInteger
from typed_python.compiler.type_wrappers.typed_list_masquerading_as_list_wrapper import TypedListMasqueradingAsList
import typed_python.compiler.type_wrappers.runtime_functions as runtime_functions
from typed_python.compiler.type_wrappers.bound_method_wrapper import BoundMethodWrapper

import typed_python.compiler.native_ast as native_ast
import typed_python.compiler

from typed_python import (
    ListOf, Float32, Int8, Int16, Int32, UInt8, UInt16, UInt32, UInt64, TupleOf, Tuple, Dict, OneOf
)

from typed_python import Class, Final, Member, pointerTo, PointerTo

from typed_python.compiler.native_ast import VoidPtr

typeWrapper = lambda t: typed_python.compiler.python_object_representation.typedPythonTypeToTypeWrapper(t)


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
    if not prefix:
        return True
    return s[:len(prefix)] == prefix


def strRangeStartswith(s, prefix, start, end):
    if start > len(s):
        return False
    if start < 0:
        start += len(s)
        if start < 0:
            start = 0
    if end < 0:
        end += len(s)
        if end < 0:
            end = 0
    if not prefix:
        return start <= 0 or end >= start
    if end < start + len(prefix):
        return False
    return s[start:start + len(prefix)] == prefix


def strStartswithTuple(s, prefixtuple):
    for prefix in prefixtuple:
        t = type(prefix)
        if t is not object and t is not str:
            raise TypeError(f"tuple for startswith must only contain str, not {t}")
        if not prefix:
            return True
        if s[:len(prefix)] == prefix:
            return True
    return False


def strRangeStartswithTuple(s, prefixtuple, start, end):
    if start > len(s):
        return False
    if start < 0:
        start += len(s)
        if start < 0:
            start = 0
    if end < 0:
        end += len(s)
        if end < 0:
            end = 0
    for prefix in prefixtuple:
        t = type(prefix)
        if t is not object and t is not str:
            raise TypeError(f"tuple for startswith must only contain str, not {t}")
        if not prefix:
            return start <= 0 or end >= start
        if end < start + len(prefix):
            continue
        if s[start:start + len(prefix)] == prefix:
            return True
    return False


def strEndswith(s, suffix):
    if not suffix:
        return True

    return s[-len(suffix):] == suffix


def strRangeEndswith(s, suffix, start, end):
    if start > len(s):
        return False
    if end > len(s):
        end = len(s)
    if start < 0:
        start += len(s)
        if start < 0:
            start = 0
    if end < 0:
        end += len(s)
        if end < 0:
            end = 0
    if not suffix:
        return start <= 0 or end >= start
    if start > end - len(suffix):
        return False

    return s[end - len(suffix):end] == suffix


def strEndswithTuple(s, suffixtuple):
    for suffix in suffixtuple:
        t = type(suffix)
        if t is not object and t is not str:
            raise TypeError(f"tuple for endswith must only contain str, not {t}")
        if not suffix:
            return True
        if s[-len(suffix):] == suffix:
            return True
    return False


def strRangeEndswithTuple(s, suffixtuple, start, end):
    if start > len(s):
        return False
    if end > len(s):
        end = len(s)
    if start < 0:
        start += len(s)
        if start < 0:
            start = 0
    if end < 0:
        end += len(s)
        if end < 0:
            end = 0
    for suffix in suffixtuple:
        t = type(suffix)
        if t is not object and t is not str:
            raise TypeError(f"tuple for endswith must only contain str, not {t}")
        if not suffix:
            if start <= 0 or end >= start:
                return True
            else:
                continue
        if start > end - len(suffix):
            continue
        if s[end - len(suffix):end] == suffix:
            return True
    return False


IS_38_OR_LOWER = sys.version_info.minor <= 8


def strReplace(s, old, new, maxCount):
    if IS_38_OR_LOWER:
        # versions 3.8 and lower have a bug where b''.replace(b'', b'SOMETHING', 1) returns
        # the empty string.
        if maxCount == 0 or (maxCount >= 0 and len(s) == 0 and len(old) == 0):
            return s
    else:
        if maxCount == 0:
            return s

        if maxCount >= 0 and len(s) == 0 and len(old) == 0:
            return new

    accumulator = ListOf(str)()

    pos = 0
    seen = 0
    inc = 0 if len(old) else 1
    if len(old) == 0:
        accumulator.append('')
        seen += 1

    while True:
        if maxCount >= 0 and seen >= maxCount:
            nextLoc = -1
        else:
            nextLoc = s.find(old, pos)

        if nextLoc >= 0 and nextLoc < len(s):
            accumulator.append(s[pos:nextLoc + inc])

            if len(old):
                pos = nextLoc + len(old)
            else:
                pos += 1

            seen += 1
        else:
            accumulator.append(s[pos:])
            return new.join(accumulator)


def strPartition(x: str, sep):
    if len(sep) == 0:
        raise ValueError("empty separator")

    pos = x.find(sep)
    if pos == -1:
        return Tuple(str, str, str)((x, '', ''))
    return Tuple(str, str, str)((x[0:pos], sep, x[pos+len(sep):]))


def strRpartition(x, sep):
    if len(sep) == 0:
        raise ValueError("empty separator")

    pos = x.rfind(sep)
    if pos == -1:
        return Tuple(str, str, str)(('', '', x))
    return Tuple(str, str, str)((x[0:pos], sep, x[pos+len(sep):]))


def strCenter(x, width, fill):
    if width <= len(x):
        return x

    left = (width - len(x)) // 2
    right = (width - len(x)) - left
    return fill * left + x + fill * right


def strLjust(x, width, fill):
    if width <= len(x):
        return x

    return x + fill * (width - len(x))


def strRjust(x, width, fill):
    if width <= len(x):
        return x

    return fill * (width - len(x)) + x


def strExpandtabs(x, tabsize):
    accumulator = ListOf(str)()

    col = 0  # column mod tabsize, not necessarily actual column
    last = 0
    for i in range(len(x)):
        c = x[i]
        if c == '\t':
            accumulator.append(x[last:i])
            spaces = tabsize - (col % tabsize) if tabsize > 0 else 0
            accumulator.append(' ' * spaces)
            last = i + 1
            col = 0
        elif c == '\n' or c == '\r':
            col = 0
        else:
            col += 1
    accumulator.append(x[last:])
    return ''.join(accumulator)


def strZfill(x, width):
    accumulator = ListOf(str)()

    sign = False
    if len(x):
        c = x[0]
        if c == '+' or c == '-':
            accumulator.append(x[0:1])
            sign = True

    fill = width - len(x)
    if fill > 0:
        accumulator.append('0' * fill)

    accumulator.append(x[1:] if sign else x)

    return ''.join(accumulator)


def strTranslate(x, table):
    accumulator = ListOf(str)()
    for c in x:
        t = c
        try:
            t = table.__getitem__(ord(c))
        except LookupError:
            pass
        if t is not None:
            accumulator.append(t)
    return ''.join(accumulator)


def strMaketransFromDict(xArg) -> Dict(int, OneOf(int, str, None)):
    # The line below is somehow necessary.  Without it, 'isinstance' type inference fails for elements of the dict key.
    if not isinstance(xArg, dict):
        raise TypeError("if you give only one argument to maketrans it must be a dict")

    x = Dict(OneOf(int, str), OneOf(int, str, None))(xArg)

    ret = Dict(int, OneOf(int, str, None))()
    for c in x:
        if isinstance(c, str):
            if len(c) != 1:
                raise ValueError("string keys in translate table must be of length 1")
            ret[ord(c)] = x[c]
        else:
            ret[c] = x[c]
    return ret


def strMaketransFromStr(x: str, y: str, z: OneOf(str, None)) -> Dict(int, OneOf(int, None)):
    if len(x) != len(y):
        raise ValueError("the first two maketrans arguments must have equal length")
    ret = Dict(int, OneOf(int, None))()
    for i in range(len(x)):
        ret[ord(x[i])] = ord(y[i])
    if z is not None:
        for c in z:
            ret[ord(c)] = None
    return ret


class StringWrapper(RefcountedWrapper):
    is_pod = False
    is_empty = False
    is_pass_by_ref = True

    def __init__(self):
        super().__init__(str)

        self.layoutType = native_ast.Type.Struct(element_types=(
            ('refcount', native_ast.Int64),
            ('hash_cache', native_ast.Int32),
            ('pointcount', native_ast.Int32),
            ('bytes_per_codepoint', native_ast.Int32),
            ('data', native_ast.UInt8)
        ), name='StringLayout').pointer()

    def convert_type_call(self, context, typeInst, args, kwargs):
        if len(args) == 0 and not kwargs:
            return context.push(self, lambda x: x.convert_default_initialize())

        if len(args) == 1 and not kwargs:
            return args[0].convert_str_cast()

        if 1 <= len(args) <= 3:
            if len(args) >= 2:
                arg1 = args[1]
            elif 'encoding' in kwargs:
                arg1 = kwargs['encoding']
            else:
                arg1 = None

            if len(args) >= 3:
                arg2 = args[2]
            elif 'errors' in kwargs:
                arg2 = kwargs['errors']
            else:
                arg2 = None

            return context.push(
                str,
                lambda ref: ref.expr.store(
                    runtime_functions.bytes_decode.call(
                        args[0].nonref_expr.cast(VoidPtr),
                        (arg1 if arg1 is not None else context.constant(0)).nonref_expr.cast(VoidPtr),
                        (arg2 if arg2 is not None else context.constant(0)).nonref_expr.cast(VoidPtr),
                    ).cast(self.layoutType)
                )
            )

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

    def _can_convert_from_type(self, otherType, explicit):
        return False

    def convert_bin_op(self, context, left, op, right, inplace):
        if op.matches.Mult and isInteger(right.expr_type.typeRepresentation):
            if left.isConstant and right.isConstant:
                return context.constant(left.constantValue * right.constantValue)

            return context.push(
                str,
                lambda strRef: strRef.expr.store(
                    runtime_functions.string_mult.call(
                        left.nonref_expr.cast(VoidPtr),
                        right.nonref_expr
                    ).cast(self.layoutType)
                )
            )

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
            if expr.isConstant:
                return context.constant(ord(expr.constantValue))

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

    def convert_getitem_unsafe(self, context, expr, item):
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
                expr.ElementPtrIntegers(0, 2).load().cast(native_ast.Int64)
            )
        )

    def has_intiter(self):
        """Does this type support the 'intiter' format?"""
        return True

    def convert_intiter_size(self, context, instance):
        """If this type supports intiter, compute the size of the iterator.

        This function will return a TypedExpression(int) or None if it set an exception."""
        return self.convert_len(context, instance)

    def convert_intiter_value(self, context, instance, valueInstance):
        """If this type supports intiter, compute the value of the iterator.

        This function will return a TypedExpression, or None if it set an exception."""
        return self.convert_getitem(context, instance, valueInstance)

    def convert_len(self, context, expr):
        if expr.constantValue is not None:
            return context.constant(len(expr.constantValue))

        return context.pushPod(int, self.convert_len_native(expr.nonref_expr))

    def constant(self, context, s):
        return typed_python.compiler.typed_expression.TypedExpression(
            context,
            native_ast.Expression.GlobalVariable(
                name='string_constant_' + sha_hash(s).hexdigest,
                type=native_ast.VoidPtr,
                metadata=GlobalVariableMetadata.StringConstant(value=s)
            ).cast(self.layoutType.pointer()),
            self,
            True,
            constantValue=s
        )

    _bool_methods = dict(
        isalpha=runtime_functions.string_isalpha,
        isalnum=runtime_functions.string_isalnum,
        isdecimal=runtime_functions.string_isdecimal,
        isdigit=runtime_functions.string_isdigit,
        isidentifier=runtime_functions.string_isidentifier,
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
        capitalize=runtime_functions.string_capitalize,
        casefold=runtime_functions.string_casefold,
        swapcase=runtime_functions.string_swapcase,
        title=runtime_functions.string_title,
    )

    _find_methods = dict(
        find=runtime_functions.string_find,
        rfind=runtime_functions.string_rfind,
        index=runtime_functions.string_index,
        rindex=runtime_functions.string_rindex,
        count=runtime_functions.string_count,
    )

    _methods = ['split', 'rsplit', 'splitlines', 'join', 'partition', 'rpartition',
                'strip', 'rstrip', 'lstrip', 'startswith', 'endswith', 'replace',
                "translate", "maketrans",
                '__iter__', 'encode', 'center', 'ljust', 'rjust', 'expandtabs', 'splitlines', 'zfill'] \
        + list(_bool_methods) + list(_str_methods) + list(_find_methods)

    def convert_attribute(self, context, instance, attr):
        if attr in self._methods:
            return instance.changeType(BoundMethodWrapper.Make(self, attr))

        return super().convert_attribute(context, instance, attr)

    @Wrapper.unwrapOneOfAndValue
    def convert_method_call(self, context, instance, methodname, args, kwargs):
        if methodname not in self._methods:
            return super().convert_method_call(context, instance, methodname, args, kwargs)

        if methodname == "__iter__" and not args and not kwargs:
            return typeWrapper(StringIterator).convert_type_call(
                context,
                None,
                [],
                dict(pos=context.constant(-1), string=instance)
            )

        if methodname in ['strip', 'lstrip', 'rstrip'] and not kwargs:
            fromLeft = methodname in ['strip', 'lstrip']
            fromRight = methodname in ['strip', 'rstrip']
            if len(args) == 0 or (len(args) == 1 and args[0].expr_type.typeRepresentation == str):
                arg0 = VoidPtr.zero() if len(args) == 0 else args[0].nonref_expr
                return context.push(
                    str,
                    lambda strRef: strRef.expr.store(
                        runtime_functions.string_strip.call(
                            instance.nonref_expr.cast(VoidPtr),
                            native_ast.const_bool_expr(len(args) == 0),
                            arg0.cast(VoidPtr),
                            native_ast.const_bool_expr(fromLeft),
                            native_ast.const_bool_expr(fromRight)
                        ).cast(self.layoutType)
                    )
                )

        if methodname in self._str_methods and not kwargs:
            if len(args) == 0:
                return context.push(
                    str,
                    lambda strRef: strRef.expr.store(
                        self._str_methods[methodname].call(
                            instance.nonref_expr.cast(VoidPtr)
                        ).cast(self.layoutType)
                    )
                )

        if methodname in self._bool_methods and not kwargs:
            if len(args) == 0:
                return context.push(
                    bool,
                    lambda bRef: bRef.expr.store(
                        self._bool_methods[methodname].call(
                            instance.nonref_expr.cast(VoidPtr)
                        )
                    )
                )
        if methodname in self._find_methods and 1 <= len(args) <= 3 and not kwargs:
            arg1 = context.constant(0) if len(args) <= 1 else args[1].nonref_expr
            arg2 = self.convert_len(context, instance) if len(args) <= 2 else args[2].nonref_expr
            return context.push(
                int,
                lambda iRef: iRef.expr.store(
                    self._find_methods[methodname].call(
                        instance.nonref_expr.cast(VoidPtr),
                        args[0].nonref_expr.cast(VoidPtr),
                        arg1,
                        arg2
                    )
                )
            )

        if methodname == "translate" and not kwargs:
            if len(args) == 1:
                return context.call_py_function(strTranslate, (instance, args[0]), {})

        if methodname == 'maketrans' and not kwargs:
            if len(args) == 1:
                return context.call_py_function(strMaketransFromDict, (args[0], ), {})
            if 2 <= len(args) <= 3:
                if len(args) == 3:
                    arg2 = args[2]
                else:
                    arg2 = context.constant(None)
                return context.call_py_function(strMaketransFromStr, (args[0], args[1], arg2), {})

        if methodname in ["startswith", "endswith"] and not kwargs:
            if len(args) >= 1 and len(args) <= 3:
                sw = (methodname == "startswith")
                t = args[0].expr_type
                if len(args) == 1:
                    if t == self:
                        return context.call_py_function(strStartswith if sw else strEndswith,
                                                        (instance, args[0]), {})
                    if t.typeRepresentation == tuple or t is typeWrapper(TupleOf(str)):
                        return context.call_py_function(strStartswithTuple if sw else strEndswithTuple,
                                                        (instance, args[0]), {})
                else:
                    if len(args) == 3:
                        arg1 = args[1]
                        arg2 = args[2]
                    elif len(args) == 2:
                        arg1 = args[1]
                        arg2 = self.convert_len(context, instance)

                    if t == self:
                        return context.call_py_function(strRangeStartswith if sw else strRangeEndswith,
                                                        (instance, args[0], arg1, arg2), {})
                    if t.typeRepresentation == tuple or t is typeWrapper(TupleOf(str)):
                        return context.call_py_function(strRangeStartswithTuple if sw else strRangeEndswithTuple,
                                                        (instance, args[0], arg1, arg2), {})

                return context.pushException(
                    TypeError,
                    f"{'starts' if sw else 'ends'}with first arg must be str or a tuple of str, not {t}"
                )

        if methodname == 'expandtabs' and len(args) == 1 and not kwargs:
            arg0type = args[0].expr_type.typeRepresentation
            if arg0type != int:
                return context.pushException(TypeError, f"an integer is required, not '{arg0type}'")
            return context.call_py_function(strExpandtabs, (instance, args[0]), {})

        if methodname == "replace" and not kwargs:
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

        if methodname == "join" and not kwargs:
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

        if methodname in ['split', 'rsplit'] and not kwargs:
            if len(args) == 0:
                sepPtr = VoidPtr.zero()
                maxCount = native_ast.const_int_expr(-1)
            elif len(args) in [1, 2] and args[0].expr_type.typeRepresentation in [str, type(None)]:
                if args[0].expr_type == typeWrapper(None):
                    sepPtr = VoidPtr.zero()
                else:
                    sepPtr = args[0].nonref_expr.cast(VoidPtr)
                    sepLen = args[0].convert_len()
                    if sepLen is None:
                        return None
                    with context.ifelse(sepLen.nonref_expr.eq(0)) as (ifTrue, ifFalse):
                        with ifTrue:
                            context.pushException(ValueError, "empty separator")

                if len(args) == 2:
                    maxCount = args[1].toInt64()
                    if maxCount is None:
                        return None
                else:
                    maxCount = native_ast.const_int_expr(-1)
            else:
                maxCount = None

            if maxCount is not None:
                fn = runtime_functions.string_split if methodname == 'split' else runtime_functions.string_rsplit
                return context.push(
                    TypedListMasqueradingAsList(ListOf(str)),
                    lambda out: out.expr.store(
                        fn.call(
                            instance.nonref_expr.cast(VoidPtr),
                            sepPtr,
                            maxCount
                        ).cast(out.expr_type.getNativeLayoutType())
                    )
                )
        if methodname == 'splitlines' and not kwargs:
            if len(args) == 0:
                arg0 = context.constant(False)
            elif len(args) == 1:
                arg0 = args[0].toBool()
                if arg0 is None:
                    return None

            return context.push(
                TypedListMasqueradingAsList(ListOf(str)),
                lambda out: out.expr.store(
                    runtime_functions.string_splitlines.call(
                        instance.nonref_expr.cast(VoidPtr),
                        arg0
                    ).cast(out.expr_type.getNativeLayoutType())
                )
            )
        if methodname == 'encode':
            if 0 <= len(args) <= 2:
                if len(args) >= 1:
                    arg0 = args[0]
                elif 'encoding' in kwargs:
                    arg0 = kwargs['encoding']
                else:
                    arg0 = None

                if len(args) >= 2:
                    arg1 = args[1]
                elif 'errors' in kwargs:
                    arg1 = kwargs['errors']
                else:
                    arg1 = None

                return context.push(
                    bytes,
                    lambda ref: ref.expr.store(
                        runtime_functions.str_encode.call(
                            instance.nonref_expr.cast(VoidPtr),
                            (arg0 if arg0 is not None else context.constant(0)).nonref_expr.cast(VoidPtr),
                            (arg1 if arg1 is not None else context.constant(0)).nonref_expr.cast(VoidPtr),
                        ).cast(typeWrapper(bytes).layoutType)
                    )
                )

        if methodname in ['partition', 'rpartition'] and len(args) == 1 and not kwargs:
            arg0type = args[0].expr_type.typeRepresentation
            if arg0type != str:
                context.pushException(TypeError, f"must be str, not '{arg0type}'")
            py_f = strPartition if methodname == 'partition' else strRpartition
            return context.call_py_function(py_f, (instance, args[0]), {})

        if methodname in ['center', 'ljust', 'rjust']:
            if len(args) in [1, 2]:
                arg0 = args[0].toInt64()
                if arg0 is None:
                    return None

                if len(args) == 2:
                    arg1 = args[1]
                    arg1type = arg1.expr_type.typeRepresentation
                    if arg1type != str:
                        context.pushException(TypeError, f"{methodname}() the fill character must be a unicode character, not {arg1type}")
                    arg1len = arg1.convert_len()
                    if arg1len is None:
                        return None
                    with context.ifelse(arg1len.nonref_expr.eq(1)) as (ifTrue, ifFalse):
                        with ifFalse:
                            context.pushException(
                                TypeError,
                                f"{methodname}() the fill character must be exactly one character long"
                            )
                else:
                    arg1 = context.constant(' ')

            py_f = strCenter if methodname == 'center' else \
                strLjust if methodname == 'ljust' else \
                strRjust if methodname == 'rjust' else None
            return context.call_py_function(py_f, (instance, arg0, arg1), {})

        if methodname == 'zfill' and len(args) == 1 and not kwargs:
            arg0 = args[0].toInt64()
            if arg0 is None:
                return None
            return context.call_py_function(strZfill, (instance, arg0), {})

        return super().convert_method_call(context, instance, methodname, args, kwargs)

    def _can_convert_to_type(self, targetType, conversionLevel):
        if not conversionLevel.isNewOrHigher():
            return False

        return targetType.typeRepresentation in (
            Float32, Int8, Int16, Int32, UInt8, UInt16, UInt32, UInt64, float, int, bool, str
        )

    def convert_to_type_with_target(self, context, instance, targetVal, conversionLevel, mayThrowOnFailure=False):
        if not conversionLevel.isNewOrHigher():
            return super().convert_to_type_with_target(context, instance, targetVal, conversionLevel, mayThrowOnFailure)

        if targetVal.expr_type.typeRepresentation is bool:
            res = context.pushPod(bool, self.convert_len_native(instance.nonref_expr).neq(0))
            context.pushEffect(
                targetVal.expr.store(res.nonref_expr)
            )
            return context.constant(True)

        if targetVal.expr_type.typeRepresentation is int:
            res = context.pushPod(
                int,
                runtime_functions.str_to_int64.call(instance.nonref_expr.cast(VoidPtr))
            )
            context.pushEffect(
                targetVal.expr.store(res.nonref_expr)
            )
            return context.constant(True)

        if targetVal.expr_type.typeRepresentation is float:
            res = context.pushPod(
                float,
                runtime_functions.str_to_float64.call(instance.nonref_expr.cast(VoidPtr))
            )
            context.pushEffect(
                targetVal.expr.store(res.nonref_expr)
            )
            return context.constant(True)

        # for the nonstandard int types, convert to 'int' first.
        if targetVal.expr_type.typeRepresentation in (Int8, UInt8, Int16, UInt16, Int32, UInt32, UInt64):
            outInt = context.allocateUninitializedSlot(int)
            isInitialized = instance.convert_to_type_with_target(outInt, conversionLevel)
            with context.ifelse(isInitialized.nonref_expr) as (ifTrue, ifFalse):
                with ifTrue:
                    outInt.convert_to_type_with_target(targetVal, conversionLevel)
            return isInitialized

        # for float32, convert to float first, then downcast
        if targetVal.expr_type.typeRepresentation is Float32:
            outFloat = context.allocateUninitializedSlot(float)
            isInitialized = instance.convert_to_type_with_target(outFloat, conversionLevel)
            with context.ifelse(isInitialized.nonref_expr) as (ifTrue, ifFalse):
                with ifTrue:
                    outFloat.convert_to_type_with_target(targetVal, conversionLevel)
            return isInitialized

        return super().convert_to_type_with_target(context, instance, targetVal, conversionLevel, mayThrowOnFailure)

    def get_iteration_expressions(self, context, expr):
        if expr.isConstant:
            return [context.constant(expr.constantValue[i]) for i in range(len(expr.constantValue))]
        else:
            return None


class StringIterator(Class, Final):
    pos = Member(int)
    string = Member(str)
    value = Member(str)

    def __fastnext__(self):
        self.pos = self.pos + 1

        if self.pos < len(self.string):
            self.value = self.string[self.pos]
            return pointerTo(self).value
        else:
            return PointerTo(str)()


class StringMaketransWrapper(Wrapper):
    is_pod = True
    is_empty = False
    is_pass_by_ref = False

    def __init__(self):
        super().__init__(str.maketrans)

    def getNativeLayoutType(self):
        return native_ast.Type.Void()

    def convert_call(self, context, expr, args, kwargs):
        if not kwargs:
            static_str_instance = StringWrapper().constant(context, '')
            if len(args) == 1:
                return static_str_instance.convert_method_call("maketrans", (args[0],), {})
            elif len(args) == 2:
                return static_str_instance.convert_method_call("maketrans", (args[0], args[1]), {})
            elif len(args) == 3:
                return static_str_instance.convert_method_call("maketrans", (args[0], args[1], args[2]), {})

        return super().convert_call(context, expr, args, kwargs)
