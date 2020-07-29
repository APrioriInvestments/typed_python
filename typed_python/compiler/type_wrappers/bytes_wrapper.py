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

from typed_python import UInt8, Int32, ListOf, Tuple
from typed_python.type_promotion import isInteger

import typed_python.compiler.native_ast as native_ast
import typed_python.compiler

from typed_python.compiler.native_ast import VoidPtr


typeWrapper = lambda t: typed_python.compiler.python_object_representation.typedPythonTypeToTypeWrapper(t)


def bytesJoinIterable(sep, iterable):
    """Converts the iterable container to list of bytes objects and call sep.join(iterable).

    If any of the values in the container is not bytes type, an exception is thrown.

    :param sep: string to separate the items
    :param iterable: iterable container with strings only
    :return: string with joined values
    """
    items = ListOf(bytes)()

    for item in iterable:
        if isinstance(item, bytes):
            items.append(item)
        else:
            raise TypeError("expected str instance")
    return sep.join(items)


def bytes_replace(x, old, new, maxCount):
    if maxCount == 0 or (maxCount >= 0 and len(x) == 0 and len(old) == 0):
        return x

    accumulator = ListOf(bytes)()

    pos = 0
    seen = 0
    inc = 0 if len(old) else 1
    if len(old) == 0:
        accumulator.append(b'')
        seen += 1

    while True:
        if maxCount >= 0 and seen >= maxCount:
            nextLoc = -1
        else:
            nextLoc = x.find(old, pos)

        if nextLoc >= 0 and nextLoc < len(x):
            accumulator.append(x[pos:nextLoc + inc])

            if len(old):
                pos = nextLoc + len(old)
            else:
                pos += 1

            seen += 1
        else:
            accumulator.append(x[pos:])
            return new.join(accumulator)


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


def bytes_startswith(x, prefix):
    if len(x) < len(prefix):
        return False
    index = 0
    for i in prefix:
        if x[index] != i:
            return False
        index += 1
    return True


def bytes_endswith(x, suffix):
    index = len(x) - len(suffix)
    if index < 0:
        return False
    for i in suffix:
        if x[index] != i:
            return False
        index += 1
    return True


# sub is a bytes-like object
def bytes_count(x, sub, start, end):
    if start < 0:
        start += len(x)
    if start < 0:
        start = 0
    if end < 0:
        end += len(x)
    if end < 0:
        end = 0
    if end > len(x):
        end = len(x)

    len_sub = len(sub)
    if len_sub == 0:
        if start > len(x):
            return 0
        count = end - start + 1
        if count < 0:
            count = 0
        return count

    count = 0
    index = start
    while index < end - len_sub + 1:
        subindex = 0
        while subindex < len_sub:
            if x[index+subindex] != sub[subindex]:
                break
            subindex += 1
            if subindex == len_sub:
                count += 1
                index += len_sub - 1
        index += 1
    return count


# sub is an integer
def bytes_count_single(x, sub, start, end):
    if sub < 0 or sub > 255:
        raise ValueError("byte must be in range(0, 256)")
    if start < 0:
        start += len(x)
    if start < 0:
        start = 0
    if end < 0:
        end += len(x)
    if end < 0:
        end = 0
    if end > len(x):
        end = len(x)

    count = 0
    index = start
    while index < end:
        if x[index] == sub:
            count += 1
        index += 1
    return count


# sub is a bytes-like object
def bytes_find(x, sub, start, end):
    if start < 0:
        start += len(x)
    if start < 0:
        start = 0
    if end < 0:
        end += len(x)
    if end < 0:
        end = 0
    if end > len(x):
        end = len(x)

    len_sub = len(sub)
    if len_sub == 0:
        if start > len(x) or start > end:
            return -1
        return start

    index = start
    while index < end - len_sub + 1:
        subindex = 0
        while subindex < len_sub:
            if x[index+subindex] != sub[subindex]:
                break
            subindex += 1
            if subindex == len_sub:
                return index
        index += 1
    return -1


# sub is an integer
def bytes_find_single(x, sub, start, end):
    if sub < 0 or sub > 255:
        raise ValueError("byte must be in range(0, 256)")
    if start < 0:
        start += len(x)
    if start < 0:
        start = 0
    if end < 0:
        end += len(x)
    if end < 0:
        end = 0
    if end > len(x):
        end = len(x)

    index = start
    while index < end:
        if x[index] == sub:
            return index
        index += 1
    return -1


# sub is a bytes-like object
def bytes_rfind(x, sub, start, end):
    if start < 0:
        start += len(x)
    if start < 0:
        start = 0
    if end < 0:
        end += len(x)
    if end < 0:
        end = 0
    if end > len(x):
        end = len(x)

    len_sub = len(sub)
    if len_sub == 0:
        if start > len(x) or start > end:
            return -1
        return end

    index = end - len_sub
    while index >= start:
        subindex = 0
        while subindex < len_sub:
            if x[index+subindex] != sub[subindex]:
                break
            subindex += 1
            if subindex == len_sub:
                return index
        index -= 1
    return -1


# sub is an integer
def bytes_rfind_single(x, sub, start, end):
    if sub < 0 or sub > 255:
        raise ValueError("byte must be in range(0, 256)")
    if start < 0:
        start += len(x)
    if start < 0:
        start = 0
    if end < 0:
        end += len(x)
    if end < 0:
        end = 0
    if end > len(x):
        end = len(x)

    index = end - 1
    while index >= start:
        if x[index] == sub:
            return index
        index -= 1
    return -1


def bytes_index(x, sub, start, end):
    ret = bytes_find(x, sub, start, end)
    if ret == -1:
        raise ValueError("subsection not found")
    return ret


def bytes_index_single(x, sub, start, end):
    ret = bytes_find_single(x, sub, start, end)
    if ret == -1:
        raise ValueError("subsection not found")
    return ret


def bytes_rindex(x, sub, start, end):
    ret = bytes_rfind(x, sub, start, end)
    if ret == -1:
        raise ValueError("subsection not found")
    return ret


def bytes_rindex_single(x, sub, start, end):
    ret = bytes_rfind_single(x, sub, start, end)
    if ret == -1:
        raise ValueError("subsection not found")
    return ret


def bytes_partition(x, sep):
    if len(sep) == 0:
        raise ValueError("empty separator")

    pos = x.find(sep)
    if pos == -1:
        return Tuple(bytes, bytes, bytes)((x, b'', b''))
    return Tuple(bytes, bytes, bytes)((x[0:pos], sep, x[pos+len(sep):]))


def bytes_rpartition(x, sep):
    if len(sep) == 0:
        raise ValueError("empty separator")

    pos = x.rfind(sep)
    if pos == -1:
        return Tuple(bytes, bytes, bytes)((b'', b'', x))
    return Tuple(bytes, bytes, bytes)((x[0:pos], sep, x[pos+len(sep):]))


def bytes_center(x, width, fill):
    if width <= len(x):
        return x

    left = (width - len(x)) // 2
    right = (width - len(x)) - left
    return fill * left + x + fill * right


def bytes_ljust(x, width, fill):
    if width <= len(x):
        return x

    return x + fill * (width - len(x))


def bytes_rjust(x, width, fill):
    if width <= len(x):
        return x

    return fill * (width - len(x)) + x


def bytes_expandtabs(x, tabsize):
    accumulator = ListOf(bytes)()

    col = 0  # column mod tabsize, not necessarily actual column
    last = 0
    for i in range(len(x)):
        c = x[i]
        if c == ord('\t'):
            accumulator.append(x[last:i])
            spaces = tabsize - (col % tabsize) if tabsize > 0 else 0
            accumulator.append(b' ' * spaces)
            last = i + 1
            col = 0
        elif c == ord('\n') or c == ord('\r'):
            col = 0
        else:
            col += 1
    accumulator.append(x[last:])
    return b''.join(accumulator)


def bytes_zfill(x, width):
    accumulator = ListOf(bytes)()

    sign = False
    if len(x):
        c = x[0]
        if c == ord('+') or c == ord('-'):
            accumulator.append(x[0:1])
            sign = True

    fill = width - len(x)
    if fill > 0:
        accumulator.append(b'0' * fill)

    accumulator.append(x[1:] if sign else x)

    return b''.join(accumulator)


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
        if op.matches.Mult and isInteger(right.expr_type.typeRepresentation):
            if left.isConstant and right.isConstant:
                return context.constant(left.constantValue * right.constantValue)

            return context.push(
                bytes,
                lambda bytesRef: bytesRef.expr.store(
                    runtime_functions.bytes_mult.call(
                        left.nonref_expr.cast(VoidPtr),
                        right.nonref_expr
                    ).cast(self.layoutType)
                )
            )

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

            if op.matches.In:
                if left.isConstant and right.isConstant:
                    return context.constant(left.constantValue in right.constantValue)

                find_converted = right.convert_method_call("find", (left,), {})
                if find_converted is None:
                    return None
                return find_converted >= 0

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

        if lower is not None and upper is None:
            upper = expr.convert_len()

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

    # these map to py functions
    _bool_methods = dict(
        isalnum=bytes_isalnum,
        isalpha=bytes_isalpha,
        isdigit=bytes_isdigit,
        islower=bytes_islower,
        isspace=bytes_isspace,
        istitle=bytes_istitle,
        isupper=bytes_isupper
    )

    # these map to py functions
    _find_methods = dict(
        count=(bytes_count, bytes_count_single),
        find=(bytes_find, bytes_find_single),
        rfind=(bytes_rfind, bytes_rfind_single),
        index=(bytes_index, bytes_index_single),
        rindex=(bytes_rindex, bytes_rindex_single),
    )

    # these map to c++ functions
    _bytes_methods = dict(
        lower=runtime_functions.bytes_lower,
        upper=runtime_functions.bytes_upper,
        capitalize=runtime_functions.bytes_capitalize,
        swapcase=runtime_functions.bytes_swapcase,
        title=runtime_functions.bytes_title,
    )

    def convert_attribute(self, context, instance, attr):
        if (
                attr in ('decode', 'translate', 'split', 'rsplit', 'join', 'partition', 'rpartition',
                         'strip', 'rstrip', 'lstrip', 'startswith', 'endswith', 'replace',
                         '__iter__', 'center', 'ljust', 'rjust', 'expandtabs', 'splitlines', 'zfill')
                or attr in self._bytes_methods
                or attr in self._find_methods
                or attr in self._bool_methods
        ):
            return instance.changeType(BoundMethodWrapper.Make(self, attr))

        return super().convert_attribute(context, instance, attr)

    def convert_method_call(self, context, instance, methodname, args, kwargs):
        if methodname == '__iter__' and not args and not kwargs:
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

        if methodname in self._bytes_methods and not args and not kwargs:
            return context.push(
                bytes,
                lambda ref: ref.expr.store(
                    self._bytes_methods[methodname].call(
                        instance.nonref_expr.cast(VoidPtr)
                    ).cast(self.layoutType)
                )
            )

        if methodname in ['strip', 'lstrip', 'rstrip']:
            fromLeft = methodname in ['strip', 'lstrip']
            fromRight = methodname in ['strip', 'rstrip']
            if len(args) == 0 and not kwargs:
                return context.push(
                    bytes,
                    lambda ref: ref.expr.store(
                        runtime_functions.bytes_strip.call(
                            instance.nonref_expr.cast(VoidPtr),
                            native_ast.const_bool_expr(fromLeft),
                            native_ast.const_bool_expr(fromRight)
                        ).cast(self.layoutType)
                    )
                )
            elif len(args) == 1 and not kwargs:
                return context.push(
                    bytes,
                    lambda ref: ref.expr.store(
                        runtime_functions.bytes_strip2.call(
                            instance.nonref_expr.cast(VoidPtr),
                            args[0].nonref_expr.cast(VoidPtr),
                            native_ast.const_bool_expr(fromLeft),
                            native_ast.const_bool_expr(fromRight)
                        ).cast(self.layoutType)
                    )
                )

        if methodname == 'startswith' and len(args) == 1 and not kwargs:
            return context.call_py_function(bytes_startswith, (instance, args[0]), {})
        if methodname == 'endswith' and len(args) == 1 and not kwargs:
            return context.call_py_function(bytes_endswith, (instance, args[0]), {})
        if methodname == 'expandtabs' and len(args) == 1 and not kwargs:
            arg0type = args[0].expr_type.typeRepresentation
            if arg0type != int:
                return context.pushException(TypeError, f"an integer is required, not '{arg0type}'")
            return context.call_py_function(bytes_expandtabs, (instance, args[0]), {})

        if methodname in self._find_methods and 1 <= len(args) <= 3 and not kwargs:
            if len(args) == 3:
                start = args[1]
                end = args[2]
            elif len(args) == 2:
                start = args[1]
                end = self.convert_len(context, instance)
            elif len(args) == 1:
                start = context.constant(0)
                end = self.convert_len(context, instance)

            if isInteger(args[0].expr_type.typeRepresentation):
                py_f = self._find_methods[methodname][1]
            else:
                py_f = self._find_methods[methodname][0]
            return context.call_py_function(py_f, (instance, args[0], start, end), {})

        # if methodname == 'replace' and not kwargs:
        #     if len(args) in [2, 3]:
        #         return context.push(
        #             bytes,
        #             lambda bytesRef: bytesRef.expr.store(
        #                 runtime_functions.bytes_replace.call(
        #                     instance.nonref_expr.cast(VoidPtr),
        #                     args[0].nonref_expr.cast(VoidPtr),
        #                     args[1].nonref_expr.cast(VoidPtr),
        #                     args[2].nonref_expr if len(args) == 3 else native_ast.const_int_expr(-1)
        #                 ).cast(self.layoutType)
        #             )
        #         )
        if methodname == 'replace':
            if len(args) in [2, 3]:
                for i in [0, 1]:
                    if args[i].expr_type != self:
                        context.pushException(
                            TypeError,
                            f"replace() argument {i + 1} must be bytes"
                        )
                        return

                if len(args) == 3 and args[2].expr_type.typeRepresentation != int:
                    context.pushException(
                        TypeError,
                        f"replace() argument 3 must be int, not {args[2].expr_type.typeRepresentation}"
                    )
                    return

                if len(args) == 2:
                    return context.call_py_function(bytes_replace, (instance, args[0], args[1], context.constant(-1)), {})
                else:
                    return context.call_py_function(bytes_replace, (instance, args[0], args[1], args[2]), {})

        if methodname == 'join' and not kwargs:
            if len(args) == 1:
                # we need to pass the list of bytes objects
                separator = instance
                itemsToJoin = args[0]

                if itemsToJoin.expr_type.typeRepresentation is ListOf(bytes):
                    return context.push(
                        bytes,
                        lambda out: runtime_functions.bytes_join.call(
                            out.expr.cast(VoidPtr),
                            separator.nonref_expr.cast(VoidPtr),
                            itemsToJoin.nonref_expr.cast(VoidPtr)
                        )
                    )
                else:
                    return context.call_py_function(bytesJoinIterable, (separator, itemsToJoin), {})

        if methodname in ['split', 'rsplit'] and not kwargs:
            if len(args) == 0:
                sepPtr = VoidPtr.zero()
                maxCount = native_ast.const_int_expr(-1)
            elif len(args) in [1, 2] and args[0].expr_type.typeRepresentation == bytes:
                sepPtr = args[0].nonref_expr.cast(VoidPtr)
                sepLen = args[0].convert_len()
                if sepLen is None:
                    return None
                with context.ifelse(sepLen.nonref_expr.eq(0)) as (ifTrue, ifFalse):
                    with ifTrue:
                        context.pushException(ValueError, "empty separator")

                if len(args) == 2:
                    maxCount = args[1].convert_to_type(int)
                    if maxCount is None:
                        return None
                else:
                    maxCount = native_ast.const_int_expr(-1)
            else:
                maxCount = None

            if maxCount is not None:
                fn = runtime_functions.bytes_split if methodname == 'split' else runtime_functions.bytes_rsplit
                return context.push(
                    TypedListMasqueradingAsList(ListOf(bytes)),
                    lambda outBytes: outBytes.expr.store(
                        fn.call(
                            instance.nonref_expr.cast(VoidPtr),
                            sepPtr,
                            maxCount
                        ).cast(outBytes.expr_type.getNativeLayoutType())
                    )
                )

        if methodname == 'splitlines' and not kwargs:
            if len(args) == 0:
                arg0 = context.constant(False)
            elif len(args) == 1:
                arg0 = args[0].convert_to_type(bool)
                if arg0 is None:
                    return None

            return context.push(
                TypedListMasqueradingAsList(ListOf(bytes)),
                lambda out: out.expr.store(
                    runtime_functions.bytes_splitlines.call(
                        instance.nonref_expr.cast(VoidPtr),
                        arg0
                    ).cast(out.expr_type.getNativeLayoutType())
                )
            )

        if methodname == 'decode' and not kwargs:
            if len(args) in [0, 1, 2]:
                return context.push(
                    str,
                    lambda ref: ref.expr.store(
                        runtime_functions.bytes_decode.call(
                            instance.nonref_expr.cast(VoidPtr),
                            (args[0] if len(args) >= 1 else context.constant(0)).nonref_expr.cast(VoidPtr),
                            (args[1] if len(args) >= 2 else context.constant(0)).nonref_expr.cast(VoidPtr),
                        ).cast(typeWrapper(str).layoutType)
                    )
                )

        if methodname == 'translate':
            if len(args) in [1, 2]:
                arg0isNone = args[0].expr_type == typeWrapper(None)
                arg0 = args[0] if not arg0isNone else context.constant(0)
                if 'delete' in kwargs and len(args) == 1:
                    arg1 = kwargs['delete']
                else:
                    arg1 = args[1] if len(args) >= 2 else context.constant(0)

                if not arg0isNone:
                    arg0type = arg0.expr_type.typeRepresentation
                    if arg0type != bytes:
                        context.pushException(TypeError, f"a bytes-like object is required, not '{arg0type}'")
                    arg0len = arg0.convert_len()
                    if arg0len is None:
                        return None
                    with context.ifelse(arg0len.nonref_expr.eq(256)) as (ifTrue, ifFalse):
                        with ifFalse:
                            context.pushException(ValueError, "translation table must be 256 characters long")

                return context.push(
                    bytes,
                    lambda ref: ref.expr.store(
                        runtime_functions.bytes_translate.call(
                            instance.nonref_expr.cast(VoidPtr),
                            arg0.nonref_expr.cast(VoidPtr),
                            arg1.nonref_expr.cast(VoidPtr),
                        ).cast(self.layoutType)
                    )
                )

        if methodname in ['partition', 'rpartition'] and len(args) == 1 and not kwargs:
            arg0type = args[0].expr_type.typeRepresentation
            if arg0type != bytes:
                context.pushException(TypeError, f"a bytes-like object is required, not '{arg0type}'")
            py_f = bytes_partition if methodname == 'partition' else bytes_rpartition
            return context.call_py_function(py_f, (instance, args[0]), {})

        if methodname in ['center', 'ljust', 'rjust']:
            if len(args) in [1, 2]:
                arg0 = args[0].convert_to_type(int)
                if arg0 is None:
                    return None

                if len(args) == 2:
                    arg1 = args[1]
                    arg1type = arg1.expr_type.typeRepresentation
                    if arg1type != bytes:
                        context.pushException(TypeError, f"{methodname}() argument 2 must be a byte string of length 1, not '{arg1type}'")
                    arg1len = arg1.convert_len()
                    if arg1len is None:
                        return None
                    with context.ifelse(arg1len.nonref_expr.eq(1)) as (ifTrue, ifFalse):
                        with ifFalse:
                            context.pushException(
                                TypeError,
                                f"{methodname}() argument 2 must be a byte string of length 1, not '{arg1type}'"
                            )
                else:
                    arg1 = context.constant(b' ')

            py_f = bytes_center if methodname == 'center' else \
                bytes_ljust if methodname == 'ljust' else \
                bytes_rjust if methodname == 'rjust' else None
            return context.call_py_function(py_f, (instance, arg0, arg1), {})

        if methodname == 'zfill' and len(args) == 1 and not kwargs:
            arg0 = args[0].convert_to_type(int)
            if arg0 is None:
                return None
            return context.call_py_function(bytes_zfill, (instance, arg0), {})

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
