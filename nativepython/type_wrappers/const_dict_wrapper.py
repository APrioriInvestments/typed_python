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

from nativepython.type_wrappers.refcounted_wrapper import RefcountedWrapper
import nativepython.type_wrappers.runtime_functions as runtime_functions
from nativepython.type_wrappers.bound_compiled_method_wrapper import BoundCompiledMethodWrapper
from nativepython.type_wrappers.util import min

from typed_python import NoneType

import nativepython.native_ast as native_ast
import nativepython

typeWrapper = lambda t: nativepython.python_object_representation.typedPythonTypeToTypeWrapper(t)


def const_dict_eq(l, r):
    if len(l) != len(r):
        return False

    for i in range(len(l)):
        if l.get_key_by_index_unsafe(i) != r.get_key_by_index_unsafe(i):
            return False

        if l.get_value_by_index_unsafe(i) != r.get_value_by_index_unsafe(i):
            return False

    return True


def const_dict_neq(l, r):
    return not const_dict_eq(l, r)


def const_dict_lt(left, right):
    """Compare two 'ConstDict' instances by comparing their individual elements."""
    for i in range(min(len(left), len(right))):
        if left.get_key_by_index_unsafe(i) > right.get_key_by_index_unsafe(i):
            return False

        if left.get_key_by_index_unsafe(i) < right.get_key_by_index_unsafe(i):
            return True

        if left.get_value_by_index_unsafe(i) > right.get_value_by_index_unsafe(i):
            return False

        if left.get_value_by_index_unsafe(i) < right.get_value_by_index_unsafe(i):
            return True

    return len(left) < len(right)


def const_dict_lte(left, right):
    """Compare two 'ConstDict' instances by comparing their individual elements."""
    for i in range(min(len(left), len(right))):
        if left.get_key_by_index_unsafe(i) > right.get_key_by_index_unsafe(i):
            return False

        if left.get_key_by_index_unsafe(i) < right.get_key_by_index_unsafe(i):
            return True

        if left.get_value_by_index_unsafe(i) > right.get_value_by_index_unsafe(i):
            return False

        if left.get_value_by_index_unsafe(i) < right.get_value_by_index_unsafe(i):
            return True

    return len(left) <= len(right)


def const_dict_gt(left, right):
    return not const_dict_lte(left, right)


def const_dict_gte(left, right):
    return not const_dict_lt(left, right)


def const_dict_getitem(constDict, key):
    # perform a binary search
    lowIx = 0
    highIx = len(constDict)

    while lowIx < highIx:
        mid = (lowIx + highIx) >> 1

        keyAtVal = constDict.get_key_by_index_unsafe(mid)

        if keyAtVal < key:
            lowIx = mid + 1
        elif key < keyAtVal:
            highIx = mid
        else:
            return constDict.get_value_by_index_unsafe(mid)

    raise Exception("Key doesn't exist")


def const_dict_contains(constDict, key):
    # perform a binary search
    lowIx = 0
    highIx = len(constDict)

    while lowIx < highIx:
        mid = (lowIx + highIx) >> 1

        keyAtVal = constDict.get_key_by_index_unsafe(mid)

        if keyAtVal < key:
            lowIx = mid + 1
        elif key < keyAtVal:
            highIx = mid
        else:
            return True

    return False


def const_dict_contains_not(constDict, key):
    return False if const_dict_contains(constDict, key) else True


class ConstDictWrapper(RefcountedWrapper):
    is_pod = False
    is_empty = False
    is_pass_by_ref = True

    def __init__(self, t):
        assert hasattr(t, '__typed_python_category__')
        super().__init__(t)

        self.keyType = typeWrapper(t.KeyType)
        self.valueType = typeWrapper(t.ValueType)

        self.kvBytecount = self.keyType.getBytecount() + self.valueType.getBytecount()
        self.keyBytecount = self.keyType.getBytecount()

        self.layoutType = native_ast.Type.Struct(element_types=(
            ('refcount', native_ast.Int64),
            ('hash_cache', native_ast.Int32),
            ('count', native_ast.Int32),
            ('subpointers', native_ast.Int32),
            ('data', native_ast.UInt8)
        ), name='TupleOfLayout').pointer()

    def getNativeLayoutType(self):
        return self.layoutType

    def convert_attribute(self, context, instance, attr):
        if attr in ("get_key_by_index_unsafe", "get_value_by_index_unsafe"):
            return instance.changeType(BoundCompiledMethodWrapper(self, attr))

        return super().convert_attribute(context, instance, attr)

    def convert_method_call(self, context, instance, methodname, args, kwargs):
        if kwargs:
            return super().convert_method_call(context, instance, methodname, args, kwargs)

        if methodname == "get_key_by_index_unsafe":
            if len(args) == 1:
                ix = args[0].toInt64()
                if ix is None:
                    return

                return self.convert_getkey_by_index_unsafe(context, instance, ix)

        if methodname == "get_value_by_index_unsafe":
            if len(args) == 1:
                ix = args[0].toInt64()
                if ix is None:
                    return

                return self.convert_getvalue_by_index_unsafe(context, instance, ix)

        return super().convert_method_call(context, instance, methodname, args, kwargs)

    def on_refcount_zero(self, context, instance):
        assert instance.isReference

        if self.keyType.is_pod and self.valueType.is_pod:
            return runtime_functions.free.call(instance.nonref_expr.cast(native_ast.UInt8Ptr))
        else:
            return (
                context.converter.defineNativeFunction(
                    "destructor_" + str(self.typeRepresentation),
                    ('destructor', self),
                    [self],
                    typeWrapper(NoneType),
                    self.generateNativeDestructorFunction
                )
                .call(instance)
            )

    def generateNativeDestructorFunction(self, context, out, inst):
        with context.loop(inst.convert_len()) as i:
            self.convert_getkey_by_index_unsafe(context, inst, i).convert_destroy()
            self.convert_getvalue_by_index_unsafe(context, inst, i).convert_destroy()

        context.pushEffect(
            runtime_functions.free.call(inst.nonref_expr.cast(native_ast.UInt8Ptr))
        )

    def convert_getkey_by_index_unsafe(self, context, expr, item):
        return context.pushReference(
            self.keyType,
            expr.nonref_expr.ElementPtrIntegers(0, 4).elemPtr(
                item.nonref_expr.mul(native_ast.const_int_expr(self.kvBytecount))
            ).cast(self.keyType.getNativeLayoutType().pointer())
        )

    def convert_getvalue_by_index_unsafe(self, context, expr, item):
        return context.pushReference(
            self.valueType,
            expr.nonref_expr.ElementPtrIntegers(0, 4).elemPtr(
                item.nonref_expr.mul(native_ast.const_int_expr(self.kvBytecount))
                .add(native_ast.const_int_expr(self.keyBytecount))
            ).cast(self.valueType.getNativeLayoutType().pointer())
        )

    def convert_bin_op(self, context, left, op, right):
        if right.expr_type == left.expr_type:
            if op.matches.Eq:
                return context.call_py_function(const_dict_eq, (left, right), {})
            if op.matches.NotEq:
                return context.call_py_function(const_dict_neq, (left, right), {})
            if op.matches.Lt:
                return context.call_py_function(const_dict_lt, (left, right), {})
            if op.matches.LtE:
                return context.call_py_function(const_dict_lte, (left, right), {})
            if op.matches.Gt:
                return context.call_py_function(const_dict_gt, (left, right), {})
            if op.matches.GtE:
                return context.call_py_function(const_dict_gte, (left, right), {})

        return super().convert_bin_op(context, left, op, right)

    def convert_bin_op_reverse(self, context, left, op, right):
        if op.matches.In or op.matches.NotIn:
            right = right.convert_to_type(self.keyType)
            if right is None:
                return None

            return context.call_py_function(
                const_dict_contains if op.matches.In else const_dict_contains_not,
                (left, right),
                {}
            )

        return super().convert_bin_op(context, left, op, right)

    def convert_getitem(self, context, instance, item):
        item = item.convert_to_type(self.keyType)
        if item is None:
            return None

        return context.call_py_function(const_dict_getitem, (instance, item), {})

    def convert_len_native(self, expr):
        return native_ast.Expression.Branch(
            cond=expr,
            false=native_ast.const_int_expr(0),
            true=expr.ElementPtrIntegers(0, 2).load().cast(native_ast.Int64)
        )

    def convert_len(self, context, expr):
        return context.pushPod(int, self.convert_len_native(expr.nonref_expr))
