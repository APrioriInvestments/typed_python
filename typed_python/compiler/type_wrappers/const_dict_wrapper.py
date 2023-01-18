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
import typed_python.compiler.type_wrappers.runtime_functions as runtime_functions
from typed_python.compiler.conversion_level import ConversionLevel
from typed_python.compiler.type_wrappers.bound_method_wrapper import BoundMethodWrapper
from typed_python.compiler.type_wrappers.util import min
from typed_python.compiler.typed_expression import TypedExpression

from typed_python import Tuple, TypeFunction, Held, Member, Final, Class, ConstDict, PointerTo

import typed_python.compiler.native_ast as native_ast
import typed_python.compiler

typeWrapper = lambda t: typed_python.compiler.python_object_representation.typedPythonTypeToTypeWrapper(t)


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

    raise KeyError(key)


def const_dict_get(constDict, key, default):
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

    return default


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


class ConstDictWrapperBase(RefcountedWrapper):
    """Common method wrappers for all ConstDicts.

    We subclass this for things like 'keys', 'values', and 'items' since
    they all basically look like a const-dict with different methods
    """
    is_pod = False
    is_empty = False
    is_pass_by_ref = True

    def __init__(self, constDictType, behavior):
        assert hasattr(constDictType, '__typed_python_category__')
        super().__init__(constDictType if behavior is None else (constDictType, behavior))

        self.constDictType = constDictType
        self.keyType = typeWrapper(constDictType.KeyType)
        self.valueType = typeWrapper(constDictType.ValueType)
        self.itemType = typeWrapper(Tuple(constDictType.KeyType, constDictType.ValueType))

        self.kvBytecount = self.keyType.getBytecount() + self.valueType.getBytecount()
        self.keyBytecount = self.keyType.getBytecount()

        self.layoutType = native_ast.Type.Struct(element_types=(
            ('refcount', native_ast.Int64),
            ('hash_cache', native_ast.Int32),
            ('count', native_ast.Int32),
            ('subpointers', native_ast.Int32),
            ('data', native_ast.UInt8)
        ), name='ConstDictLayout').pointer()

    def getNativeLayoutType(self):
        return self.layoutType

    def on_refcount_zero(self, context, instance):
        assert instance.isReference

        if self.keyType.is_pod and self.valueType.is_pod:
            return runtime_functions.free.call(instance.nonref_expr.cast(native_ast.UInt8Ptr))
        else:
            return (
                context.converter.defineNativeFunction(
                    "destructor_" + str(self.constDictType),
                    ('destructor', self),
                    [self],
                    typeWrapper(type(None)),
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


class ConstDictWrapper(ConstDictWrapperBase):
    def __init__(self, constDictType):
        super().__init__(constDictType, None)

    def convert_attribute(self, context, instance, attr):
        if attr in (
            "__iter__",
            "get_key_by_index_unsafe",
            "get_value_by_index_unsafe",
            "get_item_by_index_unsafe",
            "get_key_ptr_by_index_unsafe",
            "get_value_ptr_by_index_unsafe",
            "get_item_ptr_by_index_unsafe",
            "keys",
            "values",
            "items",
            "get"
        ):
            return instance.changeType(BoundMethodWrapper.Make(self, attr))

        return super().convert_attribute(context, instance, attr)

    def convert_default_initialize(self, context, instance):
        context.pushEffect(
            instance.expr.store(self.layoutType.zero())
        )

    def convert_method_call(self, context, instance, methodname, args, kwargs):
        if methodname == "get" and not kwargs:
            if len(args) == 1:
                return self.convert_get(context, instance, args[0], context.constant(None))
            elif len(args) == 2:
                return self.convert_get(context, instance, args[0], args[1])

        if (methodname == "__iter__" or methodname == "keys") and not args and not kwargs:
            itType = ConstDictKeysIterator(
                self.keyType.typeRepresentation,
                self.valueType.typeRepresentation
            )

            return typeWrapper(itType).convert_type_call(
                context,
                None,
                [instance],
                {}
            )

        if methodname == "values" and not args and not kwargs:
            itType = ConstDictValuesIterator(
                self.keyType.typeRepresentation,
                self.valueType.typeRepresentation
            )

            return typeWrapper(itType).convert_type_call(
                context,
                None,
                [instance],
                {}
            )

        if methodname == "items" and not args and not kwargs:
            itType = ConstDictItemsIterator(
                self.keyType.typeRepresentation,
                self.valueType.typeRepresentation
            )

            return typeWrapper(itType).convert_type_call(
                context,
                None,
                [instance],
                {}
            )

        if kwargs:
            return super().convert_method_call(context, instance, methodname, args, kwargs)

        if methodname == "get_key_by_index_unsafe":
            if len(args) == 1:
                ix = args[0].toInt64()
                if ix is None:
                    return

                return self.convert_getkey_by_index_unsafe(context, instance, ix)

        if methodname == "get_item_by_index_unsafe":
            if len(args) == 1:
                ix = args[0].toInt64()
                if ix is None:
                    return

                return self.convert_getitem_by_index_unsafe(context, instance, ix)

        if methodname == "get_value_by_index_unsafe":
            if len(args) == 1:
                ix = args[0].toInt64()
                if ix is None:
                    return

                return self.convert_getvalue_by_index_unsafe(context, instance, ix)

        if methodname == "get_key_ptr_by_index_unsafe":
            if len(args) == 1:
                ix = args[0].toInt64()
                if ix is None:
                    return

                return self.convert_getkey_by_index_unsafe(context, instance, ix).asPointer()

        if methodname == "get_item_ptr_by_index_unsafe":
            if len(args) == 1:
                ix = args[0].toInt64()
                if ix is None:
                    return

                return self.convert_getitem_by_index_unsafe(context, instance, ix).asPointer()

        if methodname == "get_value_ptr_by_index_unsafe":
            if len(args) == 1:
                ix = args[0].toInt64()
                if ix is None:
                    return

                return self.convert_getvalue_by_index_unsafe(context, instance, ix).asPointer()

        return super().convert_method_call(context, instance, methodname, args, kwargs)

    def convert_getkey_by_index_unsafe(self, context, expr, item):
        return context.pushReference(
            self.keyType,
            expr.nonref_expr.ElementPtrIntegers(0, 4).elemPtr(
                item.nonref_expr.mul(native_ast.const_int_expr(self.kvBytecount))
            ).cast(self.keyType.getNativeLayoutType().pointer())
        )

    def convert_getitem_by_index_unsafe(self, context, expr, item):
        return context.pushReference(
            self.itemType,
            expr.nonref_expr.ElementPtrIntegers(0, 4).elemPtr(
                item.nonref_expr.mul(native_ast.const_int_expr(self.kvBytecount))
            ).cast(self.itemType.getNativeLayoutType().pointer())
        )

    def convert_getvalue_by_index_unsafe(self, context, expr, item):
        return context.pushReference(
            self.valueType,
            expr.nonref_expr.ElementPtrIntegers(0, 4).elemPtr(
                item.nonref_expr.mul(native_ast.const_int_expr(self.kvBytecount))
                .add(native_ast.const_int_expr(self.keyBytecount))
            ).cast(self.valueType.getNativeLayoutType().pointer())
        )

    def convert_bin_op(self, context, left, op, right, inplace):
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

        return super().convert_bin_op(context, left, op, right, inplace)

    def convert_bin_op_reverse(self, context, left, op, right, inplace):
        if op.matches.In:
            right = right.convert_to_type(self.keyType, ConversionLevel.UpcastContainers)
            if right is None:
                return None

            return context.call_py_function(const_dict_contains, (left, right), {})

        return super().convert_bin_op_reverse(context, left, op, right, inplace)

    def convert_getitem(self, context, instance, item):
        item = item.convert_to_type(self.keyType, ConversionLevel.UpcastContainers)
        if item is None:
            return None

        return context.call_py_function(const_dict_getitem, (instance, item), {})

    def convert_get(self, context, expr, item, default):
        if item is None or expr is None or default is None:
            return None

        item = item.convert_to_type(self.keyType, ConversionLevel.UpcastContainers)
        if item is None:
            return None

        return context.call_py_function(const_dict_get, (expr, item, default), {})

    def convert_len_native(self, expr):
        if isinstance(expr, TypedExpression):
            expr = expr.nonref_expr
        return native_ast.Expression.Branch(
            cond=expr,
            false=native_ast.const_int_expr(0),
            true=expr.ElementPtrIntegers(0, 2).load().cast(native_ast.Int64)
        )

    def convert_len(self, context, expr):
        return context.pushPod(int, self.convert_len_native(expr.nonref_expr))

    def _can_convert_to_type(self, targetType, conversionLevel):
        if not conversionLevel.isNewOrHigher():
            return False

        if targetType.typeRepresentation is bool:
            return True

        if targetType.typeRepresentation is str:
            return "Maybe"

        return False

    def convert_to_type_with_target(self, context, instance, targetVal, conversionLevel, mayThrowOnFailure=False):
        if targetVal.expr_type.typeRepresentation is bool:
            res = context.pushPod(bool, self.convert_len_native(instance.nonref_expr).neq(0))
            context.pushEffect(
                targetVal.expr.store(res.nonref_expr)
            )
            return context.constant(True)

        return super().convert_to_type_with_target(context, instance, targetVal, conversionLevel, mayThrowOnFailure)


@TypeFunction
def ConstDictKeysIterator(K, V):
    @Held
    class ConstDictKeysIterator(
        Class,
        Final,
        __name__=f"ConstDictKeysIterator({K.__name__}, {V.__name__})"
    ):
        slotIx = Member(int, nonempty=True)
        count = Member(int, nonempty=True)
        instance = Member(ConstDict(K, V), nonempty=True)

        def __init__(self, instance):
            self.instance = instance
            self.slotIx = 0
            self.count = len(instance)

        def __iter__(self):
            return self

        def __next__(self) -> K:
            res = self.__fastnext__()
            if res:
                return res.get()
            raise StopIteration()

        def __fastnext__(self) -> PointerTo(K):
            if self.slotIx < self.count:
                slot = self.slotIx
                self.slotIx += 1
                return self.instance.get_key_ptr_by_index_unsafe(slot)
            else:
                return PointerTo(K)()

    return ConstDictKeysIterator


@TypeFunction
def ConstDictValuesIterator(K, V):
    @Held
    class ConstDictValuesIterator(
        Class,
        Final,
        __name__=f"ConstDictValuesIterator({K.__name__}, {V.__name__})"
    ):
        slotIx = Member(int, nonempty=True)
        count = Member(int, nonempty=True)
        instance = Member(ConstDict(K, V), nonempty=True)

        def __init__(self, instance):
            self.instance = instance
            self.slotIx = 0
            self.count = len(instance)

        def __iter__(self):
            return self

        def __next__(self) -> V:
            res = self.__fastnext__()
            if res:
                return res.get()
            raise StopIteration()

        def __fastnext__(self) -> PointerTo(V):
            if self.slotIx < self.count:
                slot = self.slotIx
                self.slotIx += 1
                return self.instance.get_value_ptr_by_index_unsafe(slot)
            else:
                return PointerTo(V)()

    return ConstDictValuesIterator


@TypeFunction
def ConstDictItemsIterator(K, V):
    @Held
    class ConstDictItemsIterator(
        Class,
        Final,
        __name__=f"ConstDictItemsIterator({K.__name__}, {V.__name__})"
    ):
        slotIx = Member(int, nonempty=True)
        count = Member(int, nonempty=True)
        instance = Member(ConstDict(K, V), nonempty=True)

        def __init__(self, instance):
            self.instance = instance
            self.slotIx = 0
            self.count = len(instance)

        def __iter__(self):
            return self

        def __next__(self) -> Tuple(K, V):
            res = self.__fastnext__()
            if res:
                return res.get()
            raise StopIteration()

        def __fastnext__(self) -> PointerTo(Tuple(K, V)):
            if self.slotIx < self.count:
                slot = self.slotIx
                self.slotIx += 1
                return self.instance.get_item_ptr_by_index_unsafe(slot)
            else:
                return PointerTo(Tuple(K, V))()

    return ConstDictItemsIterator
