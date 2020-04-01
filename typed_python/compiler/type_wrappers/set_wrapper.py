#   Copyright 2020 Braxton Mckee
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
from typed_python.compiler.typed_expression import TypedExpression
import typed_python.compiler.type_wrappers.runtime_functions as runtime_functions
from typed_python.compiler.type_wrappers.bound_method_wrapper import BoundMethodWrapper
from typed_python.compiler.type_wrappers.wrapper import Wrapper
from typed_python.compiler.type_wrappers.native_hash import NativeHash, table_add_slot, table_slot_for_key, \
    table_next_slot, table_remove_key, table_clear, table_contains, table_contains_not
from typed_python import NoneType, PointerTo, Int32, Int64, UInt8

import typed_python.compiler.native_ast as native_ast
import typed_python.compiler


typeWrapper = lambda t: typed_python.compiler.python_object_representation.typedPythonTypeToTypeWrapper(t)


def set_add(instance, key):
    itemHash = NativeHash()(key)

    slot = table_slot_for_key(instance, itemHash, key)

    if slot == -1:
        newSlot = instance._allocateNewSlotUnsafe()
        table_add_slot(instance, itemHash, newSlot)
        instance.initializeKeyByIndexUnsafe(newSlot, key)


def set_add_or_remove(instance, key):
    itemHash = NativeHash()(key)

    slot = table_slot_for_key(instance, itemHash, key)

    if slot == -1:
        newSlot = instance._allocateNewSlotUnsafe()
        table_add_slot(instance, itemHash, newSlot)
        instance.initializeKeyByIndexUnsafe(newSlot, key)
    else:
        table_remove_key(instance, key, itemHash, False)


def set_remove(instance, key):
    itemHash = NativeHash()(key)

    table_remove_key(instance, key, itemHash, True)


def set_discard(instance, key):
    itemHash = NativeHash()(key)

    table_remove_key(instance, key, itemHash, False)


def set_pop(instance):
    slotIx = 0

    while slotIx < instance._items_reserved:
        if instance._items_populated[slotIx]:
            res = instance.getKeyByIndexUnsafe(slotIx)
            set_remove(instance, res)
            return res

        slotIx += 1
    raise KeyError(instance)


# this set_copy has been replaced by more performant code in hash_table_layout
# def set_copy(instance):
#     result = type(instance)()
#     for i in instance:
#         result.add(i)
#     return result


def set_union(left, right):
    if len(left) >= len(right):
        result = left.copy()
        for i in right:
            result.add(i)
    else:
        result = right.copy()
        for i in left:
            result.add(i)
    return result


# alternate way to do set_union; not as performant as above
def set_union1(left, right):
    result = type(left)()
    for i in left:
        result.add(i)
    for i in right:
        result.add(i)
    return result


def set_intersection(left, right):
    if len(left) > len(right):
        result = right.copy()
        for i in right:
            if i not in left:
                result.discard(i)
    else:
        result = left.copy()
        for i in left:
            if i not in right:
                result.discard(i)
    return result


# alternate way to do set_intersection; not as performant
def set_intersection1(left, right):
    result = type(left)()
    if len(left) > len(right):
        for i in right:
            if i in left:
                result.add(i)
    else:
        for i in left:
            if i in right:
                result.add(i)
    return result


# alternate way to do set_intersection; primitive loop is not measurably faster than "for" iterator
def set_intersection2(left, right):
    result = type(left)()
    if len(left) > len(right):
        slotIx = 0
        while slotIx < right._items_reserved:
            if right._items_populated[slotIx]:
                i = right.getKeyByIndexUnsafe(slotIx)
                if i in left:
                    result.add(i)
            slotIx += 1
    else:
        slotIx = 0
        while slotIx < left._items_reserved:
            if left._items_populated[slotIx]:
                i = left.getKeyByIndexUnsafe(slotIx)
                if i in right:
                    result.add(i)
            slotIx += 1
    return result


def set_difference(left, right):
    if len(left) / 4 > len(right):
        result = left.copy()
        for i in right:
            result.discard(i)
    else:
        result = type(left)()
        for i in left:
            if i not in right:
                result.add(i)
    return result


def set_symmetric_difference(left, right):
    if len(left) / 2 > len(right):
        result = left.copy()
        for i in right:
            set_add_or_remove(result, i)
    elif len(right) / 2 > len(left):
        result = right.copy()
        for i in left:
            set_add_or_remove(result, i)
    else:
        result = type(left)()
        for i in left:
            if i not in right:
                result.add(i)
        for i in right:
            if i not in left:
                result.add(i)
    return result


def set_union_multiple(left, *others):
    result = left
    for o in others:
        result |= o
    return result


def set_intersection_multiple(left, *others):
    result = left
    for o in others:
        result &= o
    return result


def set_difference_multiple(left, *others):
    result = left
    for o in others:
        result -= o
    return result


def set_disjoint(left, right):
    for i in left:
        if i in right:
            return False
    return True


def set_subset(left, right):
    for i in left:
        if i not in right:
            return False
    return True


def set_proper_subset(left, right):
    for i in left:
        if i not in right:
            return False
    return len(left) < len(right)


def set_equal(left, right):
    return len(left) == len(right) and set_subset(left, right)


def set_not_equal(left, right):
    return len(left) != len(right) or not set_subset(right, left)


class SetWrapperBase(RefcountedWrapper):
    is_pod = False
    is_empty = False
    is_pass_by_ref = True

    def __init__(self, t, behavior):
        assert hasattr(t, '__typed_python_category__')
        super().__init__(t if behavior is None else (t, behavior))

        self.keyType = typeWrapper(t.KeyType)
        self.setType = t

        self.keyBytecount = self.keyType.getBytecount()

        self.layoutType = native_ast.Type.Struct(element_types=(
            ('refcount', native_ast.Int64),
            ('items', native_ast.UInt8Ptr),
            ('items_populated', native_ast.UInt8Ptr),
            ('items_reserved', native_ast.Int64),
            ('top_item_slot', native_ast.Int64),
            ('hash_table_slots', native_ast.Int32Ptr),
            ('hash_table_hashes', native_ast.Int32Ptr),
            ('hash_table_size', native_ast.Int64),
            ('hash_table_count', native_ast.Int64),
            ('hash_table_empty_slots', native_ast.Int64),
            ('reserved', native_ast.Int64)  # not used
        ), name="SetWrapper").pointer()

    def on_refcount_zero(self, context, instance):
        assert instance.isReference

        return (
            context.converter.defineNativeFunction(
                "destructor_" + str(self.typeRepresentation),
                ('destructor', self),
                [self],
                typeWrapper(NoneType),
                self.generateNativeDestructorFunction
            ).call(instance)
        )

    def getNativeLayoutType(self):
        return self.layoutType


class SetWrapper(SetWrapperBase):
    def __init__(self, setType):
        super().__init__(setType, None)

    def convert_default_initialize(self, context, instance):
        context.pushEffect(
            instance.expr.store(
                runtime_functions.table_create.call().cast(self.layoutType)
            )
        )

    def convert_attribute(self, context, expr, attr):
        if attr in (
                "getKeyByIndexUnsafe", "deleteItemByIndexUnsafe",
                "initializeKeyByIndexUnsafe", "_allocateNewSlotUnsafe", "_resizeTableUnsafe",
                "_compressItemTableUnsafe",
                "add", "remove", "discard", "pop", "clear", "copy", "log",
                "union", "intersection", "difference", "symmetric_difference",
                "issubset", "issuperset", "isdisjoint"):
            return expr.changeType(BoundMethodWrapper.Make(self, attr))

        if attr == '_items':
            return context.pushPod(
                PointerTo(UInt8),
                expr.nonref_expr.ElementPtrIntegers(0, 1).load()
            )

        if attr == '_items_populated':
            return context.pushPod(
                PointerTo(UInt8),
                expr.nonref_expr.ElementPtrIntegers(0, 2).load()
            )

        if attr == '_items_reserved':
            return context.pushPod(
                Int64,
                expr.nonref_expr.ElementPtrIntegers(0, 3).load()
            )

        if attr == '_top_item_slot':
            return context.pushPod(
                Int64,
                expr.nonref_expr.ElementPtrIntegers(0, 4).load()
            )

        if attr == '_hash_table_slots':
            return context.pushPod(
                PointerTo(Int32),
                expr.nonref_expr.ElementPtrIntegers(0, 5).load()
            )

        if attr == '_hash_table_hashes':
            return context.pushPod(
                PointerTo(Int32),
                expr.nonref_expr.ElementPtrIntegers(0, 6).load()
            )

        if attr == '_hash_table_size':
            return context.pushPod(
                int,
                expr.nonref_expr.ElementPtrIntegers(0, 7).load()
            )

        if attr == '_hash_table_count':
            return context.pushPod(
                int,
                expr.nonref_expr.ElementPtrIntegers(0, 8).load()
            )

        if attr == '_hash_table_empty_slots':
            return context.pushPod(
                int,
                expr.nonref_expr.ElementPtrIntegers(0, 9).load()
            )

        return super().convert_attribute(context, expr, attr)

    def convert_set_attribute(self, context, instance, attr, expr):
        if attr == '_items_reserved':
            val = expr.convert_to_type(int)
            if val is None:
                return None
            context.pushEffect(
                instance.nonref_expr.ElementPtrIntegers(0, 3).store(val.nonref_expr)
            )

            return context.pushVoid()

        if attr == '_top_item_slot':
            val = expr.convert_to_type(int)
            if val is None:
                return None
            context.pushEffect(
                instance.nonref_expr.ElementPtrIntegers(0, 4).store(val.nonref_expr)
            )

            return context.pushVoid()

        if attr == '_hash_table_size':
            val = expr.convert_to_type(int)
            if val is None:
                return None
            context.pushEffect(
                instance.nonref_expr.ElementPtrIntegers(0, 7).store(val.nonref_expr)
            )

            return context.pushVoid()

        if attr == '_hash_table_count':
            val = expr.convert_to_type(int)
            if val is None:
                return None
            context.pushEffect(
                instance.nonref_expr.ElementPtrIntegers(0, 8).store(val.nonref_expr)
            )

            return context.pushVoid()

        if attr == '_hash_table_empty_slots':
            val = expr.convert_to_type(int)
            if val is None:
                return None
            context.pushEffect(
                instance.nonref_expr.ElementPtrIntegers(0, 9).store(val.nonref_expr)
            )

            return context.pushVoid()

        return super().convert_set_attribute(context, instance, attr, expr)

    def convert_method_call(self, context, instance, methodname, args, kwargs):
        if kwargs:
            return super().convert_method_call(context, instance, methodname, args, kwargs)

        if methodname == "__iter__" and not args and not kwargs:
            res = context.push(
                SetKeysIteratorWrapper(self.setType),
                lambda instance:
                instance.expr.ElementPtrIntegers(0, 0).store(-1)
                # we initialize the set pointer below, so technically
                # if that were to throw, this would leak a bad value.
            )

            context.pushReference(
                self,
                res.expr.ElementPtrIntegers(0, 1)
            ).convert_copy_initialize(instance)

            return res

        if methodname == 'union':
            return context.call_py_function(set_union_multiple, (instance, *args), {})

        if methodname == 'intersection':
            return context.call_py_function(set_intersection_multiple, (instance, *args), {})

        if methodname == 'difference':
            return context.call_py_function(set_difference_multiple, (instance, *args), {})

        if methodname == 'symmetric_difference':
            if len(args) != 1:
                return context.pushException(TypeError, f"symmetric_difference() takes exactly one argument ({len(args)} given)")
            return context.call_py_function(set_symmetric_difference, (instance, args[0]), {})

        if len(args) == 0:
            if methodname == "pop":
                return context.call_py_function(set_pop, (instance,), {})
            if methodname == "clear":
                return context.call_py_function(table_clear, (instance,), {})
            if methodname == "_compressItemTableUnsafe":
                context.pushEffect(
                    runtime_functions.table_compress.call(
                        instance.nonref_expr.cast(native_ast.VoidPtr),
                        context.constant(self.keyBytecount)
                    )
                )
                return context.pushVoid()

            if methodname == "_resizeTableUnsafe":
                context.pushEffect(
                    runtime_functions.table_resize.call(
                        instance.nonref_expr.cast(native_ast.VoidPtr)
                    )
                )
                return context.pushVoid()

            if methodname == "copy":
                # alternate way, but less performant:
                # return context.call_py_function(set_copy, (instance,), {})

                tp = context.getTypePointer(self.setType)
                if tp:
                    return context.push(
                        typeWrapper(self.setType),
                        lambda ref: ref.expr.store(
                            runtime_functions.table_copy.call(
                                instance.nonref_expr.cast(native_ast.VoidPtr),
                                tp
                            ).cast(self.layoutType)
                        )
                    )

            if methodname == "_allocateNewSlotUnsafe":
                return context.pushPod(
                    Int32,
                    runtime_functions.table_allocate_new_slot.call(
                        instance.nonref_expr.cast(native_ast.VoidPtr),
                        context.constant(self.keyBytecount)
                    )
                )

        if len(args) == 1:
            if methodname == "isdisjoint":
                return context.call_py_function(set_disjoint, (instance, args[0]), {})
            if methodname == "issubset":
                return context.call_py_function(set_subset, (instance, args[0]), {})
            if methodname == "issuperset":
                return context.call_py_function(set_subset, (args[0], instance), {})

            if methodname == "add":
                key = args[0].convert_to_type(self.keyType, explicit=False)
                if key is None:
                    return None

                return context.call_py_function(set_add, (instance, key), {})

            if methodname == "remove":
                key = args[0].convert_to_type(self.keyType, explicit=False)
                if key is None:
                    return None

                return context.call_py_function(set_remove, (instance, key), {})

            if methodname == "discard":
                key = args[0].convert_to_type(self.keyType, explicit=False)
                if key is None:
                    return None

                return context.call_py_function(set_discard, (instance, key), {})

            if methodname in ("getKeyByIndexUnsafe", "deleteItemByIndexUnsafe"):
                index = args[0].convert_to_type(int)
                if index is None:
                    return None

                key = context.pushReference(
                    self.keyType,
                    instance.nonref_expr.ElementPtrIntegers(0, 1).load().cast(
                        self.keyType.getNativeLayoutType().pointer()
                    ).elemPtr(index.toInt64().nonref_expr)
                )

                if methodname == "getKeyByIndexUnsafe":
                    return key
                elif methodname == "deleteItemByIndexUnsafe":
                    key.convert_destroy()
                    return context.pushVoid()

        if len(args) == 2:
            if methodname == "initializeKeyByIndexUnsafe":
                index = args[0].convert_to_type(int)
                if index is None:
                    return None

            key = args[1].convert_to_type(self.keyType)
            if key is None:
                return None

            item = context.pushReference(
                self.keyType,
                instance.nonref_expr.ElementPtrIntegers(0, 1).load().cast(
                    self.keyType.getNativeLayoutType().pointer()
                ).elemPtr(index.toInt64().nonref_expr)
            )

            item.convert_copy_initialize(key)

            return context.pushVoid()

        return super().convert_method_call(context, instance, methodname, args, kwargs)

    def convert_len_native(self, expr):
        if isinstance(expr, TypedExpression):
            expr = expr.nonref_expr
        return expr.ElementPtrIntegers(0, 8).load().cast(native_ast.Int64)

    def convert_items_reserved_native(self, expr):
        if isinstance(expr, TypedExpression):
            expr = expr.nonref_expr
        return expr.ElementPtrIntegers(0, 3).load().cast(native_ast.Int64)

    def convert_items_reserved(self, context, expr):
        return context.pushPod(int, self.convert_items_reserved_native(expr))

    def convert_slot_populated_native(self, expr, slotIx):
        if isinstance(expr, TypedExpression):
            expr = expr.nonref_expr
        return expr.ElementPtrIntegers(0, 2).load().elemPtr(slotIx.nonref_expr).load()

    def convert_len(self, context, expr):
        return context.pushPod(int, self.convert_len_native(expr))

    def convert_bin_op(self, context, left, op, right, inplace):
        if right.expr_type == left.expr_type:
            if op.matches.BitOr:
                return context.call_py_function(set_union, (left, right), {})
            if op.matches.BitAnd:
                return context.call_py_function(set_intersection, (left, right), {})
            if op.matches.Sub:
                return context.call_py_function(set_difference, (left, right), {})
            if op.matches.BitXor:
                return context.call_py_function(set_symmetric_difference, (left, right), {})
            if op.matches.Eq:
                return context.call_py_function(set_equal, (left, right), {})
            if op.matches.NotEq:
                return context.call_py_function(set_not_equal, (left, right), {})
            if op.matches.Lt:
                return context.call_py_function(set_proper_subset, (left, right), {})
            if op.matches.LtE:
                return context.call_py_function(set_subset, (left, right), {})
            if op.matches.Gt:
                return context.call_py_function(set_proper_subset, (right, left), {})
            if op.matches.GtE:
                return context.call_py_function(set_subset, (right, left), {})

        return super().convert_bin_op(context, left, op, right, inplace)

    def convert_bin_op_reverse(self, context, right, op, left, inplace):
        if op.matches.In or op.matches.NotIn:
            left = left.convert_to_type(self.keyType, False)
            if left is None:
                return None

            return context.call_py_function(
                table_contains if op.matches.In else table_contains_not,
                (right, left),
                {}
            )

        return super().convert_bin_op_reverse(context, right, op, left, inplace)

    def convert_getkey_by_index_unsafe(self, context, expr, item):
        return context.pushReference(
            self.keyType,
            expr.nonref_expr.ElementPtrIntegers(0, 1)
            .elemPtr(item.nonref_expr.mul(native_ast.const_int_expr(self.keyBytecount)))
            .cast(self.keyType.getNativeLayoutType())
        )

    def generateNativeDestructorFunction(self, context, out, inst):
        with context.loop(self.convert_items_reserved(context, inst)) as i:
            with context.ifelse(self.convert_slot_populated_native(inst, i)) as (then, otherwise):
                with then:
                    self.convert_getkey_by_index_unsafe(context, inst, i).convert_destroy()

        context.pushEffect(
            runtime_functions.free.call(inst.nonref_expr.ElementPtrIntegers(0, 1).load().cast(native_ast.UInt8Ptr)) >>
            runtime_functions.free.call(inst.nonref_expr.ElementPtrIntegers(0, 2).load().cast(native_ast.UInt8Ptr)) >>
            runtime_functions.free.call(inst.nonref_expr.ElementPtrIntegers(0, 5).load().cast(native_ast.UInt8Ptr)) >>
            runtime_functions.free.call(inst.nonref_expr.ElementPtrIntegers(0, 6).load().cast(native_ast.UInt8Ptr)) >>
            runtime_functions.free.call(inst.nonref_expr.cast(native_ast.UInt8Ptr))
        )

    def convert_type_call(self, context, typeInst, args, kwargs):
        if len(args) == 0 and not kwargs:
            return context.push(self, lambda x: x.convert_default_initialize())

        if len(args) == 1 and not kwargs:
            return args[0].convert_to_type(self, True)

        return super().convert_type_call(context, typeInst, args, kwargs)

    def convert_bool_cast(self, context, expr):
        return context.pushPod(bool, self.convert_len_native(expr.nonref_expr).neq(0))


class SetMakeIteratorWrapper(SetWrapperBase):
    def convert_method_call(self, context, expr, methodname, args, kwargs):
        if methodname == "__iter__" and not args and not kwargs:
            res = context.push(
                # self.iteratorType is inherited from our specialized children
                # who pick whether we're an iterator over keys, values, items, etc.
                self.iteratorType,
                lambda instance:
                    instance.expr.ElementPtrIntegers(0, 0).store(-1)
            )

            context.pushReference(
                self,
                res.expr.ElementPtrIntegers(0, 1)
            ).convert_copy_initialize(expr)

            return res

        return super().convert_method_call(context, expr, methodname, args, kwargs)


class SetKeysWrapper(SetMakeIteratorWrapper):
    def __init__(self, setType):
        super().__init__(setType, "keys")
        self.iteratorType = SetKeysIteratorWrapper(setType)


class SetIteratorWrapper(Wrapper):
    is_pod = False
    is_empty = False
    is_pass_by_ref = True

    def __init__(self, setType, iteratorType):
        self.setType = setType
        self.iteratorType = iteratorType
        super().__init__((setType, "iterator", iteratorType))

    def getNativeLayoutType(self):
        return native_ast.Type.Struct(
            element_types=(("pos", native_ast.Int64), ("set", SetWrapper(self.setType).getNativeLayoutType())),
            name="const_set_iterator"
        )

    def convert_next(self, context, expr):
        nextSlotIx = context.call_py_function(table_next_slot, (self.refAs(context, expr, 1), self.refAs(context, expr, 0)), {})

        if nextSlotIx is None:
            return None, None

        context.pushEffect(
            expr.expr.ElementPtrIntegers(0, 0).store(
                nextSlotIx.nonref_expr
            )
        )
        canContinue = context.pushPod(
            bool,
            nextSlotIx.nonref_expr.gte(0)
        )

        nextIx = context.pushReference(int, expr.expr.ElementPtrIntegers(0, 0))

        return self.iteratedItemForReference(context, expr, nextIx), canContinue

    def refAs(self, context, expr, which):
        assert expr.expr_type == self

        if which == 0:
            return context.pushReference(int, expr.expr.ElementPtrIntegers(0, 0))

        if which == 1:
            return context.pushReference(
                self.setType,
                expr.expr
                    .ElementPtrIntegers(0, 1)
                    .cast(SetWrapper(self.setType).getNativeLayoutType().pointer())
            )

    def convert_assign(self, context, expr, other):
        assert expr.isReference

        for i in range(2):
            self.refAs(context, expr, i).convert_assign(self.refAs(context, other, i))

    def convert_copy_initialize(self, context, expr, other):
        for i in range(2):
            self.refAs(context, expr, i).convert_copy_initialize(self.refAs(context, other, i))

    def convert_destroy(self, context, expr):
        self.refAs(context, expr, 1).convert_destroy()


class SetKeysIteratorWrapper(SetIteratorWrapper):
    def __init__(self, setType):
        super().__init__(setType, "keys")

    def iteratedItemForReference(self, context, expr, ixExpr):
        return SetWrapper(self.setType).convert_method_call(
            context,
            self.refAs(context, expr, 1),
            "getKeyByIndexUnsafe",
            (ixExpr,),
            {}
        )
