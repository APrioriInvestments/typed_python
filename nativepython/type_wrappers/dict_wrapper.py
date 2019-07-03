#   Copyright 2018 Braxton Mckee
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
from nativepython.typed_expression import TypedExpression
import nativepython.type_wrappers.runtime_functions as runtime_functions
from nativepython.type_wrappers.bound_compiled_method_wrapper import BoundCompiledMethodWrapper

from typed_python import NoneType, Tuple, PointerTo, Int32, Int64, UInt8

import nativepython.native_ast as native_ast
import nativepython


typeWrapper = lambda t: nativepython.python_object_representation.typedPythonTypeToTypeWrapper(t)

EMPTY = -1
DELETED = -2


def dict_add_slot(instance, itemHash, slot):
    if instance._hash_table_count * 2 + 1 > instance._hash_table_size or instance._hash_table_empty_slots < instance._hash_table_size >> 2 + 1:
        instance._resizeTableUnsafe()

    offset = itemHash % instance._hash_table_size;
    while True:
        if instance._hash_table_slots[offset] == EMPTY or instance._hash_table_slots[offset] == DELETED:
            if instance._hash_table_slots[offset] == EMPTY:
                instance._hash_table_empty_slots -= 1

            instance._hash_table_slots[offset] = slot
            instance._hash_table_hashes[offset] = itemHash
            instance._items_populated[slot] = 1
            instance._hash_table_count += 1

            return

        offset += 1
        if offset > instance._hash_table_size:
            offset = 0


def dict_slot_for_key(instance, item, itemHash):
    slots = instance._hash_table_slots

    if not slots:
        return -1

    offset = itemHash % instance._hash_table_size

    while True:
        slotIndex = int((slots + offset).get())

        if slotIndex == EMPTY:
            return -1

        if slotIndex != DELETED and (instance._hash_table_hashes + offset).get() == itemHash:
            if instance.getKeyByIndexUnsafe(slotIndex) == item:
                return slotIndex

        slotIndex += 1
        if slotIndex >= instance._hash_table_size:
            slotIndex = 0

    # not necessary, but currently we don't currently realize that the while loop
    # never exits, and so we think there's a possibility we return None
    return 0


def dict_getitem(instance, item):
    itemHash = hash(item)

    slot = dict_slot_for_key(instance, itemHash, item)

    if slot == -1:
        raise Exception("Key doesn't exist.")

    return instance.getValueByIndexUnsafe(slot)


def dict_setitem(instance, key, value):
    itemHash = hash(key)

    slot = dict_slot_for_key(instance, itemHash, key)

    if slot == -1:
        newSlot = instance._allocateNewSlotUnsafe()
        dict_add_slot(instance, itemHash, newSlot)
        instance.setKeyByIndexUnsafe(newSlot, key)
        instance.setValueByIndexUnsafe(newSlot, value)
    else:
        instance.setValueByIndexUnsafe(slot, value)


class DictWrapper(RefcountedWrapper):
    is_pod = False
    is_empty = False
    is_pass_by_ref = True

    def __init__(self, t):
        assert hasattr(t, '__typed_python_category__')
        super().__init__(t)

        self.keyType = typeWrapper(t.KeyType)
        self.valueType = typeWrapper(t.ValueType)
        self.itemType = typeWrapper(Tuple(t.KeyType, t.ValueType))

        self.kvBytecount = self.keyType.getBytecount() + self.valueType.getBytecount()
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
            ('hash_table_empty_slots', native_ast.Int64)
        ), name="DictWrapper").pointer()

    def convert_attribute(self, context, expr, attr):
        if attr in ("getItemByIndexUnsafe", "getKeyByIndexUnsafe", "getValueByIndexUnsafe",
                "setValueByIndexUnsafe", "setKeyByIndexUnsafe", "_allocateNewSlotUnsafe", "_resizeTableUnsafe"):
            return expr.changeType(BoundCompiledMethodWrapper(self, attr))

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
        if attr == '_hash_table_count':
            val = expr.convert_to_type(int)
            if val is None:
                return None
            return context.pushEffect(
                instance.nonref_expr.ElementPtrIntegers(0, 8).store(val.nonref_expr)
            )

        if attr == '_hash_table_empty_slots':
            val = expr.convert_to_type(int)
            if val is None:
                return None
            return context.pushEffect(
                instance.nonref_expr.ElementPtrIntegers(0, 9).store(val.nonref_expr)
            )

        return super().convert_set_attribute(context, instance, attr, expr)

    def convert_method_call(self, context, instance, methodname, args, kwargs):
        if kwargs:
            return super().convert_method_call(context, instance, methodname, args, kwargs)

        if len(args) == 0:
            if methodname == "_resizeTableUnsafe":
                context.pushEffect(
                    runtime_functions.dict_resizeTable.call(
                        instance.nonref_expr.cast(native_ast.VoidPtr)
                    )
                )
                return context.pushVoid()

            if methodname == "_allocateNewSlotUnsafe":
                return context.pushPod(
                    Int32,
                    runtime_functions.dict_allocateNewSlot.call(
                        instance.nonref_expr.cast(native_ast.VoidPtr),
                        context.constant(self.kvBytecount)
                    )
                )

        if len(args) == 1:
            if methodname in ("getItemByIndexUnsafe", "getKeyByIndexUnsafe", "getValueByIndexUnsafe"):
                index = args[0].convert_to_type(int)
                if index is None:
                    return None

                item = context.pushReference(
                    self.itemType,
                    instance.nonref_expr.ElementPtrIntegers(0, 1).load().cast(
                        self.itemType.getNativeLayoutType().pointer()
                    ).elemPtr(index.toInt64().nonref_expr)
                )

                if methodname == "getItemByIndexUnsafe":
                    return item
                elif methodname == "getKeyByIndexUnsafe":
                    # take the first item in the tuple
                    return item.expr_type.refAs(context, item, 0)
                else:
                    # take the second item in the tuple
                    return item.expr_type.refAs(context, item, 1)

        if len(args) == 2:
            if methodname == "setValueByIndexUnsafe":
                index = args[0].convert_to_type(int)
                if index is None:
                    return None

                value = args[1].convert_to_type(self.valueType)
                if value is None:
                    return None

                item = context.pushReference(
                    self.itemType,
                    instance.nonref_expr.ElementPtrIntegers(0, 1).load().cast(
                        self.itemType.getNativeLayoutType().pointer()
                    ).elemPtr(index.toInt64().nonref_expr)
                )

                item.expr_type.refAs(context, item, 1).convert_assign(value)

                return context.pushVoid()

            if methodname == "setKeyByIndexUnsafe":
                index = args[0].convert_to_type(int)
                if index is None:
                    return None

                key = args[1].convert_to_type(self.keyType)
                if key is None:
                    return None

                item = context.pushReference(
                    self.itemType,
                    instance.nonref_expr.ElementPtrIntegers(0, 1).load().cast(
                        self.itemType.getNativeLayoutType().pointer()
                    ).elemPtr(index.toInt64().nonref_expr)
                )

                item.expr_type.refAs(context, item, 0).convert_assign(key)

                return context.pushVoid()

        return super().convert_method_call(context, instance, methodname, args, kwargs)

    def convert_getitem(self, context, expr, item):
        if item is None or expr is None:
            return None

        item = item.convert_to_type(self.keyType)
        if item is None:
            return None

        return context.call_py_function(dict_getitem, (expr, item), {})

    def convert_setitem(self, context, expr, key, value):
        if key is None or expr is None or value is None:
            return None

        key = key.convert_to_type(self.keyType)
        if key is None:
            return None

        value = value.convert_to_type(self.keyType)
        if value is None:
            return None

        return context.call_py_function(dict_setitem, (expr, key, value), {})

    def getNativeLayoutType(self):
        return self.layoutType

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

    def convert_getkey_by_index_unsafe(self, context, expr, item):
        return context.pushReference(
            self.keyType,
            expr.nonref_expr.ElementPtrIntegers(0, 1)
            .elemPtr(item.nonref_expr.mul(native_ast.const_int_expr(self.kvBytecount)))
            .cast(self.keyType.getNativeLayoutType().pointer())
        )

    def convert_getvalue_by_index_unsafe(self, context, expr, item):
        return context.pushReference(
            self.keyType,
            expr.nonref_expr.ElementPtrIntegers(0, 1)
            .elemPtr(
                item.nonref_expr.mul(native_ast.const_int_expr(self.kvBytecount))
                .add(native_ast.const_int_expr(self.keyBytecount))
            ).cast(self.valueType.getNativeLayoutType().pointer())
        )

    def generateNativeDestructorFunction(self, context, out, inst):
        with context.loop(self.convert_items_reserved(context, inst)) as i:
            with context.ifelse(self.convert_slot_populated_native(inst, i)) as (then, otherwise):
                self.convert_getkey_by_index_unsafe(context, inst, i).convert_destroy()
                self.convert_getvalue_by_index_unsafe(context, inst, i).convert_destroy()

        context.pushEffect(
            runtime_functions.free.call(inst.nonref_expr.ElementPtrIntegers(0, 1).load().cast(native_ast.UInt8Ptr)) >>
            runtime_functions.free.call(inst.nonref_expr.ElementPtrIntegers(0, 2).load().cast(native_ast.UInt8Ptr)) >>
            runtime_functions.free.call(inst.nonref_expr.ElementPtrIntegers(0, 5).load().cast(native_ast.UInt8Ptr)) >>
            runtime_functions.free.call(inst.nonref_expr.ElementPtrIntegers(0, 6).load().cast(native_ast.UInt8Ptr)) >>
            runtime_functions.free.call(inst.nonref_expr.cast(native_ast.UInt8Ptr))
        )
