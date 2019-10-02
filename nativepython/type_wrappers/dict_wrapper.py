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
from nativepython.type_wrappers.wrapper import Wrapper
from nativepython.type_wrappers.compilable_builtin import CompilableBuiltin
from typed_python import NoneType, Tuple, PointerTo, Int32, Int64, UInt8, Bool

import nativepython.native_ast as native_ast
import nativepython


typeWrapper = lambda t: nativepython.python_object_representation.typedPythonTypeToTypeWrapper(t)

EMPTY = -1
DELETED = -2


class CPlusPlusStyleMod(CompilableBuiltin):
    def convert_call(self, context, instance, args, kwargs):
        if len(args) == 2 and args[0].expr_type.typeRepresentation == Int32 and args[1].expr_type.typeRepresentation == Int64:
            return context.pushPod(
                int,
                args[0].nonref_expr.cast(native_ast.Int64).mod(args[1].nonref_expr)
            )

        if len(args) == 2 and args[0].expr_type.typeRepresentation == Int64 and args[1].expr_type.typeRepresentation == Int64:
            return context.pushPod(
                int,
                args[0].nonref_expr.mod(args[1].nonref_expr)
            )

        return super().convert_call(context, instance, args, kwargs)


def dict_add_slot(instance, itemHash, slot):
    if (instance._hash_table_count * 2 + 1 > instance._hash_table_size or
            instance._hash_table_empty_slots < instance._hash_table_size >> 2 + 1):
        instance._resizeTableUnsafe()

    if itemHash < 0:
        itemHash = -itemHash

    offset = CPlusPlusStyleMod(itemHash, instance._hash_table_size)

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

        if offset >= instance._hash_table_size:
            offset = 0


def dict_slot_for_key(instance, itemHash, item):
    slots = instance._hash_table_slots

    if not slots:
        return -1

    if itemHash < 0:
        itemHash = -itemHash

    offset = CPlusPlusStyleMod(itemHash, instance._hash_table_size)

    while True:
        slotIndex = int((slots + offset).get())

        if slotIndex == EMPTY:
            return -1

        if slotIndex != DELETED and (instance._hash_table_hashes + offset).get() == itemHash:
            if instance.getKeyByIndexUnsafe(slotIndex) == item:
                return slotIndex

        offset += 1
        if offset >= instance._hash_table_size:
            offset = 0

    # not necessary, but currently we don't realize that the while loop
    # never exits, and so we think there's a possibility we return None
    return 0


def dict_next_slot(instance, slotIx):
    slotIx += 1

    while slotIx < instance._items_reserved:
        if instance._items_populated[slotIx]:
            return slotIx
        slotIx += 1

    return -1


def dict_remove_key(instance, item, itemHash):
    slots = instance._hash_table_slots

    if not slots:
        raise Exception("Key doesn't exist")

    if instance._items_reserved > (instance._hash_table_count + 2) * 4:
        instance._compressItemTableUnsafe()

    if instance._hash_table_count < instance._hash_table_size >> 3:
        instance._resizeTableUnsafe()

    if itemHash < 0:
        itemHash = -itemHash

    offset = CPlusPlusStyleMod(itemHash, instance._hash_table_size)

    while True:
        slotIndex = int((slots + offset).get())

        if slotIndex == EMPTY:
            raise Exception("Key doesn't exist")

        if slotIndex != DELETED and (instance._hash_table_hashes + offset).get() == itemHash:
            if instance.getKeyByIndexUnsafe(slotIndex) == item:
                instance._hash_table_hashes[offset] = -1
                instance._hash_table_slots[offset] = DELETED
                instance._hash_table_count -= 1
                instance._items_populated[slotIndex] = 0

                instance.deleteItemByIndexUnsafe(slotIndex)
                return

        offset += 1
        if offset >= instance._hash_table_size:
            offset = 0

    # not necessary, but currently we don't currently realize that the while loop
    # never exits, and so we think there's a possibility we return None
    return 0


def dict_delitem(instance, item):
    itemHash = hash(item)

    dict_remove_key(instance, item, itemHash)


def dict_getitem(instance, item):
    itemHash = hash(item)

    slot = dict_slot_for_key(instance, itemHash, item)

    if slot == -1:
        raise Exception("Key doesn't exist")

    return instance.getValueByIndexUnsafe(slot)


def dict_contains(instance, item):
    itemHash = hash(item)

    slot = dict_slot_for_key(instance, itemHash, item)

    return slot != -1


def dict_contains_not(instance, item):
    itemHash = hash(item)

    slot = dict_slot_for_key(instance, itemHash, item)

    return slot == -1


def dict_setitem(instance, key, value):
    itemHash = hash(key)

    slot = dict_slot_for_key(instance, itemHash, key)

    if slot == -1:
        newSlot = instance._allocateNewSlotUnsafe()
        dict_add_slot(instance, itemHash, newSlot)
        instance.initializeKeyByIndexUnsafe(newSlot, key)
        instance.initializeValueByIndexUnsafe(newSlot, value)
    else:
        instance.assignValueByIndexUnsafe(slot, value)


def dict_setdefault(dict, item, defaultValue=None):
    if item not in dict:
        # TypeError: Can't convert from type String to type Int64
        dict[item] = defaultValue
    return dict[item]


class DictWrapperBase(RefcountedWrapper):
    is_pod = False
    is_empty = False
    is_pass_by_ref = True

    CAN_BE_NULL = False

    def __init__(self, t, behavior):
        assert hasattr(t, '__typed_python_category__')
        super().__init__(t if behavior is None else (t, behavior))

        self.keyType = typeWrapper(t.KeyType)
        self.valueType = typeWrapper(t.ValueType)
        self.itemType = typeWrapper(Tuple(t.KeyType, t.ValueType))
        self.dictType = t

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
            ('hash_table_empty_slots', native_ast.Int64),
            ('setdefault', native_ast.Int64)
        ), name="DictWrapper").pointer()

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


class DictWrapper(DictWrapperBase):
    def __init__(self, dictType):
        super().__init__(dictType, None)

    def convert_default_initialize(self, context, instance):
        context.pushEffect(
            instance.expr.store(
                runtime_functions.dict_create.call().cast(self.layoutType)
            )
        )

    def convert_attribute(self, context, expr, attr):
        if attr in (
                "getItemByIndexUnsafe", "getKeyByIndexUnsafe", "getValueByIndexUnsafe", "deleteItemByIndexUnsafe",
                "initializeValueByIndexUnsafe", "assignValueByIndexUnsafe",
                "initializeKeyByIndexUnsafe", "_allocateNewSlotUnsafe", "_resizeTableUnsafe",
                "_compressItemTableUnsafe", "get", "items", "keys", "values", "setdefault"):
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
                DictKeysIteratorWrapper(self.dictType),
                lambda instance:
                    instance.expr.ElementPtrIntegers(0, 0).store(-1)
                    # we initialize the dict pointer below, so technically
                    # if that were to throw, this would leak a bad value.
            )

            context.pushReference(
                self,
                res.expr.ElementPtrIntegers(0, 1)
            ).convert_copy_initialize(instance)

            return res

        if methodname == "keys" and not args and not kwargs:
            return instance.changeType(DictKeysWrapper(self.dictType))

        if methodname == "values" and not args and not kwargs:
            return instance.changeType(DictValuesWrapper(self.dictType))

        if methodname == "items" and not args and not kwargs:
            return instance.changeType(DictItemsWrapper(self.dictType))

        if len(args) == 0:
            if methodname == "_compressItemTableUnsafe":
                context.pushEffect(
                    runtime_functions.dict_compressItemTable.call(
                        instance.nonref_expr.cast(native_ast.VoidPtr),
                        context.constant(self.kvBytecount)
                    )
                )
                return context.pushVoid()

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

        if methodname == "setdefault":
            if len(args) == 1:
                return context.call_py_function(dict_setdefault, (instance, args[0]), {})
            else:
                return context.call_py_function(dict_setdefault, (instance, args[0], args[1]), {})

        if len(args) == 1:
            if methodname == "get":
                return self.convert_getitem(context, instance, args[0])

            if methodname in ("getItemByIndexUnsafe", "getKeyByIndexUnsafe", "getValueByIndexUnsafe", "deleteItemByIndexUnsafe"):
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
                elif methodname == "deleteItemByIndexUnsafe":
                    item.convert_destroy()
                    return context.pushVoid()
                elif methodname == "getKeyByIndexUnsafe":
                    # take the first item in the tuple
                    return item.expr_type.refAs(context, item, 0)
                else:
                    # take the second item in the tuple
                    return item.expr_type.refAs(context, item, 1)

        if len(args) == 2:
            if methodname in ("initializeValueByIndexUnsafe", 'assignValueByIndexUnsafe'):
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

                if methodname == 'assignValueByIndexUnsafe':
                    item.expr_type.refAs(context, item, 1).convert_assign(value)
                else:
                    item.expr_type.refAs(context, item, 1).convert_copy_initialize(value)

                return context.pushVoid()

            if methodname == "initializeKeyByIndexUnsafe":
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

                item.expr_type.refAs(context, item, 0).convert_copy_initialize(key)

                return context.pushVoid()

        return super().convert_method_call(context, instance, methodname, args, kwargs)

    def convert_delitem(self, context, expr, item):
        if item is None or expr is None:
            return None

        item = item.convert_to_type(self.keyType)
        if item is None:
            return None

        return context.call_py_function(dict_delitem, (expr, item), {})

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

        value = value.convert_to_type(self.valueType)
        if value is None:
            return None

        return context.call_py_function(dict_setitem, (expr, key, value), {})

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

    def convert_bin_op_reverse(self, context, left, op, right, inplace):
        if op.matches.In or op.matches.NotIn:
            right = right.convert_to_type(self.keyType)
            if right is None:
                return None

            return context.call_py_function(
                dict_contains if op.matches.In else dict_contains_not,
                (left, right),
                {}
            )

        return super().convert_bin_op(context, left, op, right, inplace)

    def convert_getkey_by_index_unsafe(self, context, expr, item):
        return context.pushReference(
            self.keyType,
            expr.nonref_expr.ElementPtrIntegers(0, 1).load()
                .elemPtr(item.nonref_expr.mul(native_ast.const_int_expr(self.kvBytecount)))
                .cast(self.keyType.getNativeLayoutType().pointer())
        )

    def convert_getvalue_by_index_unsafe(self, context, expr, item):
        return context.pushReference(
            self.valueType,
            expr.nonref_expr.ElementPtrIntegers(0, 1).load()
            .elemPtr(
                item.nonref_expr.mul(native_ast.const_int_expr(self.kvBytecount))
                .add(native_ast.const_int_expr(self.keyBytecount))
            ).cast(self.valueType.getNativeLayoutType().pointer())
        )

    def generateNativeDestructorFunction(self, context, out, inst):
        with context.loop(self.convert_items_reserved(context, inst)) as i:
            with context.ifelse(self.convert_slot_populated_native(inst, i).neq(0)) as (then, otherwise):
                with then:
                    self.convert_getkey_by_index_unsafe(context, inst, i).convert_destroy()
                    self.convert_getvalue_by_index_unsafe(context, inst, i).convert_destroy()

        context.pushEffect(
            runtime_functions.free.call(inst.nonref_expr.ElementPtrIntegers(0, 1).load().cast(native_ast.UInt8Ptr)) >>
            runtime_functions.free.call(inst.nonref_expr.ElementPtrIntegers(0, 2).load().cast(native_ast.UInt8Ptr)) >>
            runtime_functions.free.call(inst.nonref_expr.ElementPtrIntegers(0, 5).load().cast(native_ast.UInt8Ptr)) >>
            runtime_functions.free.call(inst.nonref_expr.ElementPtrIntegers(0, 6).load().cast(native_ast.UInt8Ptr)) >>
            runtime_functions.free.call(inst.nonref_expr.cast(native_ast.UInt8Ptr))
        )

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


class DictMakeIteratorWrapper(DictWrapperBase):
    def convert_method_call(self, context, expr, methodname, args, kwargs):
        if methodname == "__iter__" and not args and not kwargs:
            res = context.push(
                # self.iteratorType is inherited from our specialized children
                # who pick whether we're an interator over keys, values, items, etc.
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


class DictKeysWrapper(DictMakeIteratorWrapper):
    def __init__(self, dictType):
        super().__init__(dictType, "keys")
        self.iteratorType = DictKeysIteratorWrapper(dictType)


class DictValuesWrapper(DictMakeIteratorWrapper):
    def __init__(self, dictType):
        super().__init__(dictType, "values")
        self.iteratorType = DictValuesIteratorWrapper(dictType)


class DictItemsWrapper(DictMakeIteratorWrapper):
    def __init__(self, dictType):
        super().__init__(dictType, "items")
        self.iteratorType = DictItemsIteratorWrapper(dictType)


class DictIteratorWrapper(Wrapper):
    is_pod = False
    is_empty = False
    is_pass_by_ref = True

    def __init__(self, dictType, iteratorType):
        self.dictType = dictType
        self.iteratorType = iteratorType
        super().__init__((dictType, "iterator", iteratorType))

    def getNativeLayoutType(self):
        return native_ast.Type.Struct(
            element_types=(("pos", native_ast.Int64), ("dict", DictWrapper(self.dictType).getNativeLayoutType())),
            name="const_dict_iterator"
        )

    def convert_next(self, context, expr):
        nextSlotIx = context.call_py_function(dict_next_slot, (self.refAs(context, expr, 1), self.refAs(context, expr, 0)), {})

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
                self.dictType,
                expr.expr
                    .ElementPtrIntegers(0, 1)
                    .cast(DictWrapper(self.dictType).getNativeLayoutType().pointer())
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


class DictKeysIteratorWrapper(DictIteratorWrapper):
    def __init__(self, dictType):
        super().__init__(dictType, "keys")

    def iteratedItemForReference(self, context, expr, ixExpr):
        return DictWrapper(self.dictType).convert_method_call(
            context,
            self.refAs(context, expr, 1),
            "getKeyByIndexUnsafe",
            (ixExpr,),
            {}
        )


class DictItemsIteratorWrapper(DictIteratorWrapper):
    def __init__(self, dictType):
        super().__init__(dictType, "items")

    def iteratedItemForReference(self, context, expr, ixExpr):
        return DictWrapper(self.dictType).convert_method_call(
            context,
            self.refAs(context, expr, 1),
            "getItemByIndexUnsafe",
            (ixExpr,),
            {}
        )


class DictValuesIteratorWrapper(DictIteratorWrapper):
    def __init__(self, dictType):
        super().__init__(dictType, "values")

    def iteratedItemForReference(self, context, expr, ixExpr):
        return DictWrapper(self.dictType).convert_method_call(
            context,
            self.refAs(context, expr, 1),
            "getValueByIndexUnsafe",
            (ixExpr,),
            {}
        )
