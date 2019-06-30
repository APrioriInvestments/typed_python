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

from typed_python import NoneType, Tuple, PointerTo, Int32, Int64

import nativepython.native_ast as native_ast
import nativepython


typeWrapper = lambda t: nativepython.python_object_representation.typedPythonTypeToTypeWrapper(t)

EMPTY = -1
DELETED = -2


def dict_slot_for_key(instance, item):
    slots = instance._hash_table_slots

    if not slots:
        return -1

    itemHash = hash(item)

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
    slot = dict_slot_for_key(instance, item)

    if slot == -1:
        raise Exception("Key doesn't exist.")

    return instance.getValueByIndexUnsafe(slot)


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
        if attr in ("getItemByIndexUnsafe", "getKeyByIndexUnsafe", "getValueByIndexUnsafe"):
            return expr.changeType(BoundCompiledMethodWrapper(self, attr))

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

        return super().convert_attribute(context, expr, attr)

    def convert_method_call(self, context, instance, methodname, args, kwargs):
        if kwargs:
            return super().convert_method_call(context, instance, methodname, args, kwargs)

        if len(args) == 1:
            item = args[0]

            if item.expr_type.typeRepresentation is Int64:
                if methodname in ("getItemByIndexUnsafe", "getKeyByIndexUnsafe", "getValueByIndexUnsafe"):
                    item = context.pushReference(
                        self.itemType,
                        instance.nonref_expr.ElementPtrIntegers(0, 1).load().cast(
                            self.itemType.getNativeLayoutType().pointer()
                        ).elemPtr(item.toInt64().nonref_expr)
                    )

                    if methodname == "getItemByIndexUnsafe":
                        return item
                    elif methodname == "getKeyByIndexUnsafe":
                        # take the first item in the tuple
                        return item.expr_type.refAs(context, item, 0)
                    else:
                        # take the second item in the tuple
                        return item.expr_type.refAs(context, item, 1)

        return super().convert_method_call(context, instance, methodname, args, kwargs)

    def convert_getitem(self, context, expr, item):
        if item is None or expr is None:
            return None

        item = item.convert_to_type(self.keyType)
        if item is None:
            return None

        return context.call_py_function(dict_getitem, (expr, item), {})

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
