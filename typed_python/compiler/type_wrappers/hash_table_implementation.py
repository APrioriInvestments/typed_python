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

from typed_python import UInt64, Int32, Type
from typed_python.compiler.type_wrappers.compilable_builtin import CompilableBuiltin
from typed_python.compiler.conversion_level import ConversionLevel


EMPTY = -1
DELETED = -2
PERTURB_SHIFT = 5


class NativeHash(CompilableBuiltin):
    """A function for directly hashing a typed python value.

    Note that this is not the same as calling 'hash' from compiled code,
    which will produce different answers from python right now, and which
    is attempting at some level to mimic python's standard hash functionality.

    This is because our 'hash' function is actually a little messed up right
    now - it depends on how TP is holding a value. For instance, if we have
    a ListOf(object) and put a float in, we'll get a different hash than
    if we had a ListOf(float) and put that same float in, because there is no
    formal 'TP' hash. We should actually attempt to unify the two hashing
    models since there's not a good reason to have two of them.

    That functionality is not used internally to our datastructures (we
    allow things to hash to -1) so you can't just use 'hash' when implementing
    wrappers for tp internals.
    """
    def __init__(self, T):
        self.T = T

    def __eq__(self, other):
        return isinstance(other, NativeHash) and self.T == other.T

    def __hash__(self):
        return hash(("NativeHash", self.T))

    def convert_call(self, context, instance, args, kwargs):
        if len(args) == 1:
            if not issubclass(self.T, Type) and self.T not in (
                bool, int, float, str, bytes, type(None)
            ):
                res = context.constant(hash).convert_call([args[0]], {})

                if res is None:
                    return res

                # first convert it to an int32 since this is what
                # Dict(object, object) will do in the interpreter
                res = res.convert_to_type(
                    Int32, ConversionLevel.Implicit
                )

                # then convert it to UInt64 since that's what we're
                # using in this representation
                res = res.convert_to_type(
                    UInt64, ConversionLevel.Implicit
                )

                return res

            return args[0].convert_hash()

        return super().convert_call(context, instance, args, kwargs)

    @staticmethod
    def callHash(T, x):
        from typed_python import Entrypoint

        @Entrypoint
        def hashIt(x):
            return NativeHash(T)(x)

        return hashIt(x)


def table_add_slot(instance, itemHash, slot):
    if (instance._hash_table_count * 2 + 1 > instance._hash_table_size or
            instance._hash_table_empty_slots < (instance._hash_table_size >> 2) + 1):
        instance._resizeTableUnsafe()

    if itemHash < 0:
        itemHash = -itemHash

    mask = UInt64(instance._hash_table_size) - UInt64(1)
    perturb = UInt64(itemHash)
    offset = UInt64(itemHash)

    while True:
        if (
            instance._hash_table_slots[offset & mask] == EMPTY
            or instance._hash_table_slots[offset & mask] == DELETED
        ):
            if instance._hash_table_slots[offset & mask] == EMPTY:
                instance._hash_table_empty_slots -= 1

            instance._hash_table_slots[offset & mask] = slot
            instance._hash_table_hashes[offset & mask] = itemHash
            instance._items_populated[slot] = 1
            instance._hash_table_count += 1

            return

        offset = UInt64(offset << 2) + offset + perturb + UInt64(1)
        perturb = UInt64(perturb >> PERTURB_SHIFT)


def table_slot_for_key(instance, itemHash, item):
    slots = instance._hash_table_slots

    if not slots:
        return -1

    if itemHash < 0:
        itemHash = -itemHash

    mask = UInt64(instance._hash_table_size) - UInt64(1)
    perturb = UInt64(itemHash)
    offset = UInt64(itemHash)

    while True:
        slotIndex = int((slots + (offset & mask)).get())

        if slotIndex == EMPTY:
            return -1

        if slotIndex != DELETED and (instance._hash_table_hashes + (offset & mask)).get() == itemHash:
            if instance.getKeyByIndexUnsafe(slotIndex) == item:
                return slotIndex

        offset = UInt64(offset << 2) + offset + perturb + UInt64(1)
        perturb = UInt64(perturb >> PERTURB_SHIFT)

    # not necessary, but currently we don't realize that the while loop
    # never exits, and so we think there's a possibility we return None
    return 0


def table_next_slot(instance, slotIx):
    slotIx += 1

    while slotIx < instance._items_reserved:
        if instance._items_populated[slotIx]:
            return slotIx
        slotIx += 1

    return -1


def table_remove_key(instance, item, itemHash, raises):
    if instance._items_reserved > (instance._hash_table_count + 2) * 4:
        instance._compressItemTableUnsafe()

    if instance._hash_table_count < instance._hash_table_size >> 3:
        instance._resizeTableUnsafe()

    slots = instance._hash_table_slots

    if not slots:
        if raises:
            raise KeyError(item)
        else:
            return 0

    if itemHash < 0:
        itemHash = -itemHash

    mask = UInt64(instance._hash_table_size) - UInt64(1)
    perturb = UInt64(itemHash)
    offset = UInt64(itemHash)

    while True:
        slotIndex = int((slots + (offset & mask)).get())

        if slotIndex == EMPTY:
            if raises:
                raise KeyError(item)
            else:
                return 0

        if slotIndex != DELETED and (instance._hash_table_hashes + (offset & mask)).get() == itemHash:
            if instance.getKeyByIndexUnsafe(slotIndex) == item:
                instance._hash_table_hashes[offset & mask] = -1
                instance._hash_table_slots[offset & mask] = DELETED
                instance._hash_table_count -= 1
                instance._items_populated[slotIndex] = 0

                instance.deleteItemByIndexUnsafe(slotIndex)
                return

        offset = (offset << 2) + offset + perturb + UInt64(1)
        perturb = perturb >> PERTURB_SHIFT

    # not necessary, but currently we don't currently realize that the while loop
    # never exits, and so we think there's a possibility we return None
    return 0


def table_clear(instance):
    slotIx = 0

    while slotIx < instance._items_reserved:
        if instance._items_populated[slotIx]:
            instance.deleteItemByIndexUnsafe(slotIx)
            instance._items_populated[slotIx] = 0

        slotIx += 1

    for i in range(instance._hash_table_size):
        instance._hash_table_hashes[i] = EMPTY
        instance._hash_table_slots[i] = -1

    instance._hash_table_count = 0
    instance._hash_table_empty_slots = instance._hash_table_size
    instance._top_item_slot = 0


# Operations specific to dicts that manipulate the fields directly:


def dict_table_contains(instance, item):
    itemHash = NativeHash(instance.KeyType)(item)

    slot = table_slot_for_key(instance, itemHash, item)

    return slot != -1


def dict_delitem(instance, item):
    itemHash = NativeHash(instance.KeyType)(item)

    table_remove_key(instance, item, itemHash, True)


def dict_getitem(instance, item):
    itemHash = NativeHash(instance.KeyType)(item)

    slot = table_slot_for_key(instance, itemHash, item)

    if slot == -1:
        raise KeyError(item)

    return instance.getValueByIndexUnsafe(slot)


def dict_get(instance, item, default):
    itemHash = NativeHash(instance.KeyType)(item)

    slot = table_slot_for_key(instance, itemHash, item)

    if slot == -1:
        return default

    return instance.getValueByIndexUnsafe(slot)


def dict_setitem(instance, key, value):
    itemHash = NativeHash(instance.KeyType)(key)

    slot = table_slot_for_key(instance, itemHash, key)

    if slot == -1:
        newSlot = instance._allocateNewSlotUnsafe()
        table_add_slot(instance, itemHash, newSlot)
        instance.initializeKeyByIndexUnsafe(newSlot, key)
        instance.initializeValueByIndexUnsafe(newSlot, value)
    else:
        instance.assignValueByIndexUnsafe(slot, value)


# Operations specific to sets that manipulate the fields directly:

def set_table_contains(instance, item):
    itemHash = NativeHash(instance.ElementType)(item)

    slot = table_slot_for_key(instance, itemHash, item)

    return slot != -1


def set_add(instance, key):
    itemHash = NativeHash(instance.ElementType)(key)

    slot = table_slot_for_key(instance, itemHash, key)

    if slot == -1:
        newSlot = instance._allocateNewSlotUnsafe()
        table_add_slot(instance, itemHash, newSlot)
        instance.initializeKeyByIndexUnsafe(newSlot, key)


def set_add_or_remove(instance, key):
    itemHash = NativeHash(instance.ElementType)(key)

    slot = table_slot_for_key(instance, itemHash, key)

    if slot == -1:
        newSlot = instance._allocateNewSlotUnsafe()
        table_add_slot(instance, itemHash, newSlot)
        instance.initializeKeyByIndexUnsafe(newSlot, key)
    else:
        table_remove_key(instance, key, itemHash, False)


def set_remove(instance, key):
    itemHash = NativeHash(instance.ElementType)(key)

    table_remove_key(instance, key, itemHash, True)


def set_discard(instance, key):
    itemHash = NativeHash(instance.ElementType)(key)

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
