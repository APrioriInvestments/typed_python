/******************************************************************************
   Copyright 2017-2020 typed_python Authors

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
******************************************************************************/

#pragma once

#include <cstring>

// for Dict, items would be key, value pairs
// for Set, items would be keys
class hash_table_layout {
  public:
    hash_table_layout()
        : refcount(0)
        , items(nullptr)
        , items_populated(nullptr)
        , items_reserved(0)
        , top_item_slot(0)
        , hash_table_slots(nullptr)
        , hash_table_hashes(nullptr)
        , hash_table_size(0)
        , hash_table_count(0)
        , hash_table_empty_slots(0) {}

    // We follow the same hashing scheme as the internals of pythons dictionaries
    // as detailed here:
    //      https://hg.python.org/cpython/file/52f68c95e025/Objects/dictobject.c#l296

    enum { EMPTY = -1, DELETED = -2, PERTURB_SHIFT = 5, MIN_SIZE = 8 };

    void setTo(int32_t* ptr, int32_t value, size_t count) {
        for (size_t k = 0; k < count; k++) {
            *(ptr++) = value;
        }
    }

    // return the index of the object indexed by 'hash', or -1
    template <class eq_func>
    int32_t find(int32_t item_size, typed_python_hash_type hash, const eq_func& compare) {
        if (!hash_table_slots) {
            return -1;
        }

        if (hash < 0) {
            hash = -hash;
        }

        uint64_t mask = hash_table_size - 1;
        uint64_t perturb = hash;
        uint64_t offset = hash;

        while (true) {
            // slot is empty
            int32_t slot = hash_table_slots[offset & mask];

            if (slot == EMPTY) {
                return -1;
            }

            if (slot != DELETED && hash_table_hashes[offset & mask] == hash
                && compare(items + item_size * slot)) {
                return slot;
            }

            offset = (offset << 2) + offset + perturb + 1;
            perturb >>= PERTURB_SHIFT;
        }
    }

    // add an item to the hash table
    void add(typed_python_hash_type hash, int32_t slot) {
        if (hash_table_count * 2 + 1 > hash_table_size
            || hash_table_empty_slots < hash_table_size / 4 + 1) {
            resizeTable();
        }

        if (hash < 0) {
            hash = -hash;
        }

        uint64_t mask = hash_table_size - 1;
        uint64_t perturb = hash;
        uint64_t offset = hash;

        while (true) {
            if (hash_table_slots[offset & mask] == EMPTY || hash_table_slots[offset & mask] == DELETED) {
                if (hash_table_slots[offset & mask] == EMPTY) {
                    hash_table_empty_slots--;
                }

                hash_table_slots[offset & mask] = slot;
                hash_table_hashes[offset & mask] = hash;
                items_populated[slot] = 1;
                hash_table_count++;
                return;
            }

            offset = (offset << 2) + offset + perturb + 1;
            perturb >>= PERTURB_SHIFT;
        }
    }

    // remove an item with the given hash. returning the item slot where it
    // lived.
    //-1 if not found
    template <class eq_func>
    int32_t remove(int32_t item_size, typed_python_hash_type hash, const eq_func& compare) {
        if (!hash_table_slots) {
            return -1;
        }

        if (items_reserved > (hash_table_count + 2) * 4) {
            compressItemTable(item_size);
        }

        // compress the hashtable if it's really empty
        if (hash_table_count < hash_table_size / 8) {
            resizeTable();
        }

        if (hash < 0) {
            hash = -hash;
        }

        uint64_t mask = hash_table_size - 1;
        uint64_t perturb = hash;
        uint64_t offset = hash;

        while (true) {
            int32_t slot = hash_table_slots[offset & mask];

            if (slot == EMPTY) {
                // we never found the item
                return -1;
            }

            if (slot != DELETED && compare(items + item_size * slot)) {
                items_populated[slot] = 0;

                hash_table_slots[offset & mask] = DELETED;
                hash_table_hashes[offset & mask] = -1;
                hash_table_count -= 1;

                return slot;
            }

            offset = (offset << 2) + offset + perturb + 1;
            perturb >>= PERTURB_SHIFT;
        }
    }

    void compressItemTable(size_t item_size) {
        std::vector<int32_t> newItemPositions;
        int32_t count_so_far = 0;

        for (long k = 0; k < items_reserved; k++) {
            if (items_populated[k]) {
                newItemPositions.push_back(count_so_far);

                if (k != count_so_far) {
                    items_populated[count_so_far] = 1;
                    items_populated[k] = 0;

                    memcpy(items + item_size * count_so_far, items + item_size * k,
                           item_size);
                }

                count_so_far++;
            } else {
                newItemPositions.push_back(-1);
            }
        }

        items_populated = (uint8_t*)tp_realloc(items_populated, items_reserved, count_so_far);
        items = (uint8_t*)tp_realloc(items, items_reserved * item_size, count_so_far * item_size);

        items_reserved = count_so_far;
        top_item_slot = items_reserved;

        for (long k = 0; k < hash_table_size; k++) {
            if (hash_table_slots[k] >= 0) {
                if (hash_table_slots[k] >= newItemPositions.size()) {
                    throw std::runtime_error("corrupt slot");
                }

                hash_table_slots[k] = newItemPositions[hash_table_slots[k]];

                if (hash_table_slots[k] < 0) {
                    throw std::runtime_error("invalid slot");
                }
            }
        }

        for (long k = 0; k < hash_table_size; k++) {
            if (hash_table_slots[k] >= 0) {
                if (hash_table_slots[k] >= items_reserved) {
                    throw std::runtime_error("failed during compression");
                }
            }
        }
    }

    int32_t allocateNewSlot(size_t item_size) {
        if (!items) {
            items_reserved = 4;
            items = (uint8_t*)tp_malloc(items_reserved * item_size);
            std::memset(items, 0, items_reserved * item_size);
            items_populated = (uint8_t*)tp_malloc(items_reserved);
            std::memset(items_populated, 0, items_reserved);
            top_item_slot = 0;
            for (long k = 0; k < items_reserved; k++) {
                items_populated[k] = 0;
            }
        }

        while (top_item_slot >= items_reserved) {
            size_t old_reserved = items_reserved;
            items_reserved = items_reserved * 1.25 + 1;
            items = (uint8_t*)tp_realloc(items, item_size * old_reserved, item_size * items_reserved);
            items_populated = (uint8_t*)tp_realloc(items_populated, old_reserved, items_reserved);

            for (long k = old_reserved; k < items_reserved; k++) {
                items_populated[k] = 0;
            }
        }

        return top_item_slot++;
    }

    int32_t pickHashTableSize(int32_t minSize) {
        int32_t ct = MIN_SIZE;
        while (ct < minSize) {
            ct <<= 1;
        }
        return ct;
    }

    // called after we have deleted everything that's populated, and need to
    // zero out the hash_table's internals.
    void allItemsHaveBeenRemoved() {
        if (!hash_table_slots) {
            return;
        }

        hash_table_count = 0;
        top_item_slot = 0;
        hash_table_empty_slots = hash_table_size;

        setTo(hash_table_slots, EMPTY, hash_table_size);
        setTo(hash_table_hashes, EMPTY, hash_table_size);
        std::memset(items_populated, 0, items_reserved);
    }

    template <class copy_constructor_type>
    hash_table_layout* copyTable(size_t item_size, bool isPOD, const copy_constructor_type& copy_constructor) {
        hash_table_layout* result = (hash_table_layout*)tp_malloc(sizeof(hash_table_layout));
        new (result) hash_table_layout;

        result->refcount = 1;
        if (!items) {
            return result;
        }
        result->items_reserved = items_reserved;
        result->top_item_slot = top_item_slot;
        result->hash_table_size = hash_table_size;
        result->hash_table_count = hash_table_count;
        result->hash_table_empty_slots = hash_table_empty_slots;

        result->items = (uint8_t*)tp_malloc(item_size * items_reserved);
        if (isPOD) {
            memcpy(result->items, items, item_size * items_reserved);
        }
        else {
            for (int i=0; i<items_reserved; i++) {
                if (items_populated[i]) {
                    copy_constructor(result->items + item_size * i, items + item_size * i);
                }
             }
        }

        result->items_populated = (uint8_t*)tp_malloc(items_reserved);
        memcpy(result->items_populated, items_populated, items_reserved);

        result->hash_table_slots = (int32_t*)tp_malloc(hash_table_size * sizeof(int32_t));
        memcpy(result->hash_table_slots, hash_table_slots, hash_table_size * sizeof(int32_t));

        result->hash_table_hashes = (typed_python_hash_type*)tp_malloc(hash_table_size * sizeof(typed_python_hash_type));
        memcpy(result->hash_table_hashes, hash_table_hashes, hash_table_size * sizeof(typed_python_hash_type));
        return result;
    }

    void resizeTable() {
        if (!hash_table_slots) {
            hash_table_size = pickHashTableSize(hash_table_count * 4);
            hash_table_slots = (int32_t*)tp_malloc(hash_table_size * sizeof(int32_t));
            setTo(hash_table_slots, EMPTY, hash_table_size);
            hash_table_hashes = (typed_python_hash_type*)tp_malloc(hash_table_size * sizeof(typed_python_hash_type));
            setTo(hash_table_hashes, EMPTY, hash_table_size);
            hash_table_count = 0;
            hash_table_empty_slots = hash_table_size;

        } else {
            int32_t oldSize = hash_table_size;
            int32_t* oldSlots = hash_table_slots;
            typed_python_hash_type* oldHashes = hash_table_hashes;

            // make sure the table's not too small
            hash_table_size = pickHashTableSize(hash_table_count * 4);

            hash_table_slots = (int32_t*)tp_malloc(hash_table_size * sizeof(int32_t));
            setTo(hash_table_slots, EMPTY, hash_table_size);
            hash_table_hashes =
              (typed_python_hash_type*)tp_malloc(hash_table_size * sizeof(typed_python_hash_type));
            setTo(hash_table_hashes, EMPTY, hash_table_size);
            hash_table_count = 0;
            hash_table_empty_slots = hash_table_size;

            for (long k = 0; k < oldSize; k++) {
                if (oldSlots[k] != EMPTY && oldSlots[k] != DELETED) {
                    add(oldHashes[k], oldSlots[k]);
                }
            }

            tp_free(oldSlots);
            tp_free(oldHashes);
        }
    }

    void prepareForDeserialization(uint32_t slotCount, size_t item_size) {
        if (hash_table_size) {
            throw std::runtime_error("deserialization prepare should only be called on "
                                     "empty tables");
        }

        items_reserved = slotCount;
        items_populated = (uint8_t*)tp_malloc(slotCount);
        items = (uint8_t*)tp_malloc(slotCount * item_size);

        for (long k = 0; k < items_reserved; k++) {
            items_populated[k] = true;
        }

        top_item_slot = items_reserved;
    }

    template <class hash_fun_type>
    void buildHashTableAfterDeserialization(size_t item_size, const hash_fun_type& hash_fun) {
        hash_table_size = pickHashTableSize(items_reserved * 2);
        hash_table_slots = (int32_t*)tp_malloc(hash_table_size * sizeof(int32_t));
        hash_table_hashes =
          (typed_python_hash_type*)tp_malloc(hash_table_size * sizeof(typed_python_hash_type));
        hash_table_count = 0;
        hash_table_empty_slots = hash_table_size;

        for (long k = 0; k < hash_table_size; k++) {
            hash_table_slots[k] = EMPTY;
            hash_table_hashes[k] = -1;
        }

        for (long k = 0; k < items_reserved; k++) {
            add(hash_fun(items + item_size * k), k);
        }
    }

    hash_table_layout* deepcopy(
        DeepcopyContext& context,
        Type* dictOrSetType,
        Type* keyType,
        Type* valueType
    ) {
        hash_table_layout* dest = (hash_table_layout*)context.slab->allocate(sizeof(hash_table_layout), dictOrSetType);

        new (dest) hash_table_layout();

        dest->refcount = 1;

        int bytesPerKVPair = keyType->bytecount() + (valueType ? valueType->bytecount() : 0);

        dest->items = (uint8_t*)context.slab->allocate(
            bytesPerKVPair * this->items_reserved,
            nullptr
        );

        if (keyType->isPOD() && (!valueType || valueType->isPOD())) {
            memcpy(
                dest->items,
                this->items,
                this->items_reserved * bytesPerKVPair
            );
        } else {
            for (long k = 0; k < this->items_reserved; k++) {
                if (this->items_populated[k]) {
                    keyType->deepcopy(
                        dest->items + bytesPerKVPair * k,
                        this->items + bytesPerKVPair * k,
                        context
                    );
                    if (valueType) {
                        valueType->deepcopy(
                            dest->items + bytesPerKVPair * k + keyType->bytecount(),
                            this->items + bytesPerKVPair * k + keyType->bytecount(),
                            context
                        );
                    }
                };
            }
        }

        dest->items_populated = (uint8_t*)context.slab->allocate(
            this->items_reserved,
            nullptr
        );
        memcpy(
            dest->items_populated,
            this->items_populated,
            this->items_reserved
        );

        dest->items_reserved = this->items_reserved;

        dest->top_item_slot = this->top_item_slot;

        dest->hash_table_count = this->hash_table_count;
        dest->hash_table_size = this->hash_table_size;
        dest->hash_table_empty_slots = this->hash_table_empty_slots;

        dest->hash_table_slots = (int32_t*)context.slab->allocate(sizeof(int32_t) * this->hash_table_size, nullptr);
        memcpy(
            dest->hash_table_slots,
            this->hash_table_slots,
            sizeof(int32_t) * this->hash_table_size
        );

        dest->hash_table_hashes = (typed_python_hash_type*)context.slab->allocate(
            sizeof(typed_python_hash_type) * this->hash_table_size,
            nullptr
        );
        memcpy(
            dest->hash_table_hashes,
            this->hash_table_hashes,
            sizeof(typed_python_hash_type) * this->hash_table_size
        );

        return dest;
    }


    void checkInvariants(std::string reason) {
        int64_t popCount = 0;
        for (long k = 0; k < items_reserved; k++) {
            if (items_populated[k]) {
                popCount++;
                if (top_item_slot <= k) {
                    throw std::runtime_error(reason
                                             + ": top item slot should be greater "
                                               "than all populated items");
                }
            }
        }

        if (popCount != hash_table_count) {
            throw std::runtime_error(reason
                                     + ": populated item count is "
                                       "not the same as the "
                                       "hashtable count");
        }

        int64_t filledSlots = 0;
        int64_t deletedSlots = 0;
        for (long k = 0; k < hash_table_size; k++) {
            if (hash_table_slots[k] == DELETED) {
                deletedSlots++;
            } else if (hash_table_slots[k] != EMPTY) {
                filledSlots++;

                if (hash_table_slots[k] >= items_reserved) {
                    throw std::runtime_error(reason
                                             + ": hash table has slot entry out "
                                               "of bounds with item list");
                }

                if (!items_populated[hash_table_slots[k]]) {
                    throw std::runtime_error(reason
                                             + ": hash table points to unmarked "
                                               "slot");
                }
            }
        }

        if (filledSlots != hash_table_count) {
            throw std::runtime_error(reason
                                     + ": Filled slot count is not "
                                       "the same as the hashtable's "
                                       "known count");
        }

        if (hash_table_size - filledSlots - deletedSlots != hash_table_empty_slots) {
            throw std::runtime_error(reason + ": empty slot count is not consistent");
        }
    }

    bool empty() const { return hash_table_count == 0; }
    int32_t size() const { return hash_table_count; }

    std::atomic<int64_t> refcount;

    uint8_t* items; // packed set of key_value pairs.
    uint8_t* items_populated; // array of bool for whether populated
    size_t items_reserved; // count of items reserved
    size_t top_item_slot; // index of the next item slot to use

    int32_t* hash_table_slots; // a hashtable. each actual object hash to
                               // the slot index it holds. -1 if not
                               // populated.
    typed_python_hash_type* hash_table_hashes; // a hashtable. each actual object hash to
                                               // the hash in that part of the table. -1 if
                                               // not populated.
    size_t hash_table_size; // size of the table
    size_t hash_table_count; // populated count of the table
    size_t hash_table_empty_slots; // slots that are not empty in the
                                   // table. Recall that some slots are
                                   // 'deleted'
};


typedef hash_table_layout* hash_table_layout_ptr;
