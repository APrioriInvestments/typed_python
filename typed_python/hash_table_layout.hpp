/******************************************************************************
   Copyright 2017-2019 typed_python Authors

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

    enum { EMPTY = -1, DELETED = -2 };

    void setTo(int32_t* ptr, int32_t value, size_t count) {
        for (size_t k = 0; k < count; k++) {
            *(ptr++) = value;
        }
    }

    // return the index of the object indexed by 'hash', or -1
    template <class eq_func>
    int32_t find(int32_t kv_pair_size, typed_python_hash_type hash, const eq_func& compare) {
        if (!hash_table_slots) {
            return -1;
        }

        // because typed_python has to use python's mod, which is
        // different than c++'s mod for negative numbers, we just map
        // negatives to positives.
        if (hash < 0) {
            hash = -hash;
        }

        int32_t offset = hash % hash_table_size;

        while (true) {
            // slot is empty
            int32_t slot = hash_table_slots[offset];

            if (slot == EMPTY) {
                return -1;
            }

            if (slot != DELETED && hash_table_hashes[offset] == hash
                && compare(items + kv_pair_size * slot)) {
                return slot;
            }

            offset = nextOffset(offset);
        }
    }

    // linear search
    int32_t nextOffset(int32_t offset) const {
        offset += 1;
        if (offset >= hash_table_size) {
            offset = 0;
        }
        return offset;
    }

    // add an item to the hash table
    void add(typed_python_hash_type hash, int32_t slot) {
        if (hash_table_count * 2 + 1 > hash_table_size
            || hash_table_empty_slots < hash_table_size / 4 + 1) {
            resizeTable();
        }

        // because typed_python has to use python's mod, which is
        // different than c++'s mod for negative numbers, we just map
        // negatives to positives.
        if (hash < 0) {
            hash = -hash;
        }

        int32_t offset = hash % hash_table_size;
        while (true) {
            if (hash_table_slots[offset] == EMPTY || hash_table_slots[offset] == DELETED) {
                if (hash_table_slots[offset] == EMPTY) {
                    hash_table_empty_slots--;
                }

                hash_table_slots[offset] = slot;
                hash_table_hashes[offset] = hash;
                items_populated[slot] = 1;
                hash_table_count++;
                return;
            }

            offset = nextOffset(offset);
        }
    }

    // remove an item with the given hash. returning the item slot where it
    // lived.
    //-1 if not found
    template <class eq_func>
    int32_t remove(int32_t kv_pair_size, typed_python_hash_type hash, const eq_func& compare) {
        if (!hash_table_slots) {
            return -1;
        }

        if (items_reserved > (hash_table_count + 2) * 4) {
            compressItemTable(kv_pair_size);
        }

        // compress the hashtable if it's really empty
        if (hash_table_count < hash_table_size / 8) {
            resizeTable();
        }

        // because typed_python has to use python's mod, which is
        // different than c++'s mod for negative numbers, we just map
        // negatives to positives.
        if (hash < 0) {
            hash = -hash;
        }

        int32_t offset = hash % hash_table_size;

        while (true) {
            int32_t slot = hash_table_slots[offset];

            if (slot == EMPTY) {
                // we never found the item
                return -1;
            }

            if (slot != DELETED && compare(items + kv_pair_size * slot)) {
                items_populated[slot] = 0;

                hash_table_slots[offset] = DELETED;
                hash_table_hashes[offset] = -1;
                hash_table_count -= 1;

                return slot;
            }

            offset = nextOffset(offset);
        }
    }

    void compressItemTable(size_t kv_pair_size) {
        std::vector<int32_t> newItemPositions;
        int32_t count_so_far = 0;

        for (long k = 0; k < items_reserved; k++) {
            if (items_populated[k]) {
                newItemPositions.push_back(count_so_far);

                if (k != count_so_far) {
                    items_populated[count_so_far] = 1;
                    items_populated[k] = 0;

                    memcpy(items + kv_pair_size * count_so_far, items + kv_pair_size * k,
                           kv_pair_size);
                }

                count_so_far++;
            } else {
                newItemPositions.push_back(-1);
            }
        }

        items_reserved = count_so_far;
        items_populated = (uint8_t*)realloc(items_populated, count_so_far);
        items = (uint8_t*)realloc(items, count_so_far * kv_pair_size);
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

    int32_t allocateNewSlot(size_t kv_pair_size) {
        if (!items) {
            items = (uint8_t*)malloc(4 * kv_pair_size);
            std::memset(items, 0, 4 * kv_pair_size);
            items_populated = (uint8_t*)malloc(4);
            std::memset(items_populated, 0, 4);
            items_reserved = 4;
            top_item_slot = 0;
            for (long k = 0; k < items_reserved; k++) {
                items_populated[k] = 0;
            }
        }

        while (top_item_slot >= items_reserved) {
            size_t old_reserved = items_reserved;
            items_reserved = items_reserved * 1.25 + 1;
            items = (uint8_t*)realloc(items, kv_pair_size * items_reserved);
            items_populated = (uint8_t*)realloc(items_populated, items_reserved);

            for (long k = old_reserved; k < items_reserved; k++) {
                items_populated[k] = 0;
            }
        }

        return top_item_slot++;
    }

    int32_t computeNextPrime(int32_t p) {
        static std::vector<int32_t> primes;
        if (!primes.size()) {
            primes.push_back(2);
        }

        auto isprime = [&](int32_t candidate) {
            for (auto d : primes) {
                if (candidate % d == 0) {
                    return false;
                }
                if (d * d > candidate) {
                    return true;
                }
            }
            throw std::logic_error("Expected to clear the primes list.");
        };

        while (true) {
            while (primes.back() * primes.back() < p) {
                int32_t cur = primes.back() + 1;
                while (!isprime(cur)) {
                    cur++;
                }
                primes.push_back(cur);
            }

            if (isprime(p)) {
                return p;
            }

            p++;
        }
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
    }

    void resizeTable() {
        if (!hash_table_slots) {
            hash_table_slots = (int32_t*)malloc(7 * sizeof(int32_t));
            setTo(hash_table_slots, EMPTY, 7);
            hash_table_hashes = (typed_python_hash_type*)malloc(7 * sizeof(typed_python_hash_type));
            setTo(hash_table_hashes, EMPTY, 7);
            hash_table_size = 7;
            hash_table_count = 0;
            hash_table_empty_slots = hash_table_size;

        } else {
            int32_t oldSize = hash_table_size;
            int32_t* oldSlots = hash_table_slots;
            typed_python_hash_type* oldHashes = hash_table_hashes;

            // make sure the table's not too small
            hash_table_size = computeNextPrime(hash_table_count * 4 + 7);

            hash_table_slots = (int32_t*)malloc(hash_table_size * sizeof(int32_t));
            setTo(hash_table_slots, EMPTY, hash_table_size);
            hash_table_hashes =
              (typed_python_hash_type*)malloc(hash_table_size * sizeof(typed_python_hash_type));
            setTo(hash_table_hashes, EMPTY, hash_table_size);
            hash_table_count = 0;
            hash_table_empty_slots = hash_table_size;

            for (long k = 0; k < oldSize; k++) {
                if (oldSlots[k] != EMPTY && oldSlots[k] != DELETED) {
                    add(oldHashes[k], oldSlots[k]);
                }
            }

            free(oldSlots);
            free(oldHashes);
        }
    }

    void prepareForDeserialization(uint32_t slotCount, size_t kv_pair_size) {
        if (hash_table_size) {
            throw std::runtime_error("deserialization prepare should only be called on "
                                     "empty tables");
        }

        items_reserved = slotCount;
        items_populated = (uint8_t*)malloc(slotCount);
        items = (uint8_t*)malloc(slotCount * kv_pair_size);

        for (long k = 0; k < items_reserved; k++) {
            items_populated[k] = true;
        }

        top_item_slot = items_reserved;
    }

    template <class hash_fun_type>
    void buildHashTableAfterDeserialization(size_t kv_pair_size, const hash_fun_type& hash_fun) {
        hash_table_size = computeNextPrime(items_reserved * 2.5 + 7);
        hash_table_slots = (int32_t*)malloc(hash_table_size * sizeof(int32_t));
        hash_table_hashes =
          (typed_python_hash_type*)malloc(hash_table_size * sizeof(typed_python_hash_type));
        hash_table_count = 0;
        hash_table_empty_slots = hash_table_size;

        for (long k = 0; k < hash_table_size; k++) {
            hash_table_slots[k] = EMPTY;
            hash_table_hashes[k] = -1;
        }

        for (long k = 0; k < items_reserved; k++) {
            add(hash_fun(items + kv_pair_size * k), k);
        }
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
