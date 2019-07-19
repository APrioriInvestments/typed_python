#pragma once

#include <cstring>
#include <functional>

#include "../hash_table_layout.hpp"

template <typename T>
class HashTableLayout {
  public:
    inline static constexpr size_t byte_count_per_el = sizeof(void*);

    HashTableLayout() { new_hash_table(&table); }

    ~HashTableLayout() {
        free(table->items);
        free(table->items_populated);
        free(table->hash_table_slots);
        free(table->hash_table_hashes);
        free(table);
    }

    void new_hash_table(hash_table_layout** self) {
        *self = (hash_table_layout*)malloc(sizeof(hash_table_layout));
        hash_table_layout* record = *self;
        new (record) hash_table_layout();
    }

    std::pair<instance_ptr, size_t> add(instance_ptr el, int32_t slot = -1) {
        if (slot == -1)
            slot = table->allocateNewSlot(byte_count_per_el);
        size_t keyhash = std::hash<T>{}(*reinterpret_cast<T*>(el));
        table->add(keyhash, slot);
        instance_ptr dst = table->items + slot * byte_count_per_el;
        std::memcpy((void*)dst, (void*)el, sizeof(T));
        return std::pair<instance_ptr, size_t>(dst, keyhash);
    }

    bool cmp(instance_ptr key_to_find, instance_ptr key_in_table) {
        if (key_to_find == key_in_table) {
            return cmpResultToBoolForPyOrdering(Py_EQ, 0);
        }
        if (*reinterpret_cast<T*>(key_to_find) == *reinterpret_cast<T*>(key_in_table)) {
            return cmpResultToBoolForPyOrdering(Py_EQ, 0);
        }
        return cmpResultToBoolForPyOrdering(Py_EQ, 1);
    }

    instance_ptr lookupKey(instance_ptr key_to_find, size_t keyhash) {
        using namespace std::placeholders;
        auto cmp_func = std::bind(&HashTableLayout::cmp, this, key_to_find, _1);
        int32_t index = table->find(byte_count_per_el, keyhash, cmp_func);
        if (index >= 0) {
            return table->items + index * byte_count_per_el;
        }
        return 0;
    }

    bool remove(instance_ptr el) {
        using namespace std::placeholders;
        size_t keyhash = std::hash<T>{}(*reinterpret_cast<T*>(el));
        auto cmp_func = std::bind(&HashTableLayout::cmp, this, el, _1);
        int32_t index = table->remove(byte_count_per_el, keyhash, cmp_func);
        if (index >= 0) {
            return true;
        }
        return false;
    }

    hash_table_layout* get() const { return table; }

  private:
    hash_table_layout* table;
};
