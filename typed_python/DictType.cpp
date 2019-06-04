/******************************************************************************
   Copyright 2017-2019 Nativepython Authors

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

#include "AllTypes.hpp"

bool DictType::_updateAfterForwardTypesChanged() {
    m_size = sizeof(void*);
    m_is_default_constructible = true;
    m_bytes_per_key = m_key->bytecount();
    m_bytes_per_key_value_pair = m_key->bytecount() + m_value->bytecount();
    m_key_value_pair_type = Tuple::Make({m_key, m_value});

    std::string name = "Dict(" + m_key->name() + "->" + m_value->name() + ")";

    if (m_is_recursive) {
        name = m_recursive_name;
    }

    bool anyChanged = name != m_name;

    m_name = name;

    return anyChanged;
}

bool DictType::isBinaryCompatibleWithConcrete(Type* other) {
    if (other->getTypeCategory() != m_typeCategory) {
        return false;
    }

    DictType* otherO = (DictType*)other;

    return m_key->isBinaryCompatibleWith(otherO->m_key) &&
        m_value->isBinaryCompatibleWith(otherO->m_value);
}

// static
DictType* DictType::Make(Type* key, Type* value) {
    static std::mutex guard;

    std::lock_guard<std::mutex> lock(guard);

    static std::map<std::pair<Type*, Type*>, DictType*> m;

    auto lookup_key = std::make_pair(key,value);

    auto it = m.find(lookup_key);
    if (it == m.end()) {
        it = m.insert(std::make_pair(lookup_key, new DictType(key, value))).first;
    }

    return it->second;
}

void DictType::repr(instance_ptr self, ReprAccumulator& stream) {
    PushReprState isNew(stream, self);

    if (!isNew) {
        stream << m_name << "(" << (void*)self << ")";
        return;
    }

    stream << "{";

    layout& l = **(layout**)self;
    bool isFirst = true;

    for (long k = 0; k < l.items_reserved;k++) {
        if (l.items_populated[k]) {
            if (isFirst) {
                isFirst = false;
            } else {
                stream << ", ";
            }

            m_key->repr(l.items + k * m_bytes_per_key_value_pair, stream);
            stream << ": ";
            m_value->repr(l.items + k * m_bytes_per_key_value_pair + m_bytes_per_key, stream);
        }
    }

    stream << "}";
}

int32_t DictType::hash32(instance_ptr left) {
    throw std::logic_error(name() + " is not hashable");
}

//to make this fast(er), we do dict size comparison first, then keys, then values
bool DictType::cmp(instance_ptr left, instance_ptr right, int pyComparisonOp) {
    if (pyComparisonOp != Py_NE && pyComparisonOp != Py_EQ) {
        throw std::runtime_error("Ordered comparison not supported between objects of type " + name());
    }

    layout& l = **(layout**)left;
    layout& r = **(layout**)right;

    if (&l == &r) {
        return cmpResultToBoolForPyOrdering(pyComparisonOp, 0);
    }

    if (l.hash_table_count != r.hash_table_count) {
        return cmpResultToBoolForPyOrdering(pyComparisonOp, 1);
    }

    //check each item on the left to see if its in the right and has the same value
    for (long k = 0; k < l.items_reserved; k++) {
        if (l.items_populated[k]) {
            instance_ptr key = l.items + m_bytes_per_key_value_pair * k;
            instance_ptr value = key + m_bytes_per_key;
            instance_ptr otherValue = lookupValueByKey(right, key);

            if (!otherValue) {
                return cmpResultToBoolForPyOrdering(pyComparisonOp, 1);
            }

            if (m_value->cmp(value, otherValue, Py_NE)) {
                return cmpResultToBoolForPyOrdering(pyComparisonOp, 1);
            }
        }
    }

    return cmpResultToBoolForPyOrdering(pyComparisonOp, 0);
}

int64_t DictType::refcount(instance_ptr self) const {
    layout& record = **(layout**)self;

    return record.refcount;
}

int64_t DictType::slotCount(instance_ptr self) const {
    layout& record = **(layout**)self;

    return record.items_reserved;
}

bool DictType::slotPopulated(instance_ptr self, size_t slot) const {
    layout& record = **(layout**)self;

    return record.items_populated[slot];
}

instance_ptr DictType::keyAtSlot(instance_ptr self, size_t offset) const {
    layout& record = **(layout**)self;

    return record.items + m_bytes_per_key_value_pair * offset;
}

instance_ptr DictType::valueAtSlot(instance_ptr self, size_t offset) const {
    layout& record = **(layout**)self;

    return record.items + m_bytes_per_key_value_pair * offset + m_bytes_per_key;
}

int64_t DictType::size(instance_ptr self) const {
    layout& record = **(layout**)self;

    return record.hash_table_count;
}

instance_ptr DictType::lookupValueByKey(instance_ptr self, instance_ptr key) const {
    layout& record = **(layout**)self;

    int32_t keyHash = m_key->hash32(key);

    int32_t index = record.find(m_bytes_per_key_value_pair, keyHash, [&](instance_ptr ptr) {
        return m_key->cmp(key, ptr, Py_EQ);
    });

    if (index >= 0) {
        return record.items + index * m_bytes_per_key_value_pair + m_bytes_per_key;
    }

    return 0;
}

bool DictType::deleteKey(instance_ptr self, instance_ptr key) const {
    layout& record = **(layout**)self;

    int32_t keyHash = m_key->hash32(key);

    int32_t index = record.remove(m_bytes_per_key_value_pair, keyHash, [&](instance_ptr ptr) {
        return m_key->cmp(key, ptr, Py_EQ);
    });

    if (index >= 0) {
        m_key->destroy(record.items + index * m_bytes_per_key_value_pair);
        m_value->destroy(record.items + index * m_bytes_per_key_value_pair + m_bytes_per_key);
        return true;
    }

    return false;
}

bool DictType::deleteKeyWithUninitializedValue(instance_ptr self, instance_ptr key) const {
    layout& record = **(layout**)self;

    int32_t keyHash = m_key->hash32(key);

    int32_t index = record.remove(m_bytes_per_key_value_pair, keyHash, [&](instance_ptr ptr) {
        return m_key->cmp(key, ptr, Py_EQ);
    });

    if (index >= 0) {
        return true;
    }

    return false;
}

instance_ptr DictType::insertKey(instance_ptr self, instance_ptr key) const {
    layout& record = **(layout**)self;

    int32_t keyHash = m_key->hash32(key);

    int32_t slot = record.allocateNewSlot(m_bytes_per_key_value_pair);

    record.add(keyHash, slot);

    m_key->copy_constructor(record.items + slot * m_bytes_per_key_value_pair, key);

    return record.items + slot * m_bytes_per_key_value_pair + m_bytes_per_key;
}

void DictType::constructor(instance_ptr self) {
    assertForwardsResolved();
    (*(layout**)self) = (layout*)malloc(sizeof(layout));

    layout& record = **(layout**)self;

    new (&record) layout();

    record.refcount += 1;
}

void DictType::destroy(instance_ptr self) {
    layout& record = **(layout**)self;

    if (record.refcount.fetch_sub(1) == 1) {
        for (long k = 0; k < record.items_reserved; k++) {
            if (record.items_populated[k]) {
                m_key->destroy(record.items + m_bytes_per_key_value_pair * k);
                m_value->destroy(record.items + m_bytes_per_key_value_pair * k + m_bytes_per_key);
            }
        }

        free(record.items);
        free(record.items_populated);
        free(record.hash_table_slots);
        free(record.hash_table_hashes);
        free(&record);
    }
}

void DictType::copy_constructor(instance_ptr self, instance_ptr other) {
    (*(layout**)self) = (*(layout**)other);
    (*(layout**)self)->refcount++;
}

void DictType::assign(instance_ptr self, instance_ptr other) {
    layout* old = (*(layout**)self);

    (*(layout**)self) = (*(layout**)other);
    (*(layout**)self)->refcount++;

    destroy((instance_ptr)&old);
}

