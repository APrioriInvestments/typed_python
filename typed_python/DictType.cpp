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

#include "AllTypes.hpp"

bool DictType::_updateAfterForwardTypesChanged() {
    m_size = sizeof(void*);
    m_is_default_constructible = true;
    m_bytes_per_key = m_key->bytecount();
    m_bytes_per_key_value_pair = m_key->bytecount() + m_value->bytecount();

    std::string name = "Dict(" + m_key->name(true) + "->" + m_value->name(true) + ")";

    if (m_is_recursive_forward) {
        name = m_recursive_name;
    }

    bool anyChanged = name != m_name;

    m_name = name;
    m_stripped_name = "";

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
DictType* DictType::Make(Type* key, Type* value, DictType* knownType) {
    PyEnsureGilAcquired getTheGil;

    static std::map<std::pair<Type*, Type*>, DictType*> m;

    auto lookup_key = std::make_pair(key,value);

    auto it = m.find(lookup_key);
    if (it == m.end()) {
        it = m.insert(
            std::make_pair(
                lookup_key,
                knownType ? knownType: new DictType(key, value)
            )
        ).first;
    }

    return it->second;
}

void DictType::repr(instance_ptr self, ReprAccumulator& stream, bool isStr) {
    PushReprState isNew(stream, self);

    if (!isNew) {
        stream << m_name << "(" << (void*)self << ")";
        return;
    }

    stream << "{";

    hash_table_layout& l = **(hash_table_layout**)self;
    bool isFirst = true;

    for (long k = 0; k < l.items_reserved;k++) {
        if (l.items_populated[k]) {
            if (isFirst) {
                isFirst = false;
            } else {
                stream << ", ";
            }

            m_key->repr(l.items + k * m_bytes_per_key_value_pair, stream, false);
            stream << ": ";
            m_value->repr(l.items + k * m_bytes_per_key_value_pair + m_bytes_per_key, stream, false);
        }
    }

    stream << "}";
}

void DictType::repr_keys(instance_ptr self, ReprAccumulator& stream) {
    PushReprState isNew(stream, self);

    if (!isNew) {
        stream << m_name << "(" << (void*)self << ")";
        return;
    }

    stream << "dict_keys([";

    hash_table_layout& l = **(hash_table_layout**)self;
    bool isFirst = true;

    for (long k = 0; k < l.items_reserved;k++) {
        if (l.items_populated[k]) {
            if (isFirst) {
                isFirst = false;
            } else {
                stream << ", ";
            }

            m_key->repr(l.items + k * m_bytes_per_key_value_pair, stream, false);
        }
    }

    stream << "])";
}

void DictType::repr_values(instance_ptr self, ReprAccumulator& stream) {
    PushReprState isNew(stream, self);

    if (!isNew) {
        stream << m_name << "(" << (void*)self << ")";
        return;
    }

    stream << "dict_values([";

    hash_table_layout& l = **(hash_table_layout**)self;
    bool isFirst = true;

    for (long k = 0; k < l.items_reserved;k++) {
        if (l.items_populated[k]) {
            if (isFirst) {
                isFirst = false;
            } else {
                stream << ", ";
            }

            m_value->repr(l.items + k * m_bytes_per_key_value_pair + m_bytes_per_key, stream, false);
        }
    }

    stream << "])";
}

void DictType::repr_items(instance_ptr self, ReprAccumulator& stream) {
    PushReprState isNew(stream, self);

    if (!isNew) {
        stream << m_name << "(" << (void*)self << ")";
        return;
    }

    stream << "dict_items([";

    hash_table_layout& l = **(hash_table_layout**)self;
    bool isFirst = true;

    for (long k = 0; k < l.items_reserved;k++) {
        if (l.items_populated[k]) {
            if (isFirst) {
                isFirst = false;
            } else {
                stream << ", ";
            }

            stream << "(";
            m_key->repr(l.items + k * m_bytes_per_key_value_pair, stream, false);
            stream << ", ";
            m_value->repr(l.items + k * m_bytes_per_key_value_pair + m_bytes_per_key, stream, false);
            stream << ")";
        }
    }

    stream << "])";
}

typed_python_hash_type DictType::hash(instance_ptr left) {
    throw std::logic_error(name() + " is not hashable");
}

//to make this fast(er), we do dict size comparison first, then keys, then values
bool DictType::cmp(instance_ptr left, instance_ptr right, int pyComparisonOp, bool suppressExceptions, bool compareValues) {
    if (pyComparisonOp != Py_NE && pyComparisonOp != Py_EQ) {
        throw std::runtime_error("Ordered comparison not supported between objects of type " + name());
    }

    hash_table_layout& l = **(hash_table_layout**)left;
    hash_table_layout& r = **(hash_table_layout**)right;

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

            if (!otherValue && compareValues) {
                return cmpResultToBoolForPyOrdering(pyComparisonOp, 1);
            }

            if (compareValues && m_value->cmp(value, otherValue, Py_NE, suppressExceptions)) {
                return cmpResultToBoolForPyOrdering(pyComparisonOp, 1);
            }
        }
    }

    return cmpResultToBoolForPyOrdering(pyComparisonOp, 0);
}

int64_t DictType::refcount(instance_ptr self) const {
    hash_table_layout& record = **(hash_table_layout**)self;

    return record.refcount;
}

int64_t DictType::slotCount(instance_ptr self) const {
    hash_table_layout& record = **(hash_table_layout**)self;

    return record.items_reserved;
}

bool DictType::slotPopulated(instance_ptr self, size_t slot) const {
    hash_table_layout& record = **(hash_table_layout**)self;

    return record.items_populated[slot];
}

instance_ptr DictType::keyAtSlot(instance_ptr self, size_t offset) const {
    hash_table_layout& record = **(hash_table_layout**)self;

    return record.items + m_bytes_per_key_value_pair * offset;
}

instance_ptr DictType::valueAtSlot(instance_ptr self, size_t offset) const {
    hash_table_layout& record = **(hash_table_layout**)self;

    return record.items + m_bytes_per_key_value_pair * offset + m_bytes_per_key;
}

int64_t DictType::size(instance_ptr self) const {
    hash_table_layout& record = **(hash_table_layout**)self;

    return record.hash_table_count;
}

instance_ptr DictType::lookupValueByKey(instance_ptr self, instance_ptr key) const {
    hash_table_layout& record = **(hash_table_layout**)self;

    typed_python_hash_type keyHash = m_key->hash(key);

    int32_t index = record.find(m_bytes_per_key_value_pair, keyHash, [&](instance_ptr ptr) {
        return m_key->cmp(key, ptr, Py_EQ, false);
    });

    if (index >= 0) {
        return record.items + index * m_bytes_per_key_value_pair + m_bytes_per_key;
    }

    return 0;
}

bool DictType::deleteKey(instance_ptr self, instance_ptr key) const {
    hash_table_layout& record = **(hash_table_layout**)self;

    typed_python_hash_type keyHash = m_key->hash(key);

    int32_t index = record.remove(m_bytes_per_key_value_pair, keyHash, [&](instance_ptr ptr) {
        return m_key->cmp(key, ptr, Py_EQ, false);
    });

    if (index >= 0) {
        m_key->destroy(record.items + index * m_bytes_per_key_value_pair);
        m_value->destroy(record.items + index * m_bytes_per_key_value_pair + m_bytes_per_key);
        return true;
    }

    return false;
}

void DictType::clear(instance_ptr self) {
    hash_table_layout& record = **(hash_table_layout**)self;

    for (long k = 0; k < record.items_reserved; k++) {
        if (record.items_populated[k]) {
            m_key->destroy(record.items + k * m_bytes_per_key_value_pair);
            m_value->destroy(record.items + k * m_bytes_per_key_value_pair + m_bytes_per_key);
        }

        record.items_populated[k] = 0;
    }

    record.allItemsHaveBeenRemoved();
}

bool DictType::deleteKeyWithUninitializedValue(instance_ptr self, instance_ptr key) const {
    hash_table_layout& record = **(hash_table_layout**)self;

    typed_python_hash_type keyHash = m_key->hash(key);

    int32_t index = record.remove(m_bytes_per_key_value_pair, keyHash, [&](instance_ptr ptr) {
        return m_key->cmp(key, ptr, Py_EQ, false);
    });

    if (index >= 0) {
        return true;
    }

    return false;
}

instance_ptr DictType::insertKey(instance_ptr self, instance_ptr key) const {
    hash_table_layout& record = **(hash_table_layout**)self;

    typed_python_hash_type keyHash = m_key->hash(key);

    int32_t slot = record.allocateNewSlot(m_bytes_per_key_value_pair);

    record.add(keyHash, slot);

    m_key->copy_constructor(record.items + slot * m_bytes_per_key_value_pair, key);

    return record.items + slot * m_bytes_per_key_value_pair + m_bytes_per_key;
}

void DictType::constructor(instance_ptr self) {
    assertForwardsResolvedSufficientlyToInstantiate();

    (*(hash_table_layout**)self) = (hash_table_layout*)tp_malloc(sizeof(hash_table_layout));

    hash_table_layout& record = **(hash_table_layout**)self;

    new (&record) hash_table_layout();

    record.refcount += 1;
}

void DictType::destroy(instance_ptr self) {
    hash_table_layout& record = **(hash_table_layout**)self;

    if (record.refcount.fetch_sub(1) == 1) {
        for (long k = 0; k < record.items_reserved; k++) {
            if (record.items_populated[k]) {
                m_key->destroy(record.items + m_bytes_per_key_value_pair * k);
                m_value->destroy(record.items + m_bytes_per_key_value_pair * k + m_bytes_per_key);
            }
        }

        tp_free(record.items);
        tp_free(record.items_populated);
        tp_free(record.hash_table_slots);
        tp_free(record.hash_table_hashes);
        tp_free(&record);
    }
}

void DictType::copy_constructor(instance_ptr self, instance_ptr other) {
    (*(hash_table_layout**)self) = (*(hash_table_layout**)other);
    (*(hash_table_layout**)self)->refcount++;
}

void DictType::assign(instance_ptr self, instance_ptr other) {
    hash_table_layout* old = (*(hash_table_layout**)self);

    (*(hash_table_layout**)self) = (*(hash_table_layout**)other);
    (*(hash_table_layout**)self)->refcount++;

    destroy((instance_ptr)&old);
}
