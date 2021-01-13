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

bool ConstDictType::_updateAfterForwardTypesChanged() {
    size_t old_bytes_per_key_value_pair = m_bytes_per_key_value_pair;

    m_size = sizeof(void*);
    m_is_default_constructible = true;
    m_bytes_per_key = m_key->bytecount();
    m_bytes_per_key_value_pair = m_key->bytecount() + m_value->bytecount();
    m_bytes_per_key_subtree_pair = m_key->bytecount() + this->bytecount();

    std::string name = "ConstDict(" + m_key->name(true) + "->" + m_value->name(true) + ")";

    if (m_is_recursive_forward) {
        name = m_recursive_name;
    }

    bool anyChanged = (
        name != m_name ||
        m_bytes_per_key_value_pair != old_bytes_per_key_value_pair
    );

    m_name = name;
    m_stripped_name = "";

    return anyChanged;
}

bool ConstDictType::isBinaryCompatibleWithConcrete(Type* other) {
    if (other->getTypeCategory() != m_typeCategory) {
        return false;
    }

    ConstDictType* otherO = (ConstDictType*)other;

    return m_key->isBinaryCompatibleWith(otherO->m_key) &&
        m_value->isBinaryCompatibleWith(otherO->m_value);
}

// static
ConstDictType* ConstDictType::Make(Type* key, Type* value, ConstDictType* knownType) {
    PyEnsureGilAcquired getTheGil;

    static std::map<std::pair<Type*, Type*>, ConstDictType*> m;

    auto lookup_key = std::make_pair(key,value);

    auto it = m.find(lookup_key);
    if (it == m.end()) {
        it = m.insert(
            std::make_pair(
                lookup_key,
                knownType ? knownType : new ConstDictType(key, value)
            )
        ).first;
    }

    return it->second;
}

void ConstDictType::repr(instance_ptr self, ReprAccumulator& stream, bool isStr) {
    PushReprState isNew(stream, self);

    if (!isNew) {
        stream << m_name << "(" << (void*)self << ")";
        return;
    }

    stream << "{";

    int32_t ct = count(self);

    for (long k = 0; k < ct;k++) {
        if (k > 0) {
            stream << ", ";
        }

        m_key->repr(kvPairPtrKey(self,k),stream, false);
        stream << ": ";
        m_value->repr(kvPairPtrValue(self,k),stream, false);
    }

    stream << "}";
}

void ConstDictType::repr_keys(instance_ptr self, ReprAccumulator& stream) {
    PushReprState isNew(stream, self);

    if (!isNew) {
        stream << m_name << "(" << (void*)self << ")";
        return;
    }

    stream << "const_dict_keys([";

    int32_t ct = count(self);

    for (long k = 0; k < ct;k++) {
        if (k > 0) {
            stream << ", ";
        }

        m_key->repr(kvPairPtrKey(self,k),stream, false);
    }

    stream << "])";
}

void ConstDictType::repr_items(instance_ptr self, ReprAccumulator& stream) {
    PushReprState isNew(stream, self);

    if (!isNew) {
        stream << m_name << "(" << (void*)self << ")";
        return;
    }

    stream << "const_dict_items([";

    int32_t ct = count(self);

    for (long k = 0; k < ct;k++) {
        if (k > 0) {
            stream << ", ";
        }

        stream << "(";
        m_key->repr(kvPairPtrKey(self,k), stream, false);
        stream << ", ";
        m_value->repr(kvPairPtrValue(self,k), stream, false);
        stream << ")";
    }

    stream << "])";
}

void ConstDictType::repr_values(instance_ptr self, ReprAccumulator& stream) {
    PushReprState isNew(stream, self);

    if (!isNew) {
        stream << m_name << "(" << (void*)self << ")";
        return;
    }

    stream << "const_dict_values([";

    int32_t ct = count(self);

    for (long k = 0; k < ct;k++) {
        if (k > 0) {
            stream << ", ";
        }

        m_value->repr(kvPairPtrValue(self,k), stream, false);
    }

    stream << "])";
}

typed_python_hash_type ConstDictType::hash(instance_ptr left) {
    if (size(left) == 0) {
        return 0x123456;
    }

    if ((*(layout**)left)->hash_cache == -1) {
        HashAccumulator acc((int)getTypeCategory());

        int32_t count = size(left);
        acc.add(count);
        for (long k = 0; k < count;k++) {
            acc.add(m_key->hash(kvPairPtrKey(left,k)));
            acc.add(m_value->hash(kvPairPtrValue(left,k)));
        }

        (*(layout**)left)->hash_cache = acc.get();
        if ((*(layout**)left)->hash_cache == -1) {
            (*(layout**)left)->hash_cache = -2;
        }
    }

    return (*(layout**)left)->hash_cache;
}

//to make this fast(er), we do dict size comparison first, then keys, then values
bool ConstDictType::cmp(instance_ptr left, instance_ptr right, int pyComparisonOp, bool suppressExceptions, bool compareValues) {
    if (pyComparisonOp == Py_NE) {
        return !cmp(left, right, Py_EQ, suppressExceptions);
    }

    if (size(left) < size(right)) {
        return cmpResultToBoolForPyOrdering(pyComparisonOp, -1);
    }
    if (size(left) > size(right)) {
        return cmpResultToBoolForPyOrdering(pyComparisonOp, 1);
    }

    if (*(layout**)left == *(layout**)right) {
        return cmpResultToBoolForPyOrdering(pyComparisonOp, 0);
    }

    int ct = count(left);

    if (pyComparisonOp == Py_EQ) {
        for (long k = 0; k < ct; k++) {
            if (m_key->cmp(kvPairPtrKey(left,k), kvPairPtrKey(right,k), Py_NE, true) ||
                    (compareValues && m_value->cmp( kvPairPtrValue(left,k), kvPairPtrValue(right,k), Py_NE, true))) {
                return false;
            }
        }

        return true;
    } else {
        for (long k = 0; k < ct; k++) {
            if (m_key->cmp(kvPairPtrKey(left,k), kvPairPtrKey(right,k), Py_NE, true)) {
                if (m_key->cmp(kvPairPtrKey(left,k), kvPairPtrKey(right,k), Py_LT, true)) {
                    return cmpResultToBoolForPyOrdering(pyComparisonOp, -1);
                }
                return cmpResultToBoolForPyOrdering(pyComparisonOp, 1);
            }
        }

        if (compareValues) {
            for (long k = 0; k < ct; k++) {
                if (m_value->cmp(kvPairPtrValue(left,k), kvPairPtrValue(right,k), Py_NE, true)) {
                    if (m_value->cmp(kvPairPtrValue(left,k), kvPairPtrValue(right,k), Py_LT, true)) {
                        return cmpResultToBoolForPyOrdering(pyComparisonOp, -1);
                    }
                    return cmpResultToBoolForPyOrdering(pyComparisonOp, 1);
                }
            }
        }

        return cmpResultToBoolForPyOrdering(pyComparisonOp, 0);
    }
}

void ConstDictType::addDicts(instance_ptr lhs, instance_ptr rhs, instance_ptr output) {
    std::vector<instance_ptr> keep;

    int64_t lhsCount = count(lhs);
    int64_t rhsCount = count(rhs);

    for (long k = 0; k < lhsCount; k++) {
        instance_ptr lhsVal = kvPairPtrKey(lhs, k);

        if (!lookupValueByKey(rhs, lhsVal)) {
            keep.push_back(lhsVal);
        }
    }

    constructor(output, rhsCount + keep.size(), false);

    for (long k = 0; k < rhsCount; k++) {
        m_key->copy_constructor(kvPairPtrKey(output,k), kvPairPtrKey(rhs, k));
        m_value->copy_constructor(kvPairPtrValue(output,k), kvPairPtrValue(rhs, k));
    }
    for (long k = 0; k < keep.size(); k++) {
        m_key->copy_constructor(kvPairPtrKey(output,k + rhsCount), keep[k]);
        m_value->copy_constructor(kvPairPtrValue(output,k + rhsCount), keep[k] + m_bytes_per_key);
    }
    incKvPairCount(output, keep.size() + rhsCount);

    sortKvPairs(output);
}

void ConstDictType::subtractTupleOfKeysFromDict(instance_ptr lhs, instance_ptr rhs, instance_ptr output) {
    TupleOfType* tupleType = tupleOfKeysType();

    int64_t lhsCount = count(lhs);
    int64_t rhsCount = tupleType->count(rhs);

    std::set<int> remove;

    for (long k = 0; k < rhsCount; k++) {
        int64_t index = lookupIndexByKey(lhs, tupleType->eltPtr(rhs, k));
        if (index != -1) {
            remove.insert(index);
        }
    }

    constructor(output, lhsCount - remove.size(), false);

    long written = 0;
    for (long k = 0; k < lhsCount; k++) {
        if (remove.find(k) == remove.end()) {
            m_key->copy_constructor(kvPairPtrKey(output,written), kvPairPtrKey(lhs, k));
            m_value->copy_constructor(kvPairPtrValue(output,written), kvPairPtrValue(lhs, k));

            written++;
        }
    }

    incKvPairCount(output, written);
}

instance_ptr ConstDictType::kdPairPtrKey(instance_ptr self, int64_t i) {
    if (!(*(layout**)self)) {
        return self;
    }

    layout& record = **(layout**)self;

    return record.data + m_bytes_per_key_subtree_pair * i;
}

instance_ptr ConstDictType::kdPairPtrDict(instance_ptr self, int64_t i) {
    if (!(*(layout**)self)) {
        return self;
    }

    layout& record = **(layout**)self;

    return record.data + m_bytes_per_key_subtree_pair * i + m_bytes_per_key;
}

instance_ptr ConstDictType::kvPairPtrKey(instance_ptr self, int64_t i) {
    if (!(*(layout**)self)) {
        return self;
    }

    layout& record = **(layout**)self;

    return record.data + m_bytes_per_key_value_pair * i;
}

instance_ptr ConstDictType::kvPairPtrValue(instance_ptr self, int64_t i) {
    if (!(*(layout**)self)) {
        return self;
    }

    layout& record = **(layout**)self;

    return record.data + m_bytes_per_key_value_pair * i + m_bytes_per_key;
}

void ConstDictType::incKvPairCount(instance_ptr self, int by) {
    if (by == 0) {
        return;
    }

    layout& record = **(layout**)self;
    record.count += by;
}

void ConstDictType::sortKvPairs(instance_ptr self) {
    if (!*(layout**)self) {
        return;
    }

    layout& record = **(layout**)self;

    assert(!record.subpointers);

    if (record.count <= 1) {
        return;
    }
    else if (record.count == 2) {
        if (m_key->cmp(kvPairPtrKey(self, 0), kvPairPtrKey(self,1), Py_GT, true)) {
            m_key->swap(kvPairPtrKey(self,0), kvPairPtrKey(self,1));
            m_value->swap(kvPairPtrValue(self,0), kvPairPtrValue(self,1));
        }
        return;
    } else {
        std::vector<int> indices;
        for (long k=0;k<record.count;k++) {
            indices.push_back(k);
        }

        std::sort(indices.begin(), indices.end(), [&](int l, int r) {
            return m_key->cmp(kvPairPtrKey(self,l),kvPairPtrKey(self,r), Py_LT, true);
            });

        //create a temporary buffer
        std::vector<uint8_t> d;
        d.resize(m_bytes_per_key_value_pair * record.count);

        //final_lookup contains the location of each value in the original sort
        for (long k = 0; k < indices.size(); k++) {
            m_key->swap(kvPairPtrKey(self, indices[k]), &d[m_bytes_per_key_value_pair*k]);
            m_value->swap(kvPairPtrValue(self, indices[k]), &d[m_bytes_per_key_value_pair*k+m_bytes_per_key]);
        }

        //now move them back
        for (long k = 0; k < indices.size(); k++) {
            m_key->swap(kvPairPtrKey(self, k), &d[m_bytes_per_key_value_pair*k]);
            m_value->swap(kvPairPtrValue(self, k), &d[m_bytes_per_key_value_pair*k+m_bytes_per_key]);
        }
    }
}

instance_ptr ConstDictType::keyTreePtr(instance_ptr self, int64_t i) {
    if (!(*(layout**)self)) {
        return self;
    }

    layout& record = **(layout**)self;

    return record.data + m_bytes_per_key_subtree_pair * i;
}

bool ConstDictType::instanceIsSubtrees(instance_ptr self) {
    if (!(*(layout**)self)) {
        return self;
    }

    layout& record = **(layout**)self;

    return record.subpointers != 0;
}

int64_t ConstDictType::refcount(instance_ptr self) {
    if (!(*(layout**)self)) {
        return 0;
    }

    layout& record = **(layout**)self;

    return record.refcount;
}

int64_t ConstDictType::count(instance_ptr self) {
    if (!(*(layout**)self)) {
        return 0;
    }

    layout& record = **(layout**)self;

    if (record.subpointers) {
        return record.subpointers;
    }

    return record.count;
}

int64_t ConstDictType::size(instance_ptr self) {
    if (!(*(layout**)self)) {
        return 0;
    }

    return (*(layout**)self)->count;
}

int64_t ConstDictType::lookupIndexByKey(instance_ptr self, instance_ptr key) {
    if (!(*(layout**)self)) {
        return -1;
    }

    layout& record = **(layout**)self;

    assert(record.subpointers == 0); //this is not implemented yet

    long low = 0;
    long high = record.count;

    while (low < high) {
        long mid = (low+high)/2;

        if (m_key->cmp(kvPairPtrKey(self, mid), key, Py_EQ, true)) {
            return mid;
        } else if (m_key->cmp(kvPairPtrKey(self, mid), key, Py_LT, true)) {
            low = mid+1;
        } else {
            high = mid;
        }
    }

    return -1;
}

instance_ptr ConstDictType::lookupValueByKey(instance_ptr self, instance_ptr key) {
    int64_t offset = lookupIndexByKey(self, key);
    if (offset == -1) {
        return 0;
    }
    return kvPairPtrValue(self, offset);
}

void ConstDictType::constructor(instance_ptr self, int64_t space, bool isPointerTree) {
    assertForwardsResolvedSufficientlyToInstantiate();

    if (space == 0) {
        (*(layout**)self) = nullptr;
        return;
    }

    int bytesPer = isPointerTree ? m_bytes_per_key_subtree_pair : m_bytes_per_key_value_pair;

    (*(layout**)self) = (layout*)tp_malloc(sizeof(layout) + bytesPer * space);

    layout& record = **(layout**)self;

    record.count = 0;
    record.subpointers = 0;
    record.refcount = 1;
    record.hash_cache = -1;
}

void ConstDictType::constructor(instance_ptr self) {
    (*(layout**)self) = nullptr;
}

void ConstDictType::destroy(instance_ptr self) {
    if (!(*(layout**)self)) {
        return;
    }

    layout& record = **(layout**)self;

    if (record.refcount.fetch_sub(1) == 1) {
        if (record.subpointers == 0) {
            m_key->destroy(record.count, [&](long ix) {
                return record.data + m_bytes_per_key_value_pair * ix;
            });
            m_value->destroy(record.count, [&](long ix) {
                return record.data + m_bytes_per_key_value_pair * ix + m_bytes_per_key;
            });
        } else {
            m_key->destroy(record.subpointers, [&](long ix) {
                return record.data + m_bytes_per_key_subtree_pair * ix;
            });
            ((Type*)this)->destroy(record.subpointers, [&](long ix) {
                return record.data + m_bytes_per_key_subtree_pair * ix + m_bytes_per_key;
            });
        }

        tp_free((*(layout**)self));
    }
}

void ConstDictType::copy_constructor(instance_ptr self, instance_ptr other) {
    (*(layout**)self) = (*(layout**)other);
    if (*(layout**)self) {
        (*(layout**)self)->refcount++;
    }
}

void ConstDictType::assign(instance_ptr self, instance_ptr other) {
    layout* old = (*(layout**)self);

    (*(layout**)self) = (*(layout**)other);

    if (*(layout**)self) {
        (*(layout**)self)->refcount++;
    }

    destroy((instance_ptr)&old);
}
