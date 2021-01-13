#include "AllTypes.hpp"
#include "hash_table_layout.hpp"

#include <map>

SetType* SetType::Make(Type* eltype, SetType* knownType) {
    PyEnsureGilAcquired getTheGil;

    static std::map<Type*, SetType*> m;

    auto it = m.find(eltype);

    if (it == m.end()) {
        it = m.insert(std::make_pair(eltype, knownType ? knownType : new SetType(eltype))).first;
    }

    return it->second;
}

bool SetType::discard(instance_ptr self, instance_ptr key) {
    hash_table_layout& record = **(hash_table_layout**)self;
    typed_python_hash_type keyHash = m_key_type->hash(key);
    int32_t index = record.remove(m_bytes_per_el, keyHash, [&](instance_ptr ptr) {
        return m_key_type->cmp(key, ptr, Py_EQ);
    });
    if (index >= 0) {
        m_key_type->destroy(record.items + index * m_bytes_per_el);
        return true;
    }
    return false;
}

void SetType::clear(instance_ptr self) {
    hash_table_layout& record = **(hash_table_layout**)self;
    for (long k = 0; k < record.items_reserved; k++) {
        if (record.items_populated[k]) {
            m_key_type->destroy(record.items + m_bytes_per_el * k);
        }
    }
    record.allItemsHaveBeenRemoved();
}

instance_ptr SetType::insertKey(instance_ptr self, instance_ptr key) {
    hash_table_layout& record = **(hash_table_layout**)self;
    typed_python_hash_type keyHash = m_key_type->hash(key);
    int32_t slot = record.allocateNewSlot(m_bytes_per_el);
    record.add(keyHash, slot);
    m_key_type->copy_constructor(record.items + slot * m_bytes_per_el, key);
    return record.items + slot * m_bytes_per_el;
}

instance_ptr SetType::lookupKey(instance_ptr self, instance_ptr key) const {
    hash_table_layout& record = **(hash_table_layout**)self;
    typed_python_hash_type keyHash = m_key_type->hash(key);
    int32_t index = record.find(m_bytes_per_el, keyHash,
                                [&](instance_ptr ptr) { return m_key_type->cmp(key, ptr, Py_EQ); });
    if (index >= 0) {
        return record.items + index * m_bytes_per_el;
    }
    return 0;
}

int64_t SetType::size(instance_ptr self) const {
    hash_table_layout& record = **(hash_table_layout**)self;
    return record.hash_table_count;
}

bool SetType::_updateAfterForwardTypesChanged() {
    std::string name = "Set(" + m_key_type->name(true) + ")";
    m_size = sizeof(void*);
    m_is_default_constructible = true;
    m_bytes_per_el = m_key_type->bytecount();

    if (m_is_recursive_forward) {
        name = m_recursive_name;
    }

    bool anyChanged = name != m_name;
    m_name = name;
    m_stripped_name = "";
    return anyChanged;
}

int64_t SetType::refcount(instance_ptr self) const {
    hash_table_layout& record = **(hash_table_layout**)self;
    return record.refcount;
}

void SetType::constructor(instance_ptr self) {
    (*(hash_table_layout**)self) = (hash_table_layout*)tp_malloc(sizeof(hash_table_layout));
    hash_table_layout& record = **(hash_table_layout**)self;
    new (&record) hash_table_layout();
    record.refcount += 1;
}

void SetType::destroy(instance_ptr self) {
    hash_table_layout& record = **(hash_table_layout**)self;

    if (record.refcount.fetch_sub(1) == 1) {
        for (long k = 0; k < record.items_reserved; k++) {
            if (record.items_populated[k]) {
                m_key_type->destroy(record.items + m_bytes_per_el * k);
            }
        }

        tp_free(record.items);
        tp_free(record.items_populated);
        tp_free(record.hash_table_slots);
        tp_free(record.hash_table_hashes);
        tp_free(&record);
    }
}

void SetType::copy_constructor(instance_ptr self, instance_ptr other) {
    (*(hash_table_layout**)self) = (*(hash_table_layout**)other);
    (*(hash_table_layout**)self)->refcount++;
}

void SetType::assign(instance_ptr self, instance_ptr other) {
    hash_table_layout* old = (*(hash_table_layout**)self);
    (*(hash_table_layout**)self) = (*(hash_table_layout**)other);
    (*(hash_table_layout**)self)->refcount++;
    destroy((instance_ptr)&old);
}

bool SetType::subset(instance_ptr left, instance_ptr right) {
    hash_table_layout& l = **(hash_table_layout**)left;
    for (long k = 0; k < l.items_reserved; k++) {
        if (l.items_populated[k]) {
            instance_ptr key = l.items + m_bytes_per_el * k;
            instance_ptr otherKey = lookupKey(right, key);
            if (!otherKey) {
                return false;
            }
        }
    }
    return true;
}

bool SetType::cmp(instance_ptr left, instance_ptr right, int pyComparisonOp, bool suppressExceptions) {
    hash_table_layout& l = **(hash_table_layout**)left;
    hash_table_layout& r = **(hash_table_layout**)right;

    if (&l == &r) {
        return cmpResultToBoolForPyOrdering(pyComparisonOp, 0);
    }

    if (pyComparisonOp == Py_EQ || pyComparisonOp == Py_NE) {
        if (l.hash_table_count != r.hash_table_count)
            return cmpResultToBoolForPyOrdering(pyComparisonOp, 1);
        return cmpResultToBoolForPyOrdering(pyComparisonOp, subset(left, right) ? 0 : 1);
    }
    else if (pyComparisonOp == Py_LE) {
        if (l.hash_table_count > r.hash_table_count)
            return false;
        return subset(left, right);
    }
    else if (pyComparisonOp == Py_LT) {
        if (l.hash_table_count >= r.hash_table_count)
            return false;
        return subset(left, right);
    }
    else if (pyComparisonOp == Py_GE) {
        if (r.hash_table_count > l.hash_table_count)
            return false;
        return subset(right, left);
    }
    else if (pyComparisonOp == Py_GT) {
        if (r.hash_table_count >= l.hash_table_count)
            return false;
        return subset(right, left);
    }

    assert(false);
    return false;
}

void SetType::repr(instance_ptr self, ReprAccumulator& stream, bool isStr) {
    PushReprState isNew(stream, self);

    if (!isNew) {
        stream << m_name << "(" << (void*)self << ")";
        return;
    }

    stream << "{";
    hash_table_layout& l = **(hash_table_layout**)self;
    bool isFirst = true;

    for (long k = 0; k < l.items_reserved; k++) {
        if (l.items_populated[k]) {
            if (isFirst) {
                isFirst = false;
            } else {
                stream << ", ";
            }

            m_key_type->repr(l.items + k * m_bytes_per_el, stream, false);
        }
    }

    stream << "}";
}

int64_t SetType::slotCount(instance_ptr self) const {
    hash_table_layout& record = **(hash_table_layout**)self;
    return record.items_reserved;
}

bool SetType::slotPopulated(instance_ptr self, size_t offset) const {
    hash_table_layout& record = **(hash_table_layout**)self;
    return record.items_populated[offset];
}

instance_ptr SetType::keyAtSlot(instance_ptr self, size_t offset) const {
    hash_table_layout& record = **(hash_table_layout**)self;
    return record.items + m_bytes_per_el * offset;
}
