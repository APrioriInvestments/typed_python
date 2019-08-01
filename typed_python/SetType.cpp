#include "AllTypes.hpp"
#include "hash_table_layout.hpp"

#include <map>

SetType* SetType::Make(Type* eltype) {
    static std::mutex guard;
    std::lock_guard<std::mutex> lg(guard);
    static std::map<Type*, SetType*> m;
    auto it = m.find(eltype);
    if (it == m.end())
        it = m.insert(std::make_pair(eltype, new SetType(eltype))).first;
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
            discard(self, keyAtSlot(self, k));
        }
    }
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
    std::string name = "Set(" + m_key_type->name() + ")";
    m_size = sizeof(void*);
    m_is_default_constructible = true;
    m_bytes_per_el = m_key_type->bytecount();

    if (m_is_recursive) {
        name = m_recursive_name;
    }

    bool anyChanged = name != m_name;
    m_name = name;
    return anyChanged;
}

int64_t SetType::refcount(instance_ptr self) const {
    hash_table_layout& record = **(hash_table_layout**)self;
    return record.refcount;
}

void SetType::constructor(instance_ptr self) {
    (*(hash_table_layout**)self) = (hash_table_layout*)malloc(sizeof(hash_table_layout));
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

        free(record.items);
        free(record.items_populated);
        free(record.hash_table_slots);
        free(record.hash_table_hashes);
        free(&record);
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

bool SetType::cmp(instance_ptr left, instance_ptr right, int pyComparisonOp,
                  bool suppressExceptions) {
    if (pyComparisonOp != Py_NE && pyComparisonOp != Py_EQ) {
        throw std::runtime_error("Ordered comparison not supported between objects of type "
                                 + name());
    }

    hash_table_layout& l = **(hash_table_layout**)left;
    hash_table_layout& r = **(hash_table_layout**)right;

    if (&l == &r) {
        return cmpResultToBoolForPyOrdering(pyComparisonOp, 0);
    }

    if (l.hash_table_count != r.hash_table_count) {
        return cmpResultToBoolForPyOrdering(pyComparisonOp, 1);
    }

    // check each item on the left to see if its in the right and has the same value
    for (long k = 0; k < l.items_reserved; k++) {
        if (l.items_populated[k]) {
            instance_ptr key = l.items + m_bytes_per_el * k;
            instance_ptr otherKey = lookupKey(right, key);
            if (!otherKey) {
                return cmpResultToBoolForPyOrdering(pyComparisonOp, 1);
            }
        }
    }

    return cmpResultToBoolForPyOrdering(pyComparisonOp, 0);
}

void SetType::repr(instance_ptr self, ReprAccumulator& stream) {
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

            m_key_type->repr(l.items + k * m_bytes_per_el, stream);
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
