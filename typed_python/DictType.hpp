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

#include "Type.hpp"
#include "ReprAccumulator.hpp"
#include "hash_table_layout.hpp"

#include <unordered_map>

PyDoc_STRVAR(DictType_doc,
    "Dict(K, V)() -> new empty typed dictionary with keytype K and valuetype V\n"
    "Dict(K, V)(d) -> new typed dictionary initialized from dict d\n"
    "\n"
    "Raises TypeError if types don't match.\n"
    );

class DictType : public Type {
public:
    DictType(Type* key, Type* value) :
            Type(TypeCategory::catDict),
            m_key(key),
            m_value(value)
    {
        m_doc = DictType_doc;
        endOfConstructorInitialization(); // finish initializing the type object.
    }

    void _updateTypeMemosAfterForwardResolution() {
        DictType::Make(m_key, m_value, this);
    }

    template<class visitor_type>
    void _visitContainedTypes(const visitor_type& visitor) {
    }

    template<class visitor_type>
    void _visitReferencedTypes(const visitor_type& visitor) {
        visitor(m_key);
        visitor(m_value);
    }

    ShaHash _computeIdentityHash(MutuallyRecursiveTypeGroup* groupHead = nullptr) {
        return ShaHash(1, m_typeCategory) + m_key->identityHash(groupHead) + m_value->identityHash(groupHead);
    }

    bool _updateAfterForwardTypesChanged();

    bool isBinaryCompatibleWithConcrete(Type* other);

    static DictType* Make(Type* key, Type* value, DictType* knownType=nullptr);

    // hand 'visitor' each an instance_ptr for
    // each value. if it returns 'false', exit early.
    template<class visitor_type>
    void visitValues(instance_ptr self, visitor_type visitor) {
        hash_table_layout& l = **(hash_table_layout**)self;

        for (long k = 0; k < l.items_reserved; k++) {
            if (l.items_populated[k]) {
                if (!visitor(l.items + m_bytes_per_key_value_pair * k + m_bytes_per_key)) {
                    return;
                }
            }
        }
    }

    // hand 'visitor' each key and value instance_ptr as a single tuple.
    // if it returns 'false', exit early.
    template<class visitor_type>
    void visitKeyValuePairs(instance_ptr self, visitor_type visitor) {
        hash_table_layout& l = **(hash_table_layout**)self;

        for (long k = 0; k < l.items_reserved; k++) {
            if (l.items_populated[k]) {
                if (!visitor(l.items + m_bytes_per_key_value_pair * k)) {
                    return;
                }
            }
        }
    }

    // hand 'visitor' each key and value instance_ptr as two distinct arguments.
    // if it returns 'false', exit early.
    template<class visitor_type>
    void visitKeyValuePairsAsSeparateArgs(instance_ptr self, visitor_type visitor) {
        hash_table_layout& l = **(hash_table_layout**)self;

        for (long k = 0; k < l.items_reserved; k++) {
            if (l.items_populated[k]) {
                if (!visitor(
                    l.items + m_bytes_per_key_value_pair * k,
                    l.items + m_bytes_per_key_value_pair * k + m_bytes_per_key)
                ) {
                    return;
                }
            }
        }
    }

    void deepcopyConcrete(
        instance_ptr dest,
        instance_ptr src,
        DeepcopyContext& context
    ) {
        hash_table_layout_ptr& destRecordPtr = *(hash_table_layout**)dest;
        hash_table_layout_ptr& srcRecordPtr = *(hash_table_layout**)src;

        auto it = context.alreadyAllocated.find((instance_ptr)srcRecordPtr);

        if (it == context.alreadyAllocated.end()) {
            destRecordPtr = srcRecordPtr->deepcopy(
            context,
                this,
                m_key,
                m_value
            );

            context.alreadyAllocated[(instance_ptr)srcRecordPtr] = (instance_ptr)destRecordPtr;
        } else {
            destRecordPtr = (hash_table_layout_ptr)context.alreadyAllocated[(instance_ptr)srcRecordPtr];
            destRecordPtr->refcount++;
        }
    }

    size_t deepBytecountConcrete(instance_ptr instance, std::unordered_set<void*>& alreadyVisited, std::set<Slab*>* outSlabs) {
        hash_table_layout& l = **(hash_table_layout**)instance;

        if (alreadyVisited.find((void*)&l) != alreadyVisited.end()) {
            return 0;
        }

        alreadyVisited.insert((void*)&l);

        if (outSlabs && Slab::slabForAlloc(&l)) {
            outSlabs->insert(Slab::slabForAlloc(&l));
            return 0;
        }

        size_t res = bytesRequiredForAllocation(sizeof(hash_table_layout));
        // count 'items_populated' and 'items'
        res += bytesRequiredForAllocation(l.items_reserved);
        res += bytesRequiredForAllocation(l.items_reserved * m_bytes_per_key_value_pair);

        // count the hashtable
        res += bytesRequiredForAllocation(sizeof(int32_t) * l.hash_table_size) * 2;

        if (!m_key->isPOD()) {
            for (long k = 0; k < l.items_reserved; k++) {
                if (l.items_populated[k]) {
                    res += m_key->deepBytecount(l.items + k * m_bytes_per_key_value_pair, alreadyVisited, outSlabs);
                }
            }
        }

        if (!m_value->isPOD()) {
            for (long k = 0; k < l.items_reserved; k++) {
                if (l.items_populated[k]) {
                    res += m_value->deepBytecount(l.items + k * m_bytes_per_key_value_pair + m_bytes_per_key, alreadyVisited, outSlabs);
                }
            }
        }

        return res;
    }

    template<class buf_t>
    void serialize(instance_ptr self, buf_t& buffer, size_t fieldNumber) {
        hash_table_layout& l = **(hash_table_layout**)self;

        uint32_t id;
        bool isNew;
        std::tie(id, isNew) = buffer.cachePointer(&l, this);

        if (!isNew) {
            buffer.writeBeginSingle(fieldNumber);
            buffer.writeUnsignedVarintObject(0, id);
            return;
        }

        buffer.writeBeginCompound(fieldNumber);
        buffer.writeUnsignedVarintObject(0, id);
        buffer.writeUnsignedVarintObject(0, l.hash_table_count);

        size_t slotsWritten = 2;
        for (long k = 0; k < l.items_reserved; k++) {
            if (l.items_populated[k]) {
                m_key->serialize(l.items + m_bytes_per_key_value_pair * k, buffer, 0);
                m_value->serialize(l.items + m_bytes_per_key_value_pair * k + m_bytes_per_key, buffer, 0);
                slotsWritten += 2;
            }
        }

        buffer.writeEndCompound();

        if (slotsWritten != l.hash_table_count * 2 + 2) {
            throw std::runtime_error("invalid hash table encountered: count not in line with items_reserved");
        }
    }

    template<class buf_t>
    void deserialize(instance_ptr self, buf_t& buffer, size_t wireType) {
        size_t count = 0;
        size_t id = 0;
        bool wasFromId = false;

        size_t valuesRead = buffer.consumeCompoundMessageWithImpliedFieldNumbers(wireType,
            [&](size_t fieldNumber, size_t subWireType) {
                if (fieldNumber == 0) {
                    assertWireTypesEqual(subWireType, WireType::VARINT);
                    id = buffer.readUnsignedVarint();

                    void* ptr = buffer.lookupCachedPointer(id);

                    if (ptr) {
                        *((hash_table_layout**)self) = (hash_table_layout*)ptr;
                        (*(hash_table_layout**)self)->refcount++;
                        wasFromId = true;
                    }
                } else if (fieldNumber == 1) {
                    assertWireTypesEqual(subWireType, WireType::VARINT);
                    count = buffer.readUnsignedVarint();

                    constructor(self);

                    hash_table_layout& l = **((hash_table_layout**)self);
                    buffer.addCachedPointer(id, &l, this);
                    l.refcount++;

                    l.prepareForDeserialization(count, m_bytes_per_key_value_pair);
                } else {
                    hash_table_layout& l = **((hash_table_layout**)self);

                    size_t keyIx = (fieldNumber - 2) / 2;
                    bool isKey = fieldNumber % 2 == 0;
                    if (isKey) {
                        m_key->deserialize(l.items + m_bytes_per_key_value_pair * keyIx, buffer, subWireType);
                    } else {
                        m_value->deserialize(l.items + m_bytes_per_key_value_pair * keyIx + m_bytes_per_key, buffer, subWireType);
                    }
                }
        });

        if (!wasFromId) {
            if ((valuesRead - 2) / 2 != count) {
                throw std::runtime_error("Invalid Dict found.");
            }

            hash_table_layout& l = **((hash_table_layout**)self);
            l.buildHashTableAfterDeserialization(
                m_bytes_per_key_value_pair,
                [&](instance_ptr ptr) { return m_key->hash(ptr); }
                );
        }
    }

    void repr(instance_ptr self, ReprAccumulator& stream, bool isStr);

    void repr_keys(instance_ptr self, ReprAccumulator& stream);

    void repr_values(instance_ptr self, ReprAccumulator& stream);

    void repr_items(instance_ptr self, ReprAccumulator& stream);

    typed_python_hash_type hash(instance_ptr left);

    bool cmp(
        instance_ptr left,
        instance_ptr right,
        int pyComparisonOp,
        bool suppressExceptions,
        bool compareValues=true
    );

    int64_t refcount(instance_ptr self) const;

    int64_t size(instance_ptr self) const;

    int64_t slotCount(instance_ptr self) const;

    bool slotPopulated(instance_ptr self, size_t offset) const;

    instance_ptr keyAtSlot(instance_ptr self, size_t offset) const;

    instance_ptr valueAtSlot(instance_ptr self, size_t offset) const;

    instance_ptr lookupValueByKey(instance_ptr self, instance_ptr key) const;

    // insert a new key, copy constructing 'key' but returning an uninitialized value pointer.
    instance_ptr insertKey(instance_ptr self, instance_ptr key) const;

    bool deleteKey(instance_ptr self, instance_ptr key) const;

    bool deleteKeyWithUninitializedValue(instance_ptr self, instance_ptr key) const;

    void constructor(instance_ptr self);

    void destroy(instance_ptr self);

    void clear(instance_ptr self);

    void copy_constructor(instance_ptr self, instance_ptr other);

    void assign(instance_ptr self, instance_ptr other);

    Type* keyType() const { return m_key; }

    Type* valueType() const { return m_value; }

private:
    Type* m_key;
    Type* m_value;
    size_t m_bytes_per_key;
    size_t m_bytes_per_key_value_pair;
};
