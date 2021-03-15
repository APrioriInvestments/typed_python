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

PyDoc_STRVAR(Set_doc,
    "Set(T)() -> empty typed set, ready to contain type T\n"
    "Set(T)(s) -> typed set with elements of type T, initialized from set s\n"
    "Set(T)(x) -> typed set with elements of type T, initialized from iterable x\n"
    "\n"
    "Raises TypeError if types don't match.\n"
    );

class SetType : public Type {
  public:
    SetType(Type* eltype)
        : Type(TypeCategory::catSet)
        , m_key_type(eltype)
    {
        m_doc = Set_doc;
        endOfConstructorInitialization();
    }

    template<class visitor_type>
    void _visitReferencedTypes(const visitor_type& visitor) {
        visitor(m_key_type);
    }

    void _updateTypeMemosAfterForwardResolution() {
        SetType::Make(m_key_type, this);
    }

    ShaHash _computeIdentityHash(MutuallyRecursiveTypeGroup* groupHead = nullptr) {
        return ShaHash(1, m_typeCategory) + m_key_type->identityHash(groupHead);
    }

    instance_ptr insertKey(instance_ptr self, instance_ptr key);
    instance_ptr lookupKey(instance_ptr self, instance_ptr key) const;
    bool discard(instance_ptr self, instance_ptr key);
    void clear(instance_ptr self);
    void constructor(instance_ptr self);
    void destroy(instance_ptr self);
    void copy_constructor(instance_ptr self, instance_ptr other);
    void repr(instance_ptr self, ReprAccumulator& stream, bool isStr);
    int64_t refcount(instance_ptr self) const;
    bool _updateAfterForwardTypesChanged();
    int64_t size(instance_ptr self) const;
    bool cmp(instance_ptr left, instance_ptr right, int pyComparisonOp,
             bool suppressExceptions = false);
    void assign(instance_ptr self, instance_ptr other);
    Type* keyType() const { return m_key_type; }

    // hash_table_layout accessors
    int64_t slotCount(instance_ptr self) const;
    bool slotPopulated(instance_ptr self, size_t offset) const;
    instance_ptr keyAtSlot(instance_ptr self, size_t offset) const;

    // hand 'visitor' each set element as an instance_ptr.
    // if it returns 'false', exit early.
    template<class visitor_type>
    void visitSetElements(instance_ptr self, visitor_type visitor) {
        hash_table_layout& l = **(hash_table_layout**)self;

        for (long k = 0; k < l.items_reserved; k++) {
            if (l.items_populated[k]) {
                if (!visitor(l.items + m_bytes_per_el * k)) {
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
                m_key_type,
                nullptr
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
        res += bytesRequiredForAllocation(l.items_reserved * m_bytes_per_el);
        res += bytesRequiredForAllocation(l.items_reserved);

        // count the hashtable
        res += bytesRequiredForAllocation(sizeof(int32_t) * l.hash_table_size) * 2;

        if (!m_key_type->isPOD()) {
            for (long k = 0; k < l.items_reserved; k++) {
                if (l.items_populated[k]) {
                    res += m_key_type->deepBytecount(l.items + k * m_bytes_per_el, alreadyVisited, outSlabs);
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

        size_t slotsWritten = 0;
        for (long k = 0; k < l.items_reserved; k++) {
            if (l.items_populated[k]) {
                m_key_type->serialize(l.items + m_bytes_per_el * k, buffer, 0);
                slotsWritten++;
            }
        }

        buffer.writeEndCompound();

        if (slotsWritten != l.hash_table_count) {
            throw std::runtime_error("invalid hash table encountered: count not in line with items_reserved");
        }
    }

    typed_python_hash_type hash(instance_ptr left) {
        throw std::runtime_error("Can't hash Set instances");
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

                    l.prepareForDeserialization(count, m_bytes_per_el);
                } else {
                    hash_table_layout& l = **((hash_table_layout**)self);

                    size_t keyIx = fieldNumber - 2;
                    m_key_type->deserialize(l.items + m_bytes_per_el * keyIx, buffer, subWireType);
                }
        });

        if (!wasFromId) {
            if (valuesRead - 2 != count) {
                throw std::runtime_error("Invalid Set found.");
            }

            hash_table_layout& l = **((hash_table_layout**)self);
            l.buildHashTableAfterDeserialization(
                m_bytes_per_el,
                [&](instance_ptr ptr) { return m_key_type->hash(ptr); }
            );
        }
    }

  public:
    static SetType* Make(Type* eltype, SetType* knownType=nullptr);

  private:
    Type* m_key_type;
    size_t m_bytes_per_el;
    bool subset(instance_ptr left, instance_ptr right);
};
