#pragma once

#include "Type.hpp"

class SetType : public Type {
  public:
    SetType(Type* eltype)
        : Type(TypeCategory::catSet)
        , m_key_type(eltype) {
        endOfConstructorInitialization();
    }

    template<class visitor_type>
    void _visitReferencedTypes(const visitor_type& visitor) {
        visitor(m_key_type);
    }

    void _updateTypeMemosAfterForwardResolution() {
        SetType::Make(m_key_type, this);
    }

    instance_ptr insertKey(instance_ptr self, instance_ptr key);
    instance_ptr lookupKey(instance_ptr self, instance_ptr key) const;
    bool discard(instance_ptr self, instance_ptr key);
    void clear(instance_ptr self);
    void constructor(instance_ptr self);
    void destroy(instance_ptr self);
    void copy_constructor(instance_ptr self, instance_ptr other);
    void repr(instance_ptr self, ReprAccumulator& stream);
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
        size_t count;
        size_t id;
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
};
