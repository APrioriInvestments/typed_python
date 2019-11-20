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

#include "Type.hpp"
#include "ReprAccumulator.hpp"

class ConstDictType : public Type {
    class layout {
    public:
        std::atomic<int64_t> refcount;
        typed_python_hash_type hash_cache;
        int32_t count; //the actual number of items in the tree (in total)
        int32_t subpointers; //if 0, then all values are inline as pairs of (key,value)
                             //otherwise, its an array of '(key, ConstDict(key,value))'
        uint8_t data[];
    };

public:
    ConstDictType(Type* key, Type* value) :
            Type(TypeCategory::catConstDict),
            m_key(key),
            m_value(value)
    {
        endOfConstructorInitialization(); // finish initializing the type object.
    }

    template<class visitor_type>
    void _visitContainedTypes(const visitor_type& visitor) {
    }

    template<class visitor_type>
    void _visitReferencedTypes(const visitor_type& visitor) {
        visitor(m_key);
        visitor(m_value);
    }

    bool _updateAfterForwardTypesChanged();

    bool isBinaryCompatibleWithConcrete(Type* other);

    void _updateTypeMemosAfterForwardResolution() {
        ConstDictType::Make(m_key, m_value, this);
    }

    static ConstDictType* Make(Type* key, Type* value, ConstDictType* knownType = nullptr);


    // hand 'visitor' each an instance_ptr for
    // each value. if it returns 'false', exit early.
    template<class visitor_type>
    void visitValues(instance_ptr self, visitor_type visitor) {
        size_t ct = count(self);

        for (long k = 0; k < ct; k++) {
            if (!visitor(kvPairPtrValue(self, k))) {
                return;
            }
        }
    }

    // hand 'visitor' each key and value instance_ptr as a single tuple.
    // if it returns 'false', exit early.
    template<class visitor_type>
    void visitKeyValuePairs(instance_ptr self, visitor_type visitor) {
        size_t ct = count(self);

        for (long k = 0; k < ct; k++) {
            if (!visitor(kvPairPtrKey(self, k))) {
                return;
            }
        }
    }

    // hand 'visitor' each key and value instance_ptr as two separate arguments.
    // if it returns 'false', exit early.
    template<class visitor_type>
    void visitKeyValuePairsSeparately(instance_ptr self, visitor_type visitor) {
        size_t ct = count(self);

        for (long k = 0; k < ct; k++) {
            if (!visitor(kvPairPtrKey(self, k), kvPairPtrValue(self, k))) {
                return;
            }
        }
    }

    template<class buf_t>
    void serialize(instance_ptr self, buf_t& buffer, size_t fieldNumber) {
        size_t ct = count(self);

        buffer.writeBeginCompound(fieldNumber);
        buffer.writeUnsignedVarintObject(0, ct);
        for (long k = 0; k < ct;k++) {
            m_key->serialize(kvPairPtrKey(self,k),buffer, 0);
            m_value->serialize(kvPairPtrValue(self,k),buffer, 0);
        }

        buffer.writeEndCompound();
    }

    template<class buf_t>
    void deserialize(instance_ptr self, buf_t& buffer, size_t wireType) {
        int32_t ct = -1;

        size_t valuesRead = buffer.consumeCompoundMessageWithImpliedFieldNumbers(wireType,
            [&](size_t fieldNumber, size_t subWireType) {
                if (fieldNumber == 0) {
                    if (subWireType != WireType::VARINT) {
                        throw std::runtime_error("Corrupt ConstDict");
                    }
                    ct = buffer.readUnsignedVarint();
                    constructor(self, ct, false);
                } else {
                    size_t keyIx = (fieldNumber - 1) / 2;
                    bool isKey = fieldNumber % 2;
                    if (isKey) {
                        m_key->deserialize(kvPairPtrKey(self, keyIx), buffer, subWireType);
                    } else {
                        m_value->deserialize(kvPairPtrValue(self, keyIx), buffer, subWireType);
                    }
                }
        });

        if (ct == -1 || (valuesRead - 1) / 2 != ct) {
            throw std::runtime_error("Corrupt ConstDict.");
        }

        incKvPairCount(self, ct);
    }

    void repr(instance_ptr self, ReprAccumulator& stream, bool isStr);

    void repr_keys(instance_ptr self, ReprAccumulator& stream);

    void repr_values(instance_ptr self, ReprAccumulator& stream);

    void repr_items(instance_ptr self, ReprAccumulator& stream);

    typed_python_hash_type hash(instance_ptr left);

    bool cmp(instance_ptr left, instance_ptr right, int pyComparisonOp, bool suppressExceptions, bool compareValues=true);

    void addDicts(instance_ptr lhs, instance_ptr rhs, instance_ptr output);

    TupleOfType* tupleOfKeysType() const {
        return TupleOfType::Make(m_key);
    }

    void subtractTupleOfKeysFromDict(instance_ptr lhs, instance_ptr rhs, instance_ptr output);

    instance_ptr kvPairPtrKey(instance_ptr self, int64_t i);

    instance_ptr kvPairPtrValue(instance_ptr self, int64_t i);

    void incKvPairCount(instance_ptr self, int by = 1);

    void sortKvPairs(instance_ptr self);

    instance_ptr keyTreePtr(instance_ptr self, int64_t i);

    bool instanceIsSubtrees(instance_ptr self);

    int64_t refcount(instance_ptr self);

    int64_t count(instance_ptr self);

    int64_t size(instance_ptr self);

    int64_t lookupIndexByKey(instance_ptr self, instance_ptr key);

    instance_ptr lookupValueByKey(instance_ptr self, instance_ptr key);

    void constructor(instance_ptr self, int64_t space, bool isPointerTree);

    template<class constructor_fun>
    void constructor(instance_ptr self, int64_t space, const constructor_fun& initializer) {
        constructor(self, space, false);

        try {
            for (long ix = 0; ix < space; ix++) {
                initializer(kvPairPtrKey(self, ix), kvPairPtrValue(self, ix));

                incKvPairCount(self);
            }

            sortKvPairs(self);
        } catch(...) {
            destroy(self);
            throw;
        }
    }

    void constructor(instance_ptr self);

    void destroy(instance_ptr self);

    void copy_constructor(instance_ptr self, instance_ptr other);

    void assign(instance_ptr self, instance_ptr other);


    Type* keyType() const { return m_key; }
    Type* valueType() const { return m_value; }

private:
    Type* m_key;
    Type* m_value;
    size_t m_bytes_per_key;
    size_t m_bytes_per_key_value_pair;
    size_t m_bytes_per_key_subtree_pair;
};
