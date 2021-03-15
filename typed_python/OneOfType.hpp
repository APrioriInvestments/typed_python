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

PyDoc_STRVAR(OneOf_doc,
    "OneOfType(T1, T2, ...)() -> default-initialized variable type\n"
    "\n"
    "Each of T1, T2, ... may be a typed_python type, or any python primitive value.\n"
    );

class OneOfType : public Type {
public:
    OneOfType(const std::vector<Type*>& types) noexcept :
                    Type(TypeCategory::catOneOf),
                    m_types(types)
    {
        if (m_types.size() > 255) {
            throw std::runtime_error("OneOf types are limited to 255 alternatives in this implementation");
        }

        m_doc = OneOf_doc;

        endOfConstructorInitialization(); // finish initializing the type object.
    }

    ShaHash _computeIdentityHash(MutuallyRecursiveTypeGroup* groupHead = nullptr) {
        ShaHash newHash = ShaHash(1, m_typeCategory);

        for (auto t: m_types) {
            newHash += t->identityHash(groupHead);
        }

        return newHash;
    }

    bool isBinaryCompatibleWithConcrete(Type* other);

    template<class visitor_type>
    void _visitContainedTypes(const visitor_type& visitor) {
        for (auto& typePtr: m_types) {
            visitor(typePtr);
        }
    }

    template<class visitor_type>
    void _visitReferencedTypes(const visitor_type& visitor) {
        _visitContainedTypes(visitor);
    }

    bool _updateAfterForwardTypesChanged();

    bool isPODConcrete() {
        for (auto t: m_types) {
            if (!t->isPOD()) {
                return false;
            }
        }

        return true;
    }

    std::string computeName() const;

    void deepcopyConcrete(
        instance_ptr dest,
        instance_ptr src,
        DeepcopyContext& context
    ) {
        uint8_t which = *(uint8_t*)dest = *(uint8_t*)src;
        m_types[which]->deepcopy(dest+1, src+1, context);
    }

    size_t deepBytecountConcrete(instance_ptr instance, std::unordered_set<void*>& alreadyVisited, std::set<Slab*>* outSlabs) {
        if (isPOD()) {
            return 0;
        }

        int fieldNumber = *(uint8_t*)instance;

        return m_types[fieldNumber]->deepBytecount(instance + 1, alreadyVisited, outSlabs);
    }

    template<class buf_t>
    void deserialize(instance_ptr self, buf_t& buffer, size_t wireType) {
        bool hitOne = false;

        buffer.consumeCompoundMessage(wireType, [&](size_t fieldNumber, size_t subWireType) {
            if (hitOne) {
                throw std::runtime_error("Corrupt OneOf had multiple fields.");
            }

            if (fieldNumber < m_types.size()) {
                *(uint8_t*)self = fieldNumber;
                m_types[fieldNumber]->deserialize(self+1, buffer, subWireType);
                hitOne = true;
            }
        });

        if (!hitOne) {
            constructor(self);
        }
    }

    template<class buf_t>
    void serialize(instance_ptr self, buf_t& buffer, size_t fieldNumber) {
        buffer.writeBeginSingle(fieldNumber);
        m_types[*((uint8_t*)self)]->serialize(self+1, buffer, *(uint8_t*)self);
    }

    void repr(instance_ptr self, ReprAccumulator& stream, bool isStr);

    typed_python_hash_type hash(instance_ptr left);

    bool cmp(instance_ptr left, instance_ptr right, int pyComparisonOp, bool suppressExceptions);

    std::pair<Type*, instance_ptr> unwrap(instance_ptr self) {
        return std::make_pair(m_types[*(uint8_t*)self], self+1);
    }

    size_t computeBytecount() const;

    void constructor(instance_ptr self);

    void destroy(instance_ptr self);

    void copy_constructor(instance_ptr self, instance_ptr other);

    void assign(instance_ptr self, instance_ptr other);

    const std::vector<Type*>& getTypes() const {
        return m_types;
    }

    void _updateTypeMemosAfterForwardResolution() {
        OneOfType::Make(m_types, this);
    }

    static OneOfType* Make(const std::vector<Type*>& types, OneOfType* knownType = nullptr);

private:
    std::vector<Type*> m_types;
};
