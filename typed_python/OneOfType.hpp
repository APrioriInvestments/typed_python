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

#pragma once

#include "Type.hpp"

class OneOfType : public Type {
public:
    OneOfType(const std::vector<Type*>& types) :
                    Type(TypeCategory::catOneOf),
                    m_types(types)
    {
        if (m_types.size() > 255) {
            throw std::runtime_error("OneOf types are limited to 255 alternatives in this implementation");
        }

        m_resolved = false;
        forwardTypesMayHaveChanged();
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

    void _forwardTypesMayHaveChanged();

    std::string computeName() const;

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

    void repr(instance_ptr self, ReprAccumulator& stream);

    int32_t hash32(instance_ptr left);

    bool cmp(instance_ptr left, instance_ptr right, int pyComparisonOp);

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

    static OneOfType* Make(const std::vector<Type*>& types);

private:
    std::vector<Type*> m_types;
};
