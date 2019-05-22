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
#include "CompositeType.hpp"

class Alternative : public Type {
public:
    class layout {
    public:
        std::atomic<int64_t> refcount;

        int64_t which;
        uint8_t data[];
    };

    Alternative(std::string name,
                const std::vector<std::pair<std::string, NamedTuple*> >& subtypes,
                const std::map<std::string, Function*>& methods
                ) :
            Type(TypeCategory::catAlternative),
            m_default_construction_ix(0),
            m_default_construction_type(nullptr),
            m_subtypes(subtypes),
            m_methods(methods)
    {
        m_name = name;
        m_is_simple = false;

        if (m_subtypes.size() > 255) {
            throw std::runtime_error("Can't have an alternative with more than 255 subelements");
        }

        _forwardTypesMayHaveChanged();
        forwardTypesMayHaveChanged();
    }

    bool isBinaryCompatibleWithConcrete(Type* other);

    template<class visitor_type>
    void _visitContainedTypes(const visitor_type& visitor) {
    }

    template<class visitor_type>
    void _visitReferencedTypes(const visitor_type& visitor) {
        for (auto& subtype_pair: m_subtypes) {
            Type* t = (Type*)subtype_pair.second;
            visitor(t);
            assert(t == subtype_pair.second);
        }
        for (long k = 0; k < m_subtypes.size(); k++) {
            Type* t = concreteSubtype(k);
            visitor(t);
        }

        for (auto& method_pair: m_methods) {
            Type* t = (Type*)method_pair.second;
            visitor(t);
            assert(t == method_pair.second);
        }
    }

    void _forwardTypesMayHaveChanged();

    bool cmp(instance_ptr left, instance_ptr right, int pyComparisonOp);

    template<class buf_t>
    void serialize(instance_ptr self, buf_t& buffer, size_t fieldNumber) {
        buffer.writeBeginSingle(fieldNumber);
        m_subtypes[which(self)].second->serialize(eltPtr(self), buffer, which(self));
    }

    template<class buf_t>
    void deserialize(instance_ptr self, buf_t& buffer, size_t wireType) {
        if (wireType != WireType::SINGLE) {
            throw std::runtime_error("Corrupt data (Alternative expects a SINGLE wire type)");
        }

        std::pair<size_t, size_t> fieldAndWire = buffer.readFieldNumberAndWireType();
        size_t which = fieldAndWire.first;

        if (which >= m_subtypes.size()) {
            throw std::runtime_error("Corrupt data (Alternative field number was out of bounds)");
        }

        if (m_all_alternatives_empty) {
            *(uint8_t*)self = which;
            //still need to consume whatever is in this message
            m_subtypes[which].second->deserialize(self, buffer, fieldAndWire.second);
            return;
        }

        *(layout**)self = (layout*)malloc(
            sizeof(layout) +
            m_subtypes[which].second->bytecount()
            );

        layout& record = **(layout**)self;

        record.refcount = 1;
        record.which = which;

        m_subtypes[which].second->deserialize(record.data, buffer, fieldAndWire.second);
    }

    void repr(instance_ptr self, ReprAccumulator& stream);

    int32_t hash32(instance_ptr left);

    instance_ptr eltPtr(instance_ptr self) const;

    int64_t which(instance_ptr self) const;

    int64_t refcount(instance_ptr self) const;

    void constructor(instance_ptr self);

    void destroy(instance_ptr self);

    void copy_constructor(instance_ptr self, instance_ptr other);

    void assign(instance_ptr self, instance_ptr other);

    static Alternative* Make(std::string name,
                         const std::vector<std::pair<std::string, NamedTuple*> >& types,
                         const std::map<std::string, Function*>& methods //methods preclude us from being in the memo
                         );

    Alternative* renamed(std::string newName) {
        return Make(newName, m_subtypes, m_methods);
    }

    const std::vector<std::pair<std::string, NamedTuple*> >& subtypes() const {
        return m_subtypes;
    }

    bool all_alternatives_empty() const {
        return m_all_alternatives_empty;
    }

    Type* pickConcreteSubclassConcrete(instance_ptr data);

    const std::map<std::string, Function*>& getMethods() const {
        return m_methods;
    }

    Type* concreteSubtype(size_t which);

private:
    bool m_all_alternatives_empty;

    int m_default_construction_ix;

    Type* m_default_construction_type;

    std::vector<std::pair<std::string, NamedTuple*> > m_subtypes;

    std::vector<Type*> m_subtypes_concrete;

    std::map<std::string, Function*> m_methods;

    std::map<std::string, int> m_arg_positions;
};

