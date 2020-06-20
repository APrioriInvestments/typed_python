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

class PointerTo : public Type {
protected:
    typedef void* instance;

public:
    PointerTo(Type* t) :
        Type(TypeCategory::catPointerTo),
        m_element_type(t)
    {
        m_size = sizeof(instance);
        m_is_default_constructible = true;

        endOfConstructorInitialization(); // finish initializing the type object.
    }

    ShaHash _computeIdentityHash(Type* groupHead = nullptr) {
        return ShaHash(1, m_typeCategory) + m_element_type->identityHash(groupHead);
    }

    void _updateTypeMemosAfterForwardResolution() {
        PointerTo::Make(m_element_type, this);
    }

    static PointerTo* Make(Type* elt, PointerTo* knownType = nullptr) {
        static std::mutex guard;

        std::lock_guard<std::mutex> lock(guard);

        static std::map<Type*, PointerTo*> m;

        auto it = m.find(elt);
        if (it == m.end()) {
            it = m.insert(std::make_pair(elt, knownType ? knownType : new PointerTo(elt))).first;
        }

        return it->second;
    }

    bool isPODConcrete() {
        return true;
    }

    bool _updateAfterForwardTypesChanged() {
        std::string name = "PointerTo(" + m_element_type->name(true) + ")";

        if (m_is_recursive_forward) {
            name = m_recursive_name;
        }

        bool anyChanged = name != m_name;

        m_name = name;
        m_stripped_name = "";

        return anyChanged;
    }

    template<class visitor_type>
    void _visitContainedTypes(const visitor_type& visitor) {
    }

    template<class visitor_type>
    void _visitReferencedTypes(const visitor_type& visitor) {
        visitor(m_element_type);
    }

    template<class buf_t>
    void serialize(instance_ptr self, buf_t& buffer, size_t fieldNumber) {
        throw std::runtime_error("Can't serialize Pointers");
    }

    template<class buf_t>
    void deserialize(instance_ptr self, buf_t& buffer, size_t wireType) {
        throw std::runtime_error("Can't deserialize Pointers");
    }

    void repr(instance_ptr self, ReprAccumulator& stream, bool isRepr) {
        stream << "(" << m_element_type->name() << "*)" << *(void**)self;
    }

    typed_python_hash_type hash(instance_ptr left) {
        HashAccumulator acc((int)getTypeCategory());

        acc.addRegister((uint64_t)*(void**)left);

        return acc.get();
    }

    bool cmp(instance_ptr left, instance_ptr right, int pyComparisonOp, bool suppressExceptions) {
        if (*(void**)left < *(void**)right) {
            return cmpResultToBoolForPyOrdering(pyComparisonOp, -1);
        }
        if (*(void**)left > *(void**)right) {
            return cmpResultToBoolForPyOrdering(pyComparisonOp, 1);
        }

        return cmpResultToBoolForPyOrdering(pyComparisonOp, 0);
    }

    Type* getEltType() const {
        return m_element_type;
    }

    void constructor(instance_ptr self) {
        *(void**)self = nullptr;
    }

    instance_ptr dereference(instance_ptr self) {
        return (instance_ptr)*(void**)self;
    }

    void destroy(instance_ptr self)  {
    }

    void copy_constructor(instance_ptr self, instance_ptr other) {
        *(void**)self = *(void**)other;
    }

    void assign(instance_ptr self, instance_ptr other) {
        *(void**)self = *(void**)other;
    }

    void offsetBy(instance_ptr out, instance_ptr in, long ix) {
        *(uint8_t**)out = *(uint8_t**)in + (ix * m_element_type->bytecount());
    }

private:
    Type* m_element_type;
};
