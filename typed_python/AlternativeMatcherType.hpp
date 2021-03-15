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

// Models an alternatives '.matches' result
class AlternativeMatcher : public Type {
public:
    AlternativeMatcher(Type* inAlternative) : Type(TypeCategory::catAlternativeMatcher)
    {
        m_is_default_constructible = false;
        m_alternative = inAlternative;
        m_size = inAlternative->bytecount();
        m_is_simple = false;

        endOfConstructorInitialization(); // finish initializing the type object.
    }

    ShaHash _computeIdentityHash(MutuallyRecursiveTypeGroup* groupHead = nullptr) {
        return ShaHash(1, m_typeCategory) + m_alternative->identityHash(groupHead);
    }

    template<class visitor_type>
    void _visitContainedTypes(const visitor_type& visitor) {
        visitor(m_alternative);
    }

    template<class visitor_type>
    void _visitReferencedTypes(const visitor_type& visitor) {
        visitor(m_alternative);
    }

    bool _updateAfterForwardTypesChanged() {
        bool anyChanged = false;

        std::string name = "AlternativeMatcher(" + m_alternative->name(true) + ")";
        size_t size = m_alternative->bytecount();

        anyChanged = (
            name != m_name ||
            size != m_size
        );

        m_name = name;
        m_stripped_name = "";
        m_size = size;

        return anyChanged;
    }

    void _updateTypeMemosAfterForwardResolution() {
        AlternativeMatcher::Make(m_alternative, this);
    }

    static AlternativeMatcher* Make(Type* alt, AlternativeMatcher* knownType=nullptr) {
        PyEnsureGilAcquired getTheGil;

        static std::map<Type*, AlternativeMatcher*> m;

        auto it = m.find(alt);

        if (it == m.end()) {
            it = m.insert(
                std::make_pair(alt, knownType ? knownType : new AlternativeMatcher(alt))
            ).first;
        }

        return it->second;
    }

    void repr(instance_ptr self, ReprAccumulator& stream, bool isStr) {
        stream << m_name;
    }

    typed_python_hash_type hash(instance_ptr left) {
        return m_alternative->hash(left);
    }

    size_t deepBytecountConcrete(instance_ptr instance, std::unordered_set<void*>& alreadyVisited, std::set<Slab*>* outSlabs) {
        return m_alternative->deepBytecount(instance, alreadyVisited, outSlabs);
    }

    void deepcopyConcrete(
        instance_ptr dest,
        instance_ptr src,
        DeepcopyContext& context
    ) {
        m_alternative->deepcopy(dest, src, context);
    }

    template<class buf_t>
    void deserialize(instance_ptr self, buf_t& buffer, size_t wireType) {
        m_alternative->deserialize(self, buffer, wireType);
    }

    template<class buf_t>
    void serialize(instance_ptr self, buf_t& buffer, size_t fieldNumber) {
        m_alternative->serialize(self, buffer, fieldNumber);
    }

    bool cmp(instance_ptr left, instance_ptr right, int pyComparisonOp, bool suppressExceptions) {
        return m_alternative->cmp(left, right, pyComparisonOp, suppressExceptions);
    }

    void constructor(instance_ptr self) {
        m_alternative->constructor(self);
    }

    void destroy(instance_ptr self) {
        m_alternative->destroy(self);
    }

    void copy_constructor(instance_ptr self, instance_ptr other) {
        m_alternative->copy_constructor(self, other);
    }

    void assign(instance_ptr self, instance_ptr other) {
        m_alternative->assign(self, other);
    }

    Type* getBaseAlternative() const {
        if (m_alternative->isConcreteAlternative()) {
            return ((ConcreteAlternative*)m_alternative)->getAlternative();
        }

        return m_alternative;
    }

    Type* getAlternative() const {
        return m_alternative;
    }

    Type* m_alternative;
};
