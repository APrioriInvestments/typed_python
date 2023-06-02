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
    AlternativeMatcher() : Type(TypeCategory::catAlternativeMatcher)
    {
        m_needs_post_init = true;
    }
public:
    AlternativeMatcher(Type* inAlternative) : Type(TypeCategory::catAlternativeMatcher)
    {
        m_is_forward_defined = true;
        m_is_default_constructible = false;
        m_alternative = inAlternative;
        m_size = inAlternative->bytecount();
        m_is_simple = false;
    }

    std::string computeRecursiveNameConcrete(TypeStack& typeStack) {
        return "AlternativeMatcher("
            + m_alternative->computeRecursiveName(typeStack)
            + ")";
    }

    template<class visitor_type>
    void _visitCompilerVisibleInternals(const visitor_type& v) {
        v.visitHash(ShaHash(1, m_typeCategory));
        v.visitTopo(m_alternative);
    }

    template<class visitor_type>
    void _visitContainedTypes(const visitor_type& visitor) {
        visitor(m_alternative);
    }

    template<class visitor_type>
    void _visitReferencedTypes(const visitor_type& visitor) {
        visitor(m_alternative);
    }

    void postInitializeConcrete() {
        m_size = m_alternative->bytecount();
    }

    void initializeFromConcrete(Type* forwardDefinitionOfSelf) {
        m_alternative = ((AlternativeMatcher*)forwardDefinitionOfSelf)->m_alternative;
    }

    Type* cloneForForwardResolutionConcrete() {
        return new AlternativeMatcher();
    }

    void updateInternalTypePointersConcrete(const std::map<Type*, Type*>& groupMap) {
        updateTypeRefFromGroupMap(m_alternative, groupMap);
    }

    static AlternativeMatcher* Make(Type* alt) {
        if (alt->isForwardDefined()) {
            return new AlternativeMatcher(alt);
        }

        PyEnsureGilAcquired getTheGil;

        static std::map<Type*, AlternativeMatcher*> memo;

        auto it = memo.find(alt);
        if (it != memo.end()) {
            return it->second;
        }

        AlternativeMatcher* res = new AlternativeMatcher(alt);
        AlternativeMatcher* concrete = (AlternativeMatcher*)res->forwardResolvesTo();

        memo[alt] = concrete;

        return concrete;
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
