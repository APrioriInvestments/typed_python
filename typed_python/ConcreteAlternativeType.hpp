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

#include "ReprAccumulator.hpp"
#include "AlternativeType.hpp"

class ConcreteAlternative : public Type {
    ConcreteAlternative() :
        Type(TypeCategory::catConcreteAlternative),
        m_which(0),
        m_alternative(nullptr)
    {
        m_needs_post_init = true;
    }

public:
    typedef Alternative::layout layout;

    ConcreteAlternative(Alternative* m_alternative, int64_t which) :
            Type(TypeCategory::catConcreteAlternative),
            m_alternative(m_alternative),
            m_which(which)
    {
        m_is_forward_defined = true;
    }

    std::string computeRecursiveNameConcrete(TypeStack& typeStack) {
        return m_alternative->name() + "." + m_alternative->subtypes()[m_which].first;
    }

    void initializeFromConcrete(Type* forwardDefinitionOfSelf) {
        m_base = ((ConcreteAlternative*)forwardDefinitionOfSelf)->m_base;
        m_alternative = ((ConcreteAlternative*)forwardDefinitionOfSelf)->m_alternative;
        m_which = ((ConcreteAlternative*)forwardDefinitionOfSelf)->m_which;
    }

    void updateInternalTypePointersConcrete(const std::map<Type*, Type*>& groupMap) {
        updateTypeRefFromGroupMap(m_alternative, groupMap);
        updateTypeRefFromGroupMap(m_base, groupMap);
    }

    Type* cloneForForwardResolutionConcrete() {
        return new ConcreteAlternative();
    }

    void postInitializeConcrete() {
        if (m_which < 0 || m_which >= m_alternative->subtypes().size()) {
            throw std::runtime_error(
                "invalid alternative index: " +
                format(m_which) + " not in [0," +
                format(m_alternative->subtypes().size()) + ")"
            );
        }

        m_base = m_alternative;
        size_t size = m_alternative->bytecount();
        bool is_default_constructible = m_alternative->subtypes()[m_which].second->is_default_constructible();

        m_size = size;
        m_is_default_constructible = is_default_constructible;
    }

    std::string nameWithModuleConcrete() {
        if (m_alternative->moduleName().size() == 0) {
            return m_name;
        }

        return m_alternative->moduleName() + "." + m_name;
    }

    std::string moduleNameConcrete() {
        return m_alternative->moduleName();
    }

    void deepcopyConcrete(
        instance_ptr dest,
        instance_ptr src,
        DeepcopyContext& context
    ) {
        return m_alternative->deepcopy(dest, src, context);
    }

    size_t deepBytecountConcrete(instance_ptr instance, std::unordered_set<void*>& alreadyVisited, std::set<Slab*>* outSlabs) {
        return m_alternative->deepBytecount(instance, alreadyVisited, outSlabs);
    }

    bool isBinaryCompatibleWithConcrete(Type* other);

    template<class visitor_type>
    void _visitContainedTypes(const visitor_type& visitor) {
        Type* t = m_alternative;
        visitor(t);
        assert(t == m_alternative);
    }

    template<class visitor_type>
    void _visitReferencedTypes(const visitor_type& visitor) {
        Type* t = m_alternative;
        visitor(t);
        assert(t == m_alternative);
    }


    template<class visitor_type>
    void _visitCompilerVisibleInternals(const visitor_type& v) {
        v.visitHash(ShaHash(1, m_typeCategory) + ShaHash(m_which));
        v.visitTopo(m_alternative);
    }

    typed_python_hash_type hash(instance_ptr left) {
        return m_alternative->hash(left);
    }

    void repr(instance_ptr self, ReprAccumulator& stream, bool isStr) {
        m_alternative->repr(self,stream, isStr);
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

    void constructor(instance_ptr self);

    int64_t refcount(instance_ptr self) const {
        return m_alternative->refcount(self);
    }

    //returns an uninitialized object of type-index 'which'
    template<class subconstructor>
    void constructor(instance_ptr self, const subconstructor& s) const {
        if (m_alternative->all_alternatives_empty()) {
            *(uint8_t*)self = m_which;
            s(self);
        } else {
            *(layout**)self = (layout*)tp_malloc(
                sizeof(layout) +
                elementType()->bytecount()
                );

            layout& record = **(layout**)self;
            record.refcount = 1;
            record.which = m_which;
            try {
                s(record.data);
            } catch(...) {
                tp_free(*(layout**)self);
                throw;
            }
        }
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

    instance_ptr eltPtr(instance_ptr self) const {
        return m_alternative->eltPtr(self);
    }

    static ConcreteAlternative* Make(Alternative* alt, int64_t which);

    Type* elementType() const {
        return m_alternative->subtypes()[m_which].second;
    }

    Alternative* getAlternative() const {
        return m_alternative;
    }

    int64_t which() const {
        return m_which;
    }

    template<class visitor_type>
    void visitMROConcrete(const visitor_type& v) {
        v(this);
        m_alternative->visitMROConcrete(v);
    }

private:
    Alternative* m_alternative;

    int64_t m_which;
};
