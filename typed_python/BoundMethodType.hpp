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

PyDoc_STRVAR(BoundMethod_doc,
    "BoundMethod(T, name)\n\n"
    "Holds an instance of 'T' as 't' and has a call method that dispatches to t.name(...)\n"
);


class BoundMethod : public Type {
public:
    BoundMethod() : Type(TypeCategory::catBoundMethod)
    {
    }

    BoundMethod(Type* inFirstArg, std::string funcName) : Type(TypeCategory::catBoundMethod)
    {
        m_is_forward_defined = true;

        m_funcName = funcName;
        m_first_arg = inFirstArg;
        m_size = inFirstArg->bytecount();
    }

    const char* docConcrete() {
        return BoundMethod_doc;
    }

    template<class visitor_type>
    void _visitCompilerVisibleInternals(const visitor_type& v) {
        v.visitHash(ShaHash(1, m_typeCategory));
        v.visitTopo(m_first_arg);
        v.visitName(m_funcName);
    }

    template<class visitor_type>
    void _visitContainedTypes(const visitor_type& visitor) {
        visitor(m_first_arg);
    }

    template<class visitor_type>
    void _visitReferencedTypes(const visitor_type& visitor) {
        visitor(m_first_arg);
    }

    std::string computeRecursiveNameConcrete(TypeStack& typeStack) {
        return "BoundMethod("
            + m_first_arg->computeRecursiveName(typeStack)
            + ", " + m_funcName
            + ")";
    }

    void postInitializeConcrete() {
        m_size = m_first_arg->bytecount();
        m_is_default_constructible = false;
    }

    void initializeFromConcrete(Type* forwardDefinitionOfSelf) {
        m_first_arg = ((BoundMethod*)forwardDefinitionOfSelf)->m_first_arg;
        m_funcName = ((BoundMethod*)forwardDefinitionOfSelf)->m_funcName;
    }

    Type* cloneForForwardResolutionConcrete() {
        return new BoundMethod();
    }

    void updateInternalTypePointersConcrete(const std::map<Type*, Type*>& groupMap) {
        updateTypeRefFromGroupMap(m_first_arg, groupMap);
    }

    static BoundMethod* Make(Type* firstArg, std::string funcName) {
        if (firstArg->isForwardDefined()) {
            return new BoundMethod(firstArg, funcName);
        }

        PyEnsureGilAcquired getTheGil;

        typedef std::pair<Type*, std::string> keytype;

        static std::map<keytype, BoundMethod*> memo;

        auto it = memo.find(keytype(firstArg, funcName));
        if (it != memo.end()) {
            return it->second;
        }

        BoundMethod* res = new BoundMethod(firstArg, funcName);
        BoundMethod* concrete = (BoundMethod*)res->forwardResolvesTo();

        memo[keytype(firstArg, funcName)] = concrete;

        return concrete;
    }

    void repr(instance_ptr self, ReprAccumulator& stream, bool isStr) {
        stream << m_name;
    }

    typed_python_hash_type hash(instance_ptr left) {
        return m_first_arg->hash(left);
    }

    size_t deepBytecountConcrete(instance_ptr instance, std::unordered_set<void*>& alreadyVisited, std::set<Slab*>* outSlabs) {
        return m_first_arg->deepBytecount(instance, alreadyVisited, outSlabs);
    }

    void deepcopyConcrete(
        instance_ptr dest,
        instance_ptr src,
        DeepcopyContext& context
    ) {
        m_first_arg->deepcopy(dest, src, context);
    }

    template<class buf_t>
    void deserialize(instance_ptr self, buf_t& buffer, size_t wireType) {
        m_first_arg->deserialize(self, buffer, wireType);
    }

    template<class buf_t>
    void serialize(instance_ptr self, buf_t& buffer, size_t fieldNumber) {
        m_first_arg->serialize(self, buffer, fieldNumber);
    }

    bool cmp(instance_ptr left, instance_ptr right, int pyComparisonOp, bool suppressExceptions) {
        return m_first_arg->cmp(left, right, pyComparisonOp, suppressExceptions);
    }

    void constructor(instance_ptr self) {
        m_first_arg->constructor(self);
    }

    void destroy(instance_ptr self) {
        m_first_arg->destroy(self);
    }

    void copy_constructor(instance_ptr self, instance_ptr other) {
        m_first_arg->copy_constructor(self, other);
    }

    void assign(instance_ptr self, instance_ptr other) {
        m_first_arg->assign(self, other);
    }

    Type* getFirstArgType() const {
        return m_first_arg;
    }

    std::string getFuncName() const {
        return m_funcName;
    }

    Function* getFunction() const {
        if (m_first_arg->getTypeCategory() == Type::TypeCategory::catClass) {
            Class* c = (Class*)m_first_arg;

            auto it = c->getMemberFunctions().find(m_funcName);

            if (it != c->getMemberFunctions().end()) {
                return it->second;
            }

            return nullptr;
        }

        if (m_first_arg->getTypeCategory() == Type::TypeCategory::catHeldClass) {
            HeldClass* c = (HeldClass*)m_first_arg;

            auto it = c->getMemberFunctions().find(m_funcName);

            if (it != c->getMemberFunctions().end()) {
                return it->second;
            }

            return nullptr;
        }

        if (m_first_arg->getTypeCategory() == Type::TypeCategory::catRefTo) {
            HeldClass* c = (HeldClass*)((RefTo*)m_first_arg)->getEltType();

            auto it = c->getMemberFunctions().find(m_funcName);

            if (it != c->getMemberFunctions().end()) {
                return it->second;
            }

            return nullptr;
        }

        if (m_first_arg->getTypeCategory() == Type::TypeCategory::catAlternative ||
                m_first_arg->getTypeCategory() == Type::TypeCategory::catConcreteAlternative) {
            Alternative* a;

            if (m_first_arg->getTypeCategory() == Type::TypeCategory::catAlternative) {
                a = (Alternative*)m_first_arg;
            } else {
                a = ((ConcreteAlternative*)m_first_arg)->getAlternative();
            }

            auto it = a->getMethods().find(m_funcName);

            if (it != a->getMethods().end()) {
                return it->second;
            }

            return nullptr;
        }

        return nullptr;
    }

    std::string m_funcName;

    Type* m_first_arg;
};
