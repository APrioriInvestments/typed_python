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

class BoundMethod : public Type {
public:
    BoundMethod(Type* inFirstArg, std::string funcName) : Type(TypeCategory::catBoundMethod)
    {
        m_funcName = funcName;
        m_is_default_constructible = false;
        m_first_arg = inFirstArg;
        m_size = inFirstArg->bytecount();
        m_is_simple = false;

        endOfConstructorInitialization(); // finish initializing the type object.
    }

    ShaHash _computeIdentityHash(MutuallyRecursiveTypeGroup* groupHead = nullptr) {
        return (
            ShaHash(1, m_typeCategory) +
            m_first_arg->identityHash(groupHead) +
            ShaHash(m_funcName)
        );
    }

    template<class visitor_type>
    void _visitContainedTypes(const visitor_type& visitor) {
        visitor(m_first_arg);
    }

    template<class visitor_type>
    void _visitReferencedTypes(const visitor_type& visitor) {
        visitor(m_first_arg);
    }

    bool _updateAfterForwardTypesChanged() {
        bool anyChanged = false;

        // note that you can't have '.' in the name of a type, so we use '::'.
        // otherwise, the __name__ attribute of the type gets cut off at the last '.' and
        // looks like a memory corruption issue. It also prevents you from knowing what
        // type you're looking at.
        std::string name = "BoundMethod(" + m_first_arg->name(true) + ", " + m_funcName + ")";
        size_t size = m_first_arg->bytecount();

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
        BoundMethod::Make(m_first_arg, m_funcName, this);
    }

    static BoundMethod* Make(Type* c, std::string funcName, BoundMethod* knownType=nullptr) {
        PyEnsureGilAcquired getTheGil;

        typedef std::pair<Type*, std::string> keytype;

        static std::map<keytype, BoundMethod*> m;

        auto it = m.find(keytype(c, funcName));

        if (it == m.end()) {
            it = m.insert(
                std::make_pair(keytype(c, funcName), knownType ? knownType : new BoundMethod(c, funcName))
            ).first;
        }

        return it->second;
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
