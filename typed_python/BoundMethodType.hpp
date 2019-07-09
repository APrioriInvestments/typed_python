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
#include "ReprAccumulator.hpp"

class BoundMethod : public Type {
public:
    BoundMethod(Type* inFirstArg, Function* inFunc) : Type(TypeCategory::catBoundMethod)
    {
        m_function = inFunc;
        m_is_default_constructible = false;
        m_first_arg = inFirstArg;
        m_size = inFirstArg->bytecount();
        m_is_simple = false;

        endOfConstructorInitialization(); // finish initializing the type object.
    }

    template<class visitor_type>
    void _visitContainedTypes(const visitor_type& visitor) {
        visitor(m_first_arg);
    }

    template<class visitor_type>
    void _visitReferencedTypes(const visitor_type& visitor) {
        visitor(m_first_arg);
        Type* f = m_function;
        visitor(f);
        assert(f == m_function);
    }

    bool _updateAfterForwardTypesChanged() {
        bool anyChanged = false;

        std::string name = "BoundMethod(" + m_first_arg->name() + "." + m_function->name() + ")";
        size_t size = m_first_arg->bytecount();

        anyChanged = (
            name != m_name ||
            size != m_size
        );

        m_name = name;
        m_size = size;

        return anyChanged;
    }

    static BoundMethod* Make(Type* c, Function* f) {
        static std::mutex guard;

        std::lock_guard<std::mutex> lock(guard);

        typedef std::pair<Type*, Function*> keytype;

        static std::map<keytype, BoundMethod*> m;

        auto it = m.find(keytype(c,f));

        if (it == m.end()) {
            it = m.insert(
                std::make_pair(keytype(c,f), new BoundMethod(c, f))
                ).first;
        }

        return it->second;
    }

    void repr(instance_ptr self, ReprAccumulator& stream) {
        stream << m_name;
    }

    typed_python_hash_type hash(instance_ptr left) {
        return m_first_arg->hash(left);
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
        return m_first_arg->cmp(left,right,pyComparisonOp, suppressExceptions);
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

    Function* getFunction() const {
        return m_function;
    }

private:
    Function* m_function;
    Type* m_first_arg;
};



