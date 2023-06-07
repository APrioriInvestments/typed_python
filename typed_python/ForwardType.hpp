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

PyDoc_STRVAR(Forward_doc,
    "Forward(n) -> new forward type named n\n"
    "\n"
    "Forward types must be resolved before any types that contain them can be used.\n"
    );

// forward types must be resolved (removed from the graph) before
// any types that contain them can be used.
class Forward : public Type {
public:
    Forward(std::string name) :
        Type(TypeCategory::catForward),
        mTarget(nullptr)
    {
        m_name = name;
        m_is_forward_defined = true;
    }

    const char* docConcrete() {
        return Forward_doc;
    }

    std::string moduleNameConcrete() {
        if (mTarget) {
            return mTarget->moduleNameConcrete();
        }

        return "<unresolved>";
    }

    std::string nameWithModuleConcrete() {
        if (mTarget) {
            return mTarget->nameWithModule();
        }

        return m_name;
    }

    static Forward* Make() {
        return Make("unnamed");
    }

    static Forward* Make(std::string name) {
        return new Forward(name);
    }

    Type* getTargetTransitive() {
        if (!mTarget) {
            return nullptr;
        }

        if (!mTarget->isForward()) {
            return mTarget;
        }

        std::set<Type*> seen;

        Type* curFwd = this;

        while (curFwd->isForward()) {
            if (seen.find(curFwd) != seen.end()) {
                throw std::runtime_error("Forward cycle detected.");
            }

            seen.insert(curFwd);

            curFwd = ((Forward*)curFwd)->getTarget();

            if (!curFwd) {
                return nullptr;
            }
        }

        return curFwd;
    }

    Type* define(Type* target) {
        if (!target) {
            throw std::runtime_error("Can't resolve a Forward to the nullptr");
        }

        if (mTarget) {
            throw std::runtime_error("Forward is already resolved.");
        }

        mTarget = target;

        return target;
    }

    Type* getTarget() const {
        return mTarget;
    }

    template<class visitor_type>
    void _visitCompilerVisibleInternals(const visitor_type& v) {
        v.visitHash(ShaHash(1, m_typeCategory));
        if (mTarget) {
            v.visitHash(ShaHash(1));
            v.visitTopo(mTarget);
        } else {
            v.visitHash(ShaHash(2));
        }
    }

    template<class visitor_type>
    void _visitReferencedTypes(const visitor_type& v) {
        if (mTarget)
            v(mTarget);
    }

    void constructor(instance_ptr self) {
        throw std::runtime_error("Forward types should never be explicity instantiated.");
    }

    void destroy(instance_ptr self) {
        throw std::runtime_error("Forward types should never be explicity instantiated.");
    }

    void copy_constructor(instance_ptr self, instance_ptr other) {
        throw std::runtime_error("Forward types should never be explicity instantiated.");
    }

    void assign(instance_ptr self, instance_ptr other) {
        throw std::runtime_error("Forward types should never be explicity instantiated.");
    }

    template<class buf_t>
    void deserialize(instance_ptr self, buf_t& buffer, size_t wireType) {
        throw std::runtime_error("Forward types should never be explicity instantiated.");
    }

    template<class buf_t>
    void serialize(instance_ptr self, buf_t& buffer, size_t fieldNumber) {
        throw std::runtime_error("Forward types should never be explicity instantiated.");
    }

private:
    Type* mTarget;
};
