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

// forward types are never actually used - they must be removed from the graph before
// any types that contain them can be used.
class Forward : public Type {
public:
    Forward(PyObject* deferredDefinition, std::string name) :
        Type(TypeCategory::catForward),
        mTarget(nullptr),
        mDefinition(deferredDefinition)
    {
        m_references_unresolved_forwards = true;
        m_name = name;
    }

    // this constructor is for c++ direct types
    Forward(std::string name) :
        Type(TypeCategory::catForward),
        mTarget(nullptr),
        mDefinition(nullptr)
    {
        m_references_unresolved_forwards = true;
        m_name = name;
    }

    // this method is for c++ direct types
    void setTarget(Type* target) {
        mTarget = target;
    }

    Type* getTarget() const {
        return mTarget;
    }

    template<class resolve_py_callable_to_type>
    Type* guaranteeForwardsResolvedConcrete(resolve_py_callable_to_type& resolver) {
        if (mTarget) {
            return mTarget;
        }

        Type* t = resolver(mDefinition);
        if (!t) {
            m_failed_resolution = true;
        }

        mTarget = t;

        if (mTarget) {
            return mTarget;
        }

        return this;
    }

    template<class visitor_type>
    void _visitContainedTypes(const visitor_type& v) {
        v(mTarget);
    }

    template<class visitor_type>
    void _visitReferencedTypes(const visitor_type& v) {
        v(mTarget);
    }

    void _forwardTypesMayHaveChanged() {
        // This is only meant to be triggered when we are resolving c++ forward references,
        // (which lack the python mDefinition).
        // Without resetting m_references_unresolved_forwards, we would only resolve a single c++ reference
        // Maybe there is a better place for this.
        if (!mDefinition)
            m_references_unresolved_forwards = true;
    }

    void resolveDuringSerialization(Type* newTarget) {
        if (mTarget && mTarget != newTarget) {
            throw std::runtime_error("can't resolve a forward type to a new value.");
        }

        mTarget = newTarget;
    }

private:
    Type* mTarget;
    PyObject* mDefinition;
};
