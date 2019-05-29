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

// forward types must be resolved (removed from the graph) before
// any types that contain them can be used.
class Forward : public Type {
public:
    Forward() :
        Type(TypeCategory::catForward),
        mTarget(nullptr)
    {
        m_name = "unnamed";
        m_resolved = false;
        forwardTypesMayHaveChanged();
    }

    Forward(std::string name) :
        Type(TypeCategory::catForward),
        mTarget(nullptr)
    {
        m_resolved = false;
        m_name = name;
        forwardTypesMayHaveChanged();
    }

    void _forwardTypesMayHaveChanged() {
        if (mTarget) {
            m_resolved = true;
            m_size = mTarget->bytecount();
            m_name = mTarget->name();
        }
    }

    static Forward* Make() {
        return new Forward();
    }

    static Forward* Make(std::string name) {
        return new Forward(name);
    }

    Type* define(Type* target) {
        // TODO: check for containment cycles
        mTarget = target;
        calc_internals_and_propagate();

        return mTarget;
    }

    Type* getTarget() const {
        return mTarget;
    }

    template<class visitor_type>
    void _visitContainedTypes(const visitor_type& v) {
//        if (mTarget)
//            v(mTarget);
    }

    template<class visitor_type>
    void _visitReferencedTypes(const visitor_type& v) {
//        if (mTarget)
//            v(mTarget);
    }

    // TODO: is this still needed?
    void resolveDuringSerialization(Type* newTarget) {
        if (mTarget && mTarget != newTarget) {
            throw std::runtime_error("can't resolve a forward type to a new value.");
        }

        mTarget = newTarget;
    }

private:
    Type* mTarget;
};
