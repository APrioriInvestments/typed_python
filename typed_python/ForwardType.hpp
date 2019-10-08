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

// forward types must be resolved (removed from the graph) before
// any types that contain them can be used.
class Forward : public Type {
public:
    Forward(std::string name = "unnamed") :
        Type(TypeCategory::catForward),
        mTarget(nullptr)

    {
        m_name = name;

        // deliberately don't invoke 'endOfConstructorInitialization'
    }

    static Forward* Make() {
        return new Forward();
    }

    static Forward* Make(std::string name) {
        return new Forward(name);
    }

    Type* define(Type* target) {
        if (!target) {
            throw std::runtime_error("Can't resolve a Forward to the nullptr");
        }

        while (target->getTypeCategory() == TypeCategory::catForward) {
            if (((Forward*)target)->getTarget()) {
                target = ((Forward*)target)->getTarget();
            } else {
                throw std::runtime_error("Please resolve forwards to concrete types only.");
            }
        }

        if (target == mTarget) {
            return mTarget;
        }

        // check the containment graph. If we are a forward that's
        // directly contained by the target, then we shouldn't allow this
        // definition because we'd be infinitely large.
        auto& containedForwards = target->getContainedForwards();

        if (containedForwards.find(this) != containedForwards.end()) {
            throw std::runtime_error("Can't resolve forward " + m_name + " to " +
                target->name() + " because it would create a type-containment cycle" +
                " (specifically, the type would have an infinite bytecount because it 'contains' itself)."
            );
        }

        bool thisIsRecursive = target->getReferencedForwards().find(this) != target->getReferencedForwards().end();

        mTarget = target;
        m_resolved = true;

        if (thisIsRecursive) {
            target->setNameForRecursiveType(m_name);
            target->_updateAfterForwardTypesChanged();
        }

        // we are not yet resolved
        for (auto typePtr: m_referencing_us_indirectly) {
            typePtr->forwardResolvedTo(this, target);
        }

        bool anyChanged = true;

        while (anyChanged) {
            anyChanged = false;

            for (auto typePtr: m_referencing_us_indirectly) {
                if (typePtr->_updateAfterForwardTypesChanged()) {
                    anyChanged = true;
                }
            }
        }


        for (auto typePtr: m_referencing_us_indirectly) {
            if (typePtr->getReferencedForwards().size() == 0) {
                typePtr->forwardTypesAreResolved();
            }
        }

        m_name = mTarget->name();
        m_referencing_us_indirectly.clear();

        return target;
    }

    Type* getTarget() const {
        return mTarget;
    }

    template<class visitor_type>
    void _visitContainedTypes(const visitor_type& v) {
        if (mTarget)
            v(mTarget);
    }

    template<class visitor_type>
    void _visitReferencedTypes(const visitor_type& v) {
        if (mTarget)
            v(mTarget);
    }

    void markIndirectForwardUse(Type* user) {
        if (m_resolved) {
            throw std::runtime_error("already resolved forward type " + name() + " can't be used like this.");
        }
        m_referencing_us_indirectly.insert(user);
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

    const std::set<Type*> getReferencing() const {
        return m_referencing_us_indirectly;
    }
private:
    Type* mTarget;

    std::set<Type*> m_referencing_us_indirectly;
};
