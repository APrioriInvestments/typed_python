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

#include "AllTypes.hpp"

void OneOfType::repr(instance_ptr self, ReprAccumulator& stream, bool isStr) {
    m_types[*((uint8_t*)self)]->repr(self+1, stream, isStr);
}

typed_python_hash_type OneOfType::hash(instance_ptr left) {
    return m_types[*((uint8_t*)left)]->hash(left+1);
}

bool OneOfType::cmp(instance_ptr left, instance_ptr right, int pyComparisonOp, bool suppressExceptions) {
    if (((uint8_t*)left)[0] < ((uint8_t*)right)[0]) {
        return cmpResultToBoolForPyOrdering(pyComparisonOp, -1);
    }
    if (((uint8_t*)left)[0] > ((uint8_t*)right)[0]) {
        return cmpResultToBoolForPyOrdering(pyComparisonOp, 1);
    }

    return m_types[*((uint8_t*)left)]->cmp(left+1,right+1, pyComparisonOp, suppressExceptions);
}

size_t OneOfType::computeBytecount() const {
    size_t res = 0;

    for (auto t: m_types)
        res = std::max(res, t->bytecount());

    return res + 1;
}

void OneOfType::constructor(instance_ptr self) {
    assertForwardsResolvedSufficientlyToInstantiate();
    if (!m_is_default_constructible) {
        throw std::runtime_error(m_name + " is not default-constructible");
    }

    for (size_t k = 0; k < m_types.size(); k++) {
        if (m_types[k]->is_default_constructible()) {
            *(uint8_t*)self = k;
            m_types[k]->constructor(self+1);
            return;
        }
    }
}

void OneOfType::destroy(instance_ptr self) {
    uint8_t which = *(uint8_t*)(self);
    m_types[which]->destroy(self+1);
}

void OneOfType::copy_constructor(instance_ptr self, instance_ptr other) {
    uint8_t which = *(uint8_t*)self = *(uint8_t*)other;
    m_types[which]->copy_constructor(self+1, other+1);
}

void OneOfType::assign(instance_ptr self, instance_ptr other) {
    uint8_t which = *(uint8_t*)self;
    if (which == *(uint8_t*)other) {
        m_types[which]->assign(self+1,other+1);
    } else {
        m_types[which]->destroy(self+1);

        uint8_t otherWhich = *(uint8_t*)other;
        *(uint8_t*)self = otherWhich;
        m_types[otherWhich]->copy_constructor(self+1,other+1);
    }
}

// static
OneOfType* OneOfType::Make(const std::vector<Type*>& types) {
    bool anyForward = false;

    for (auto t: types) {
        if (t->isForwardDefined()) {
            anyForward = true;
        }
    }

    if (anyForward) {
        return new OneOfType(types);
    }

    PyEnsureGilAcquired getTheGil;

    typedef const std::vector<Type*> keytype;
    static std::map<keytype, OneOfType*> memo;

    auto it = memo.find(types);
    if (it != memo.end()) {
        return it->second;
    }

    OneOfType* res = new OneOfType(types);
    OneOfType* concrete = (OneOfType*)res->forwardResolvesTo();

    memo[types] = concrete;
    return concrete;
}

Type* OneOfType::cloneForForwardResolutionConcrete() {
    // create a 'blank' oneof type
    return new OneOfType();
}

void OneOfType::initializeFromConcrete(
    Type* forwardDefinitionOfSelf
) {
    OneOfType* selfT = (OneOfType*)forwardDefinitionOfSelf;

    m_types = selfT->m_types;
}

void OneOfType::updateInternalTypePointersConcrete(
    const std::map<Type*, Type*>& groupMap
) {
    std::vector<Type*> newTypes;
    std::set<Type*> seenTypes;
    std::set<Type*> everVisited;

    std::function<void (Type*)> visit = [&](Type* t) {
        if (everVisited.find(t) != everVisited.end()) {
            return;
        }
        everVisited.insert(t);

        if (t->getTypeCategory() == catOneOf) {
            for (auto subt: ((OneOfType*)t)->getTypes()) {
                visit(subt);
            }
        } else {
            auto it = groupMap.find(t);
            if (it != groupMap.end()) {
                visit(it->second);
            } else {
                if (seenTypes.find(t) == seenTypes.end()) {
                    newTypes.push_back(t);
                    seenTypes.insert(t);
                }
            }
        }
    };

    for (auto t: m_types) {
        visit(t);
    }

    m_types = newTypes;

    if (m_types.size() > 255) {
        throw std::runtime_error("OneOf types are limited to 255 alternatives in this implementation");
    }
}

void OneOfType::postInitializeConcrete() {
    m_size = computeBytecount();

    m_is_default_constructible = false;

    for (auto typePtr: m_types) {
        if (typePtr->is_default_constructible()) {
            m_is_default_constructible = true;
            break;
        }
    }
}

std::string OneOfType::computeRecursiveNameConcrete(TypeStack& typeStack) {
    std::string res = "OneOf(";

    bool first = true;

    for (auto t: m_types) {
        if (first) {
            first = false;
        } else {
            res += ", ";
        }

        res += t->computeRecursiveName(typeStack);
    }

    res += ")";

    return res;
}
