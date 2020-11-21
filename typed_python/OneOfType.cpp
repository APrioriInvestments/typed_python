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

bool OneOfType::isBinaryCompatibleWithConcrete(Type* other) {
    if (other->getTypeCategory() != TypeCategory::catOneOf) {
        return false;
    }

    OneOfType* otherO = (OneOfType*)other;

    if (m_types.size() != otherO->m_types.size()) {
        return false;
    }

    for (long k = 0; k < m_types.size(); k++) {
        if (!m_types[k]->isBinaryCompatibleWith(otherO->m_types[k])) {
            return false;
        }
    }

    return true;
}

bool OneOfType::_updateAfterForwardTypesChanged() {
    size_t size = computeBytecount();
    std::string name = computeName();

    if (m_is_recursive_forward) {
        name = m_recursive_name;
    }

    bool is_default_constructible = false;

    for (auto typePtr: m_types) {
        if (typePtr->is_default_constructible()) {
            is_default_constructible = true;
            break;
        }
    }

    bool anyChanged = (
        size != m_size ||
        name != m_name ||
        is_default_constructible != m_is_default_constructible
    );

    m_size = size;
    m_stripped_name = "";
    m_name = name;
    m_stripped_name = "";

    m_is_default_constructible = is_default_constructible;

    return anyChanged;
}

std::string OneOfType::computeName() const {
    std::string res = "OneOf(";

    bool first = true;

    for (auto t: m_types) {
        if (first) {
            first = false;
        } else {
            res += ", ";
        }

        res += t->name(true);
    }

    res += ")";

    return res;
}

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
OneOfType* OneOfType::Make(const std::vector<Type*>& types, OneOfType* knownType) {
    std::vector<Type*> flat_typelist;
    std::set<Type*> seen;

    //make sure we only get each type once and don't have any other 'OneOfType' in there...
    std::function<void (const std::vector<Type*>)> visit = [&](const std::vector<Type*>& subvec) {
        for (auto t: subvec) {
            if (t->getTypeCategory() == catOneOf) {
                visit( ((OneOfType*)t)->getTypes() );
            } else if (seen.find(t) == seen.end()) {
                flat_typelist.push_back(t);
                seen.insert(t);
            }
        }
    };

    visit(types);

    PyEnsureGilAcquired getTheGil;

    typedef const std::vector<Type*> keytype;

    static std::map<keytype, OneOfType*> m;

    auto it = m.find(flat_typelist);
    if (it == m.end()) {
        it = m.insert(std::make_pair(flat_typelist, knownType ? knownType : new OneOfType(flat_typelist))).first;
    }

    return it->second;
}
