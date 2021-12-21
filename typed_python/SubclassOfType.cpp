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

bool SubclassOfType::_updateAfterForwardTypesChanged() {
    size_t size = sizeof(Type*);
    std::string name = computeName();

    if (m_is_recursive_forward) {
        name = m_recursive_name;
    }

    bool is_default_constructible = true;

    bool anyChanged = (
        size != m_size ||
        name != m_name ||
        is_default_constructible != m_is_default_constructible
    );

    m_size = size;
    m_name = name;
    m_stripped_name = "";

    m_is_default_constructible = is_default_constructible;

    return anyChanged;
}

std::string SubclassOfType::computeName() const {
    return "SubclassOf(" + m_subclassOf->name(true) + ")";
}

void SubclassOfType::repr(instance_ptr self, ReprAccumulator& stream, bool isStr) {
    stream << "<class '" + ((Type**)self)[0]->name() + "'>";
}

typed_python_hash_type SubclassOfType::hash(instance_ptr left) {
    HashAccumulator acc;
    acc.addBytes(left, sizeof(Type*));
    return acc.get();
}

bool SubclassOfType::cmp(instance_ptr left, instance_ptr right, int pyComparisonOp, bool suppressExceptions) {
    if (pyComparisonOp == Py_EQ) {
        return ((Type**)left)[0] == ((Type**)right)[0];
    }

    if (pyComparisonOp == Py_NE) {
        return ((Type**)left)[0] != ((Type**)right)[0];
    }

    if (suppressExceptions) {
        return false;
    }

    throw std::runtime_error("comparison not supported between Type objects.");
}

void SubclassOfType::constructor(instance_ptr self) {
    assertForwardsResolvedSufficientlyToInstantiate();

    *((Type**)self) = m_subclassOf;
}

void SubclassOfType::destroy(instance_ptr self) {
}

void SubclassOfType::copy_constructor(instance_ptr self, instance_ptr other) {
    *((Type**)self) = *((Type**)other);
}

void SubclassOfType::assign(instance_ptr self, instance_ptr other) {
    *((Type**)self) = *((Type**)other);
}

// static
SubclassOfType* SubclassOfType::Make(Type* subclassOf, SubclassOfType* knownType) {
    PyEnsureGilAcquired getTheGil;

    static std::map<Type*, SubclassOfType*> m;

    auto it = m.find(subclassOf);

    if (it == m.end()) {
        it = m.insert(std::make_pair(subclassOf, knownType ? knownType : new SubclassOfType(subclassOf))).first;
    }

    return it->second;
}
