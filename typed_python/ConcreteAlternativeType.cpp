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
#include "../typed_python/Format.hpp"

bool ConcreteAlternative::isBinaryCompatibleWithConcrete(Type* other) {
    if (other->getTypeCategory() == TypeCategory::catConcreteAlternative) {
        ConcreteAlternative* otherO = (ConcreteAlternative*)other;

        return otherO->m_alternative->isBinaryCompatibleWith(m_alternative) &&
            m_which == otherO->m_which;
    }

    if (other->getTypeCategory() == TypeCategory::catAlternative) {
        return m_alternative->isBinaryCompatibleWith(other);
    }

    return false;
}

ShaHash ConcreteAlternative::_computeIdentityHash(MutuallyRecursiveTypeGroup* groupHead) {
    return ShaHash(1, m_typeCategory) + ShaHash(m_which) + m_alternative->identityHash(groupHead);
}

bool ConcreteAlternative::_updateAfterForwardTypesChanged() {
    if (m_which < 0 || m_which >= m_alternative->subtypes().size()) {
      throw std::runtime_error(
        "invalid alternative index: " +
        format(m_which) + " not in [0," +
        format(m_alternative->subtypes().size()) + ")"
      );
    }

    m_base = m_alternative;

    std::string name = m_alternative->name() + "." + m_alternative->subtypes()[m_which].first;
    size_t size = m_alternative->bytecount();
    bool is_default_constructible = m_alternative->subtypes()[m_which].second->is_default_constructible();

    bool anyChanged = (
        m_name != name ||
        m_size != size ||
        m_is_default_constructible != is_default_constructible
    );

    m_name = name;
    m_stripped_name = "";
    m_size = size;
    m_is_default_constructible = is_default_constructible;

    return anyChanged;
}

void ConcreteAlternative::constructor(instance_ptr self) {
    assertForwardsResolvedSufficientlyToInstantiate();

    if (m_alternative->all_alternatives_empty()) {
        *(uint8_t*)self = m_which;
    } else {
        constructor(self, [&](instance_ptr i) {
            m_alternative->subtypes()[m_which].second->constructor(i);
        });
    }
}

// static
ConcreteAlternative* ConcreteAlternative::Make(Alternative* alt, int64_t which, ConcreteAlternative* knownType) {
    PyEnsureGilAcquired getTheGil;

    typedef std::pair<Alternative*, int64_t> keytype;

    static std::map<keytype, ConcreteAlternative*> m;

    auto it = m.find(keytype(alt ,which));

    if (it == m.end()) {
        if (which < 0 || which >= alt->subtypes().size()) {
            throw std::runtime_error(
                "invalid alternative index: " +
                format(which) + " not in [0," +
                format(alt->subtypes().size()) + ")"
            );
        }

        it = m.insert(
            std::make_pair(
                keytype(alt,which),
                knownType ? knownType : new ConcreteAlternative(alt,which)
            )
        ).first;
    }

    return it->second;
}
