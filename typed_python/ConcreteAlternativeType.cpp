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

#include "AllTypes.hpp"

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

void ConcreteAlternative::_forwardTypesMayHaveChanged() {
    m_base = m_alternative;
    m_name = m_alternative->name() + "." + m_alternative->subtypes()[m_which].first;
    m_size = m_alternative->bytecount();
    m_is_default_constructible = m_alternative->subtypes()[m_which].second->is_default_constructible();
}

void ConcreteAlternative::constructor(instance_ptr self) {
    if (m_alternative->all_alternatives_empty()) {
        *(uint8_t*)self = m_which;
    } else {
        constructor(self, [&](instance_ptr i) {
            m_alternative->subtypes()[m_which].second->constructor(i);
        });
    }
}

// static
ConcreteAlternative* ConcreteAlternative::Make(Alternative* alt, int64_t which) {
    static std::mutex guard;

    std::lock_guard<std::mutex> lock(guard);

    typedef std::pair<Alternative*, int64_t> keytype;

    static std::map<keytype, ConcreteAlternative*> m;

    auto it = m.find(keytype(alt ,which));

    if (it == m.end()) {
        it = m.insert(
            std::make_pair(keytype(alt,which), new ConcreteAlternative(alt,which))
            ).first;
    }

    return it->second;
}

