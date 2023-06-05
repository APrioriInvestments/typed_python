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
ConcreteAlternative* ConcreteAlternative::Make(Alternative* alt, int64_t which) {
    if (alt->isForwardDefined()) {
        return new ConcreteAlternative(alt, which);
    }

    PyEnsureGilAcquired getTheGil;

    typedef std::pair<Alternative*, int64_t> keytype;

    static std::map<keytype, ConcreteAlternative*> memo;

    auto it = memo.find(keytype(alt, which));

    if (it != memo.end()) {
        return it->second;
    }

    ConcreteAlternative* res = new ConcreteAlternative(alt, which);
    ConcreteAlternative* concrete = (ConcreteAlternative*)res->forwardResolvesTo();

    memo[keytype(alt, which)] = concrete;
    return concrete;
}
