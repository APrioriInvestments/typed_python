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

bool HeldClass::isBinaryCompatibleWithConcrete(Type* other) {
    if (other->getTypeCategory() != m_typeCategory) {
        return false;
    }

    HeldClass* otherO = (HeldClass*)other;

    if (m_members.size() != otherO->m_members.size()) {
        return false;
    }

    for (long k = 0; k < m_members.size(); k++) {
        if (std::get<0>(m_members[k]) != std::get<0>(otherO->m_members[k]) ||
                !std::get<1>(m_members[k])->isBinaryCompatibleWith(std::get<1>(otherO->m_members[k]))) {
            return false;
        }
    }

    return true;
}

void HeldClass::_forwardTypesMayHaveChanged() {
    m_is_default_constructible = true;
    m_byte_offsets.clear();

    //first m_members.size() bits (rounded up to nearest byte) contains the initialization flags.
    m_size = int((m_members.size() + 7) / 8); //round up to nearest byte

    for (auto t: m_members) {
        m_byte_offsets.push_back(m_size);
        m_size += std::get<1>(t)->bytecount();
    }
}

bool HeldClass::cmp(instance_ptr left, instance_ptr right, int pyComparisonOp) {
    uint64_t leftPtr = *(uint64_t*)left;
    uint64_t rightPtr = *(uint64_t*)right;

    if (leftPtr < rightPtr) {
        return cmpResultToBoolForPyOrdering(pyComparisonOp, -1);
    }
    if (leftPtr > rightPtr) {
        return cmpResultToBoolForPyOrdering(pyComparisonOp, 1);
    }

    return cmpResultToBoolForPyOrdering(pyComparisonOp, 0);
}

void HeldClass::repr(instance_ptr self, ReprAccumulator& stream) {
    PushReprState isNew(stream, self);

    if (!isNew) {
        stream << m_name << "(" << (void*)self << ")";
        return;
    }

    stream << m_name << "(";

    for (long k = 0; k < m_members.size();k++) {
        if (k > 0) {
            stream << ", ";
        }

        if (checkInitializationFlag(self, k)) {
            stream << std::get<0>(m_members[k]) << "=";

            std::get<1>(m_members[k])->repr(eltPtr(self,k),stream);
        } else {
            stream << std::get<0>(m_members[k]) << " not initialized";
        }
    }
    if (m_members.size() == 1) {
        stream << ",";
    }

    stream << ")";
}

int32_t HeldClass::hash32(instance_ptr left) {
    Hash32Accumulator acc((int)getTypeCategory());

    //hash the class pointer, since the values within the class can change.
    acc.addRegister(*(uint64_t*)left);

    return acc.get();
}

void HeldClass::setAttribute(instance_ptr self, int memberIndex, instance_ptr other) const {
    Type* member_t = std::get<1>(m_members[memberIndex]);
    if (checkInitializationFlag(self, memberIndex)) {
        member_t->assign(
            eltPtr(self, memberIndex),
            other
            );
    } else {
        member_t->copy_constructor(
            eltPtr(self, memberIndex),
            other
            );
        setInitializationFlag(self, memberIndex);
    }
}

void HeldClass::emptyConstructor(instance_ptr self) {
    //more efficient would be to just write over the bytes directly.
    for (size_t k = 0; k < m_members.size(); k++) {
        clearInitializationFlag(self, k);
    }
}

void HeldClass::constructor(instance_ptr self) {
    for (size_t k = 0; k < m_members.size(); k++) {
        Type* member_t = std::get<1>(m_members[k]);

        if (wantsToDefaultConstruct(member_t)) {
            if (memberHasDefaultValue(k)) {
                member_t->copy_constructor(self+m_byte_offsets[k], getMemberDefaultValue(k).data());
            } else {
                member_t->constructor(self+m_byte_offsets[k]);
            }
            setInitializationFlag(self, k);
        } else {
            clearInitializationFlag(self, k);
        }
    }
}

void HeldClass::destroy(instance_ptr self) {
    for (long k = (long)m_members.size() - 1; k >= 0; k--) {
        Type* member_t = std::get<1>(m_members[k]);
        if (checkInitializationFlag(self, k)) {
            member_t->destroy(self+m_byte_offsets[k]);
        }
    }
}

void HeldClass::copy_constructor(instance_ptr self, instance_ptr other) {
    for (long k = (long)m_members.size() - 1; k >= 0; k--) {
        Type* member_t = std::get<1>(m_members[k]);
        if (checkInitializationFlag(other, k)) {
            member_t->copy_constructor(self+m_byte_offsets[k], other+m_byte_offsets[k]);
            setInitializationFlag(self, k);
        }
    }
}

void HeldClass::assign(instance_ptr self, instance_ptr other) {
    for (long k = (long)m_members.size() - 1; k >= 0; k--) {
        bool selfInit = checkInitializationFlag(self,k);
        bool otherInit = checkInitializationFlag(other,k);
        Type* member_t = std::get<1>(m_members[k]);
        if (selfInit && otherInit) {
            member_t->assign(self + m_byte_offsets[k], other+m_byte_offsets[k]);
        }
        else if (selfInit && !otherInit) {
            member_t->destroy(self+m_byte_offsets[k]);
            clearInitializationFlag(self, k);
        } else if (!selfInit && otherInit) {
            member_t->copy_constructor(self + m_byte_offsets[k], other+m_byte_offsets[k]);
            clearInitializationFlag(self, k);
        }
    }
}

void HeldClass::setInitializationFlag(instance_ptr self, int memberIndex) const {
    int byte = memberIndex / 8;
    int bit = memberIndex % 8;
    uint8_t mask = (1 << bit);
    ((uint8_t*)self)[byte] |= mask;
}

void HeldClass::clearInitializationFlag(instance_ptr self, int memberIndex) const {
    int byte = memberIndex / 8;
    int bit = memberIndex % 8;
    uint8_t mask = (1 << bit);
    ((uint8_t*)self)[byte] &= ~mask;
}

int HeldClass::memberNamed(const char* c) const {
    for (long k = 0; k < m_members.size(); k++) {
        if (std::get<0>(m_members[k]) == c) {
            return k;
        }
    }

    return -1;
}

