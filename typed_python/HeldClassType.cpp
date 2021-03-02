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

bool HeldClass::isBinaryCompatibleWithConcrete(Type* other) {
    if (other->getTypeCategory() != m_typeCategory) {
        return false;
    }

    HeldClass* otherO = (HeldClass*)other;

    if (m_members.size() != otherO->m_members.size()) {
        return false;
    }

    for (long k = 0; k < m_members.size(); k++) {
        if (m_members[k].getName() != otherO->m_members[k].getName() ||
                !m_members[k].getType()->isBinaryCompatibleWith(otherO->m_members[k].getType())) {
            return false;
        }
    }

    return true;
}

void HeldClass::updateBytesOfInitBits() {
    bool anyMembersWithInitializers = false;
    for (auto& m: m_members) {
        if (!m.getIsNonempty()) {
            anyMembersWithInitializers = true;
        }
    }

    if (!anyMembersWithInitializers) {
        mBytesOfInitializationBits = 0;
    } else {
        mBytesOfInitializationBits = int((m_members.size() + 7) / 8); //round up to nearest byte
    }
}

bool HeldClass::_updateAfterForwardTypesChanged() {
    m_byte_offsets.clear();

    updateBytesOfInitBits();

    size_t size = mBytesOfInitializationBits;

    for (auto t: m_members) {
        m_byte_offsets.push_back(size);
        size += t.getType()->bytecount();
    }

    bool anyChanged = size != m_size;

    m_is_default_constructible = m_memberFunctions.find("__init__") == m_memberFunctions.end();
    m_size = size;

    return anyChanged;
}

bool HeldClass::cmp(instance_ptr left, instance_ptr right, int pyComparisonOp, bool suppressExceptions) {
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

void HeldClass::repr(instance_ptr self, ReprAccumulator& stream, bool isStr, bool isClassNotHeldClass) {
    PushReprState isNew(stream, self);

    std::string name = isClassNotHeldClass ? getClassType()->name() : m_name;

    if (!isNew) {
        stream << name << "(" << (void*)self << ")";
        return;
    }

    stream << name << "(";

    for (long k = 0; k < m_members.size();k++) {
        if (k > 0) {
            stream << ", ";
        }

        if (checkInitializationFlag(self, k)) {
            stream << m_members[k].getName() << "=";

            m_members[k].getType()->repr(eltPtr(self,k), stream, false);
        } else {
            stream << m_members[k].getName() << " not initialized";
        }
    }
    if (m_members.size() == 1) {
        stream << ",";
    }

    stream << ")";
}

typed_python_hash_type HeldClass::hash(instance_ptr left) {
    HashAccumulator acc((int)getTypeCategory());

    //hash our address
    acc.addRegister((uint64_t)left);

    return acc.get();
}

RefTo* HeldClass::getRefToType() {
    if (!m_refToType) {
        m_refToType = RefTo::Make(this);
    }

    return m_refToType;
}

void HeldClass::delAttribute(instance_ptr self, int memberIndex) const {
    Type* member_t = m_members[memberIndex].getType();
    if (checkInitializationFlag(self, memberIndex)) {
        member_t->destroy(eltPtr(self, memberIndex));
        clearInitializationFlag(self, memberIndex);
    }
}

void HeldClass::setAttribute(instance_ptr self, int memberIndex, instance_ptr other) const {
    Type* member_t = m_members[memberIndex].getType();
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

void HeldClass::constructor(instance_ptr self, bool allowEmpty) {
    if (!m_is_default_constructible and !allowEmpty) {
        throw std::runtime_error(m_name + " is not default-constructible");
    }

    for (size_t k = 0; k < m_members.size(); k++) {
        Type* member_t = m_members[k].getType();

        if (memberHasDefaultValue(k)) {
            member_t->copy_constructor(self+m_byte_offsets[k], getMemberDefaultValue(k).data());
            setInitializationFlag(self, k);
        }
        else if (wantsToDefaultConstruct(member_t) || m_members[k].getIsNonempty()) {
            member_t->constructor(self+m_byte_offsets[k]);
            setInitializationFlag(self, k);
        } else {
            clearInitializationFlag(self, k);
        }
    }
}

void HeldClass::destroy(instance_ptr self) {
    for (long k = (long)m_members.size() - 1; k >= 0; k--) {
        Type* member_t = m_members[k].getType();
        if (checkInitializationFlag(self, k)) {
            member_t->destroy(self+m_byte_offsets[k]);
        }
    }
}

void HeldClass::copy_constructor(instance_ptr self, instance_ptr other) {
    for (long k = (long)m_members.size() - 1; k >= 0; k--) {
        Type* member_t = m_members[k].getType();
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
        Type* member_t = m_members[k].getType();
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

bool HeldClass::checkInitializationFlag(instance_ptr self, int memberIndex) const {
    if (fieldGuaranteedInitialized(memberIndex)) {
        return true;
    }

    int byte = memberIndex / 8;
    int bit = memberIndex % 8;
    return bool( ((uint8_t*)self)[byte] & (1 << bit) );
}

void HeldClass::setInitializationFlag(instance_ptr self, int memberIndex) const {
    if (fieldGuaranteedInitialized(memberIndex)) {
        return;
    }

    int byte = memberIndex / 8;
    int bit = memberIndex % 8;
    uint8_t mask = (1 << bit);
    ((uint8_t*)self)[byte] |= mask;
}

void HeldClass::clearInitializationFlag(instance_ptr self, int memberIndex) const {
    if (fieldGuaranteedInitialized(memberIndex)) {
        return;
    }

    int byte = memberIndex / 8;
    int bit = memberIndex % 8;
    uint8_t mask = (1 << bit);
    ((uint8_t*)self)[byte] &= ~mask;
}

int HeldClass::memberNamed(const char* c) const {
    for (long k = 0; k < m_members.size(); k++) {
        if (m_members[k].getName() == c) {
            return k;
        }
    }

    return -1;
}

BoundMethod* HeldClass::getMemberFunctionMethodType(const char* attr, bool forHeld) {
    auto& methodTypeDict = m_memberFunctionMethodTypes[forHeld ? 1 : 0];

    if (methodTypeDict.size() != m_memberFunctions.size()) {
        for (auto name: m_memberFunctions) {
            // note that we explicitly leak the string so that the refcount on c_str
            // stays active. I'm sure there's a better way to do this, but types are
            // permanent, so we would never have cleaned this up anyways.
            methodTypeDict[(new std::string(name.first))->c_str()] =
                BoundMethod::Make(
                    forHeld ? (Type*)getRefToType() : (Type*)getClassType(),
                    name.first
                );
        }
    }

    auto it = methodTypeDict.find(attr);
    if (it != methodTypeDict.end()) {
        return it->second;
    }

    return nullptr;
}

void ClassDispatchTable::allocateUpcastDispatchTables() {
    mUpcastDispatches = (uint16_t*)tp_malloc(sizeof(uint16_t) * mInterfaceClass->getMro().size());

    for (long castToIx = 0; castToIx < mInterfaceClass->getMro().size(); castToIx++) {
        int mroIndex = mImplementingClass->getMroIndex(mInterfaceClass->getMro()[castToIx]);

        if (mroIndex < 0) {
            throw std::runtime_error("invalid MRO index encountered");
        }

        mUpcastDispatches[castToIx] = mroIndex;
    }
}


HeldClass* HeldClass::Make(
    std::string inName,
    const std::vector<HeldClass*>& bases,
    bool isFinal,
    const std::vector<MemberDefinition>& members,
    const std::map<std::string, Function*>& memberFunctions,
    const std::map<std::string, Function*>& staticFunctions,
    const std::map<std::string, Function*>& propertyFunctions,
    const std::map<std::string, PyObject*>& classMembers
) {
    //we only allow one base class to have members because we want native code to be
    //able to just find those values in subclasses without hitting the vtable.
    long countWithMembers = 0;

    for (auto base: bases) {
        if (base->m_members.size()) {
            countWithMembers++;
        }

        if (base->isFinal()) {
            throw std::runtime_error("Can't subclass " + base->getClassType()->name() + " because it's marked 'final'.");
        }
    }

    if (countWithMembers > 1) {
        throw std::runtime_error("Can't inherit from multiple base classes that both have members.");
    }

    if (!isFinal) {
        for (auto nameAndMemberFunc: memberFunctions) {
            for (auto overload: nameAndMemberFunc.second->getOverloads()) {
                if (!overload.getReturnType()) {
                    throw std::runtime_error("Overload of " + inName + "." + nameAndMemberFunc.first
                        + " has no return type, but the class is not marked final, so the compiler"
                        + " won't have a return type defined for this method. Either add an '-> object' annotation"
                        + " to the method, or mark the class Final (so that the compiler can apply type inference directly)"
                    );
                }
            }
        }
    }

    HeldClass* result = new HeldClass(
        "Held(" + inName + ")",
        bases,
        isFinal,
        members,
        memberFunctions,
        staticFunctions,
        propertyFunctions,
        classMembers
    );

    // we do these outside of the constructor so that if they throw we
    // don't destroy the HeldClass type object (and just leak it instead) because
    // we need to ensure we never delete Type objects.
    result->initializeMRO();

    result->endOfConstructorInitialization();

    return result;
}
