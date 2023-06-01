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

MemberDefinition::MemberDefinition(
    const std::string& inName,
    Type* inType,
    const Instance& inDefaultValue,
    bool nonempty
) :
    mName(inName),
    mType(inType),
    mDefaultValue(inDefaultValue),
    mIsNonempty(nonempty)
{
    mDefaultValueAsPyobj = PyInstance::extractPythonObject(mDefaultValue);
}

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

bool HeldClass::cmp(instance_ptr left, instance_ptr right, int pyComparisonOp, bool suppressExceptions) {
    if (hasAnyComparisonOperators()) {
        auto it = getMemberFunctions().find(Class::pyComparisonOpToMethodName(pyComparisonOp));

        if (it != getMemberFunctions().end()) {
            //we found a user-defined method for this comparison function.
            PyObjectStealer leftAsPyObj(PyInstance::extractPythonObject(left, this));
            PyObjectStealer rightAsPyObj(PyInstance::extractPythonObject(right, this));

            std::pair<bool, PyObject*> res = PyFunctionInstance::tryToCall(
                it->second,
                nullptr,
                leftAsPyObj,
                rightAsPyObj
                );

            if (res.first && !res.second) {
                throw PythonExceptionSet();
            }

            int result = PyObject_IsTrue(res.second);
            decref(res.second);

            if (result == -1) {
                throw PythonExceptionSet();
            }

            return result != 0;
        }
    }

    if (pyComparisonOp == Py_NE) {
        return !cmp(left, right, Py_EQ, suppressExceptions);
    }

    if (pyComparisonOp == Py_EQ) {
        for (long k = 0; k < m_members.size(); k++) {
            bool lhsInit = checkInitializationFlag(left, k);
            bool rhsInit = checkInitializationFlag(right, k);

            if (lhsInit != rhsInit) {
                return false;
            }

            if (lhsInit && rhsInit) {
                if (!m_members[k].getType()->cmp(eltPtr(left, k), eltPtr(right, k), pyComparisonOp, suppressExceptions)) {
                    return false;
                }
            }
        }

        return true;
    }

    PyErr_Format(
        PyExc_TypeError,
        "'%s' not defined between instances of '%s' and '%s'",
        pyComparisonOp == Py_EQ ? "==" :
        pyComparisonOp == Py_NE ? "!=" :
        pyComparisonOp == Py_LT ? "<" :
        pyComparisonOp == Py_LE ? "<=" :
        pyComparisonOp == Py_GT ? ">" :
        pyComparisonOp == Py_GE ? ">=" : "?",
        name().c_str(),
        name().c_str()
    );

    throw PythonExceptionSet();
}

void HeldClass::repr(instance_ptr self, ReprAccumulator& stream, bool isStr, bool isClassNotHeldClass) {
    auto it = getMemberFunctions().find(isStr ? "__str__" : "__repr__");

    if (it != getMemberFunctions().end()) {
        PyEnsureGilAcquired acquireTheGil;

        PyObjectStealer selfAsPyObj(PyInstance::extractPythonObject(self, this));

        std::pair<bool, PyObject*> res = PyFunctionInstance::tryToCall(
            it->second,
            nullptr,
            selfAsPyObj
        );

        if (res.first) {
            if (!res.second) {
                throw PythonExceptionSet();
            }
            if (!PyUnicode_Check(res.second)) {
                decref(res.second);
                throw std::runtime_error(
                    stream.isStrCall() ? "__str__ returned a non-string" : "__repr__ returned a non-string"
                    );
            }

            stream << PyUnicode_AsUTF8(res.second);
            decref(res.second);

            return;
        }

        throw std::runtime_error(
            stream.isStrCall() ? "Found a __str__ method but failed to call it with 'self'"
                : "Found a __repr__ method but failed to call it with 'self'"
            );
    }

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
    if (m_hasDelMagicMethod) {
        PyEnsureGilAcquired getTheGil;

        // don't suppress errors
        PyErrorStasher stashErrors;

        Function* method = m_memberFunctions.find("__del__")->second;

        PyObjectStealer targetArgTuple(PyTuple_New(1));

        PyTuple_SetItem(
            targetArgTuple,
            0,
            PyInstance::initializeTemporaryRef(
                this,
                self
            )
        ); //steals a reference

        std::pair<bool, PyObject*> res = PyFunctionInstance::tryToCallAnyOverload(
            method,
            nullptr,
            nullptr,
            targetArgTuple,
            nullptr
        );

        // we just swallow any exceptions that get thrown here
        // probably we ought to log...
        if (res.first && !res.second) {
            PyErr_Clear();
        } else if (res.second) {
            decref(res.second);
        }
    }

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
        } else {
            clearInitializationFlag(self, k);
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
    // TODO: are we accidentally leaking forwards out of this? Probably.
    const std::map<std::string, PyObject*>& classMembers,
    const std::map<std::string, Function*>& classMethods
) {
    //we only allow one base class to have members because we want native code to be
    //able to just find those values in subclasses without hitting the vtable.
    std::vector<HeldClass*> withMembers;

    for (auto base: bases) {
        if (base->m_members.size()) {
            withMembers.push_back(base);
        }

        if (base->isFinal()) {
            PyErr_Format(
                PyExc_TypeError,
                "Can't subclass %s because it's marked 'Final'",
                base->getClassType()->name().c_str()
            );

            throw PythonExceptionSet();
        }

        for (auto nameAndCM: classMembers) {
            auto it = base->getOwnClassMembers().find(nameAndCM.first);
            if (it != base->getOwnClassMembers().end()
                && nameAndCM.first != "__typed_python_template__") {
                // check that they're the same object
                if (it->second != nameAndCM.second && nameAndCM.first != "__qualname__"
                    && nameAndCM.first != "__module__"
                    && nameAndCM.first != "__classcell__"
                ) {
                    PyErr_Format(
                        PyExc_TypeError,
                        "Class %s can't redefine classmember %s",
                        inName.c_str(),
                        nameAndCM.first.c_str()
                    );
                    throw PythonExceptionSet();
                }
            }
        }
    }

    if (withMembers.size() > 1) {
        PyErr_Format(
            PyExc_TypeError,
            "Class %s can't have data members because its base classes %s and %s both have members.",
            inName.c_str(),
            withMembers[0]->getClassType()->name().c_str(),
            withMembers[1]->getClassType()->name().c_str()
        );
        throw PythonExceptionSet();
    }

    // this is a forward-defined HeldClass
    HeldClass* result = new HeldClass(
        "Held(" + inName + ")",
        bases,
        isFinal,
        members,
        memberFunctions,
        staticFunctions,
        propertyFunctions,
        classMembers,
        classMethods
    );

    Class* clsType = new Class(inName, result);

    // initialize the corresonding Class type
    result->setClassType(clsType);

    // check if the forward has any references to forward types. If not, we can
    // (and should) resolve it immediately, since our contract is that we only produce forwards
    // if we refer to forwards.
    bool anyForward = false;

    result->_visitReferencedTypes(
        [&](Type* t) { if (t->isForwardDefined() && t != clsType) { anyForward = true; } }
    );

    if (!anyForward) {
        return (HeldClass*)result->forwardResolvesTo();
    }

    // we have a forward ref - we have to go through the normal duplicate-and-intern forward
    // construction process for this to work.
    return result;
}
