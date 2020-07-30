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

bool Alternative::isBinaryCompatibleWithConcrete(Type* other) {
    if (other->getTypeCategory() == TypeCategory::catConcreteAlternative) {
        other = other->getBaseType();
    }

    if (other->getTypeCategory() != m_typeCategory) {
        return false;
    }

    Alternative* otherO = (Alternative*)other;

    if (m_subtypes.size() != otherO->m_subtypes.size()) {
        return false;
    }

    for (long k = 0; k < m_subtypes.size(); k++) {
        if (m_subtypes[k].first != otherO->m_subtypes[k].first) {
            return false;
        }
        if (!m_subtypes[k].second->isBinaryCompatibleWith(otherO->m_subtypes[k].second)) {
            return false;
        }
    }

    return true;
}

int64_t Alternative::refcount(instance_ptr i) const {
    if (m_all_alternatives_empty) {
        return 0;
    }
    return ((layout**)i)[0]->refcount;
}

ShaHash Alternative::_computeIdentityHash(MutuallyRecursiveTypeGroup* groupHead) {
    ShaHash newHash(1, m_typeCategory);

    newHash = newHash + ShaHash(m_name) + ShaHash(0);
    for (auto& subtype_pair: m_subtypes) {
        newHash = newHash + ShaHash(subtype_pair.first) + subtype_pair.second->identityHash(groupHead);
    }
    newHash = newHash + ShaHash(1);
    for (auto nameAndMethod: m_methods) {
        newHash = newHash + ShaHash(nameAndMethod.first) + nameAndMethod.second->identityHash(groupHead);
    }
    return newHash;
}

bool Alternative::_updateAfterForwardTypesChanged() {
    m_arg_positions.clear();
    m_default_construction_type = nullptr;

    bool is_default_constructible = false;
    bool all_alternatives_empty = true;
    int default_construction_ix = 0;

    for (auto& subtype_pair: m_subtypes) {
        if (subtype_pair.second->bytecount() > 0) {
            all_alternatives_empty = false;
        }

        if (m_arg_positions.find(subtype_pair.first) != m_arg_positions.end()) {
            throw std::runtime_error("Can't create an alternative with " +
                    subtype_pair.first + " defined twice.");
        }

        size_t argPosition = m_arg_positions.size();

        m_arg_positions[subtype_pair.first] = argPosition;

        if (subtype_pair.second->is_default_constructible() && !is_default_constructible) {
            is_default_constructible = true;
            default_construction_ix = m_arg_positions[subtype_pair.first];
        }
    }

    size_t size = (all_alternatives_empty ? 1 : sizeof(void*));

    bool anyChanged = (
        size != m_size ||
        m_default_construction_ix != default_construction_ix ||
        m_all_alternatives_empty != all_alternatives_empty ||
        m_is_default_constructible != is_default_constructible
    );

    m_size = size;
    m_default_construction_ix = default_construction_ix;
    m_all_alternatives_empty = all_alternatives_empty;
    m_is_default_constructible = is_default_constructible;

    return anyChanged;
}

bool Alternative::cmp(instance_ptr left, instance_ptr right, int pyComparisonOp, bool suppressExceptions) {
    if (m_all_alternatives_empty) {
        if (*(uint8_t*)left < *(uint8_t*)right) {
            return cmpResultToBoolForPyOrdering(pyComparisonOp, -1);
        }
        if (*(uint8_t*)left > *(uint8_t*)right) {
            return cmpResultToBoolForPyOrdering(pyComparisonOp, 1);
        }
        return cmpResultToBoolForPyOrdering(pyComparisonOp, 0);
    }

    layout& record_l = **(layout**)left;
    layout& record_r = **(layout**)right;

    if ( &record_l == &record_r ) {
        return cmpResultToBoolForPyOrdering(pyComparisonOp, 0);
    }

    if (record_l.which < record_r.which) {
        return cmpResultToBoolForPyOrdering(pyComparisonOp, -1);
    }
    if (record_l.which > record_r.which) {
        return cmpResultToBoolForPyOrdering(pyComparisonOp, 1);
    }

    return m_subtypes[record_l.which].second->cmp(record_l.data, record_r.data, pyComparisonOp, suppressExceptions);
}

//static
bool Alternative::cmpStatic(Alternative* altT, instance_ptr left, instance_ptr right, int64_t pyComparisonOp) {
    if (altT->m_all_alternatives_empty) {
        if (*(uint8_t*)left < *(uint8_t*)right) {
            return cmpResultToBoolForPyOrdering(pyComparisonOp, -1);
        }
        if (*(uint8_t*)left > *(uint8_t*)right) {
            return cmpResultToBoolForPyOrdering(pyComparisonOp, 1);
        }
        return cmpResultToBoolForPyOrdering(pyComparisonOp, 0);
    }

    layout& record_l = **(layout**)left;
    layout& record_r = **(layout**)right;

    if ( &record_l == &record_r ) {
        return cmpResultToBoolForPyOrdering(pyComparisonOp, 0);
    }

    if (record_l.which < record_r.which) {
        return cmpResultToBoolForPyOrdering(pyComparisonOp, -1);
    }
    if (record_l.which > record_r.which) {
        return cmpResultToBoolForPyOrdering(pyComparisonOp, 1);
    }

    return altT->m_subtypes[record_l.which].second->cmp(record_l.data, record_r.data, pyComparisonOp, false);
}

void Alternative::repr(instance_ptr self, ReprAccumulator& stream, bool isStr) {
    PushReprState isNew(stream, self);

    if (!isNew) {
        stream << m_name << "(" << (void*)self << ")";
        return;
    }

    auto it = m_methods.find(isStr ? "__str__" : "__repr__");

    if (it != m_methods.end()) {
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
    }

    stream << m_subtypes[which(self)].first;

    m_subtypes[which(self)].second->repr(eltPtr(self), stream, isStr);
}

typed_python_hash_type Alternative::hash(instance_ptr left) {
    HashAccumulator acc((int)TypeCategory::catAlternative);

    acc.add(which(left));
    acc.add(m_subtypes[which(left)].second->hash(eltPtr(left)));

    return acc.get();
}

instance_ptr Alternative::eltPtr(instance_ptr self) const {
    if (m_all_alternatives_empty) {
        return self;
    }

    layout& record = **(layout**)self;

    return record.data;
}

int64_t Alternative::which(instance_ptr self) const {
    if (m_all_alternatives_empty) {
        return *(uint8_t*)self;
    }

    layout& record = **(layout**)self;

    return record.which;
}

void Alternative::destroy(instance_ptr self) {
    if (m_all_alternatives_empty) {
        return;
    }

    layout& record = **(layout**)self;

    if (record.refcount.fetch_sub(1) == 1) {
        m_subtypes[record.which].second->destroy(record.data);
        tp_free(*(layout**)self);
    }
}

void Alternative::copy_constructor(instance_ptr self, instance_ptr other) {
    if (m_all_alternatives_empty) {
        *(uint8_t*)self = *(uint8_t*)other;
        return;
    }

    (*(layout**)self) = (*(layout**)other);

    if (*(layout**)self) {
        (*(layout**)self)->refcount++;
    }
}

void Alternative::assign(instance_ptr self, instance_ptr other) {
    layout* old = (*(layout**)self);

    (*(layout**)self) = (*(layout**)other);

    if (*(layout**)self) {
        (*(layout**)self)->refcount++;
    }

    destroy((instance_ptr)&old);
}

// static
Alternative* Alternative::Make(
                            std::string name,
                            std::string moduleName,
                            const std::vector<std::pair<std::string, NamedTuple*> >& types,
                            const std::map<std::string, Function*>& methods //methods preclude us from being in the memo
                            ) {
    return new Alternative(name, moduleName, types, methods);
}

Type* Alternative::concreteSubtype(size_t which) {
    if (!m_subtypes_concrete.size()) {
        for (long k = 0; k < m_subtypes.size(); k++) {
            m_subtypes_concrete.push_back(ConcreteAlternative::Make(this, k));
        }
    }

    if (which < 0 || which >= m_subtypes_concrete.size()) {
        throw std::runtime_error("Invalid alternative index.");
    }

    return m_subtypes_concrete[which];
}

Type* Alternative::pickConcreteSubclassConcrete(instance_ptr data) {
    if (!m_subtypes_concrete.size()) {
        for (long k = 0; k < m_subtypes.size(); k++) {
            m_subtypes_concrete.push_back(ConcreteAlternative::Make(this, k));
        }
    }

    uint8_t i = which(data);

    return m_subtypes_concrete[i];
}

void Alternative::constructor(instance_ptr self) {
    assertForwardsResolvedSufficientlyToInstantiate();

    if (!m_default_construction_type) {
        m_default_construction_type = ConcreteAlternative::Make(this, m_default_construction_ix);
    }

    m_default_construction_type->constructor(self);
}
