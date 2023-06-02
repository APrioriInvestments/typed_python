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

bool CompositeType::isBinaryCompatibleWithConcrete(Type* other) {
    if (other->getTypeCategory() != m_typeCategory) {
        return false;
    }

    CompositeType* otherO = (CompositeType*)other;

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

bool CompositeType::cmp(instance_ptr left, instance_ptr right, int pyComparisonOp, bool suppressExceptions) {
    if (pyComparisonOp == Py_NE) {
        return !cmp(left, right, Py_EQ, suppressExceptions);
    }

    if (pyComparisonOp == Py_EQ) {
        for (long k = 0; k < m_types.size(); k++) {
            if (m_types[k]->cmp(left + m_byte_offsets[k], right + m_byte_offsets[k], Py_NE, suppressExceptions)) {
                return false;
            }
        }

        return true;
    }

    for (long k = 0; k < m_types.size(); k++) {
        if (m_types[k]->cmp(left + m_byte_offsets[k], right + m_byte_offsets[k], Py_NE, suppressExceptions)) {
            if (m_types[k]->cmp(left + m_byte_offsets[k], right + m_byte_offsets[k], Py_LT, suppressExceptions)) {
                return cmpResultToBoolForPyOrdering(pyComparisonOp, -1);
            }
            return cmpResultToBoolForPyOrdering(pyComparisonOp, 1);
        }
    }

    return cmpResultToBoolForPyOrdering(pyComparisonOp, 0);
}

void CompositeType::repr(instance_ptr self, ReprAccumulator& stream, bool isStr) {
    stream << "(";

    for (long k = 0; k < getTypes().size();k++) {
        if (k > 0) {
            stream << ", ";
        }

        if (k < m_names.size()) {
            stream << m_names[k] << "=";
        }

        getTypes()[k]->repr(eltPtr(self,k),stream, false);
    }
    if (getTypes().size() == 1) {
        stream << ",";
    }

    stream << ")";
}

typed_python_hash_type CompositeType::hash(instance_ptr left) {
    HashAccumulator acc(0);

    for (long k = 0; k < getTypes().size();k++) {
        acc.add(getTypes()[k]->hash(eltPtr(left,k)));
    }

    return acc.get();
}

void CompositeType::constructor(instance_ptr self) {
    assertForwardsResolvedSufficientlyToInstantiate();

    if (!m_is_default_constructible) {
        throw std::runtime_error(m_name + " is not default-constructible");
    }

    for (size_t k = 0; k < m_types.size(); k++) {
        m_types[k]->constructor(self+m_byte_offsets[k]);
    }
}

void CompositeType::destroy(instance_ptr self) {
    for (long k = (long)m_types.size() - 1; k >= 0; k--) {
        m_types[k]->destroy(self+m_byte_offsets[k]);
    }
}

void CompositeType::copy_constructor(instance_ptr self, instance_ptr other) {
    for (long k = (long)m_types.size() - 1; k >= 0; k--) {
        m_types[k]->copy_constructor(self + m_byte_offsets[k], other+m_byte_offsets[k]);
    }
}

void CompositeType::assign(instance_ptr self, instance_ptr other) {
    for (long k = (long)m_types.size() - 1; k >= 0; k--) {
        m_types[k]->assign(self + m_byte_offsets[k], other+m_byte_offsets[k]);
    }
}

std::string NamedTuple::computeRecursiveNameConcrete(TypeStack& typeStack) {
    std::string name = "NamedTuple(";

    if (m_types.size() != m_names.size()) {
        throw std::logic_error("Names mismatched with types!");
    }

    for (long k = 0; k < m_types.size();k++) {
        if (k) {
            name += ", ";
        }
        name += m_names[k] + "=" + m_types[k]->computeRecursiveName(typeStack);
    }
    name += ")";

    return name;
}

std::string Tuple::computeRecursiveNameConcrete(TypeStack& typeStack) {
    std::string name = "Tuple(";

    for (long k = 0; k < m_types.size();k++) {
        if (k) {
            name += ", ";
        }
        name += m_types[k]->computeRecursiveName(typeStack);
    }
    name += ")";

    return name;
}
