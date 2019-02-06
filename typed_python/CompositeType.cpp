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

void CompositeType::_forwardTypesMayHaveChanged() {
    m_is_default_constructible = true;
    m_size = 0;
    m_byte_offsets.clear();

    for (auto t: m_types) {
        m_byte_offsets.push_back(m_size);
        m_size += t->bytecount();
    }

    for (auto t: m_types) {
        if (!t->is_default_constructible()) {
            m_is_default_constructible = false;
        }
    }
}

bool CompositeType::cmp(instance_ptr left, instance_ptr right, int pyComparisonOp) {
    if (pyComparisonOp == Py_NE) {
        return !cmp(left, right, Py_EQ);
    }

    if (pyComparisonOp == Py_EQ) {
        for (long k = 0; k < m_types.size(); k++) {
            if (m_types[k]->cmp(left + m_byte_offsets[k], right + m_byte_offsets[k], Py_NE)) {
                return false;
            }
        }

        return true;
    }

    for (long k = 0; k < m_types.size(); k++) {
        if (m_types[k]->cmp(left + m_byte_offsets[k], right + m_byte_offsets[k], Py_NE)) {
            if (m_types[k]->cmp(left + m_byte_offsets[k], right + m_byte_offsets[k], Py_LT)) {
                return cmpResultToBoolForPyOrdering(pyComparisonOp, -1);
            }
            return cmpResultToBoolForPyOrdering(pyComparisonOp, 1);
        }
    }

    return cmpResultToBoolForPyOrdering(pyComparisonOp, 0);
}

void CompositeType::repr(instance_ptr self, ReprAccumulator& stream) {
    stream << "(";

    for (long k = 0; k < getTypes().size();k++) {
        if (k > 0) {
            stream << ", ";
        }

        if (k < m_names.size()) {
            stream << m_names[k] << "=";
        }

        getTypes()[k]->repr(eltPtr(self,k),stream);
    }
    if (getTypes().size() == 1) {
        stream << ",";
    }

    stream << ")";
}

int32_t CompositeType::hash32(instance_ptr left) {
    Hash32Accumulator acc((int)getTypeCategory());

    for (long k = 0; k < getTypes().size();k++) {
        acc.add(getTypes()[k]->hash32(eltPtr(left,k)));
    }

    acc.add(getTypes().size());

    return acc.get();
}

void CompositeType::constructor(instance_ptr self) {
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

void NamedTuple::_forwardTypesMayHaveChanged() {
    ((CompositeType*)this)->_forwardTypesMayHaveChanged();

    std::string oldName = m_name;

    m_name = "NamedTuple(";
    for (long k = 0; k < m_types.size();k++) {
        if (k) {
            m_name += ", ";
        }
        m_name += m_names[k] + "=" + m_types[k]->name();
    }
    m_name += ")";
}

void Tuple::_forwardTypesMayHaveChanged() {
    ((CompositeType*)this)->_forwardTypesMayHaveChanged();

    m_name = "Tuple(";
    for (long k = 0; k < m_types.size();k++) {
        if (k) {
            m_name += ", ";
        }
        m_name += m_types[k]->name();
    }
    m_name += ")";
}

