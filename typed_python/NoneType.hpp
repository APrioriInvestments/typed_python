#pragma once

#include "Type.hpp"

class None : public Type {
public:
    None() : Type(TypeCategory::catNone)
    {
        m_name = "NoneType";
        m_size = 0;
        m_is_default_constructible = true;
    }

    bool isBinaryCompatibleWithConcrete(Type* other) {
        if (other->getTypeCategory() != m_typeCategory) {
            return false;
        }

        return true;
    }

    void _forwardTypesMayHaveChanged() {}

    template<class visitor_type>
    void _visitReferencedTypes(const visitor_type& v) {}

    template<class visitor_type>
    void _visitContainedTypes(const visitor_type& v) {}


    bool cmp(instance_ptr left, instance_ptr right, int pyComparisonOp) {
        return cmpResultToBoolForPyOrdering(pyComparisonOp, 0);
    }

    int32_t hash32(instance_ptr left) {
        return (int)getTypeCategory();
    }

    void constructor(instance_ptr self) {}

    void destroy(instance_ptr self) {}

    void copy_constructor(instance_ptr self, instance_ptr other) {}

    void assign(instance_ptr self, instance_ptr other) {}

    static None* Make() { static None res; return &res; }

    template<class buf_t>
    void deserialize(instance_ptr self, buf_t& buffer) {
    }

    template<class buf_t>
    void serialize(instance_ptr self, buf_t& buffer) {
    }

    void repr(instance_ptr self, ReprAccumulator& stream) {
        stream << "None";
    }
};

