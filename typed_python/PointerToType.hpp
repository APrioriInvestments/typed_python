#pragma once

#include "Type.hpp"

class PointerTo : public Type {
protected:
    typedef void* instance;

public:
    PointerTo(Type* t) :
        Type(TypeCategory::catPointerTo),
        m_element_type(t)
    {
        m_size = sizeof(instance);
        m_is_default_constructible = true;

        forwardTypesMayHaveChanged();
    }

    static PointerTo* Make(Type* elt) {
        static std::mutex guard;

        std::lock_guard<std::mutex> lock(guard);

        static std::map<Type*, PointerTo*> m;

        auto it = m.find(elt);
        if (it == m.end()) {
            it = m.insert(std::make_pair(elt, new PointerTo(elt))).first;
        }

        return it->second;
    }

    void forwardTypesMayHaveChanged() {
        m_name = "PointerTo(" + m_element_type->name() + ")";
    }

    template<class visitor_type>
    void _visitContainedTypes(const visitor_type& visitor) {

    }

    template<class visitor_type>
    void _visitReferencedTypes(const visitor_type& visitor) {
        visitor(m_element_type);
    }

    template<class buf_t>
    void serialize(instance_ptr self, buf_t& buffer) {
        throw std::runtime_error("Can't serialize Pointers");
    }

    template<class buf_t>
    void deserialize(instance_ptr self, buf_t& buffer) {
        throw std::runtime_error("Can't deserialize Pointers");
    }

    void repr(instance_ptr self, ReprAccumulator& stream) {
        stream << "(" << m_element_type->name() << "*)" << *(void**)self;
    }

    int32_t hash32(instance_ptr left) {
        Hash32Accumulator acc((int)getTypeCategory());

        acc.addRegister((uint64_t)*(void**)left);

        return acc.get();
    }

    bool cmp(instance_ptr left, instance_ptr right, int pyComparisonOp) {
        if (*(void**)left < *(void**)right) {
            return cmpResultToBoolForPyOrdering(pyComparisonOp, -1);
        }
        if (*(void**)left > *(void**)right) {
            return cmpResultToBoolForPyOrdering(pyComparisonOp, 1);
        }

        return cmpResultToBoolForPyOrdering(pyComparisonOp, 0);
    }

    Type* getEltType() const {
        return m_element_type;
    }

    void constructor(instance_ptr self) {
        *(void**)self = nullptr;
    }

    instance_ptr dereference(instance_ptr self) {
        return (instance_ptr)*(void**)self;
    }

    void destroy(instance_ptr self)  {
    }

    void copy_constructor(instance_ptr self, instance_ptr other) {
        *(void**)self = *(void**)other;
    }

    void assign(instance_ptr self, instance_ptr other) {
        *(void**)self = *(void**)other;
    }

    void offsetBy(instance_ptr out, instance_ptr in, long ix) {
        *(uint8_t**)out = *(uint8_t**)in + (ix * m_element_type->bytecount());
    }

private:
    Type* m_element_type;
};

