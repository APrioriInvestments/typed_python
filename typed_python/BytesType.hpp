#pragma once

#include "Type.hpp"
#include "ReprAccumulator.hpp"

class Bytes : public Type {
public:
    class layout {
    public:
        std::atomic<int64_t> refcount;
        int32_t hash_cache;
        int32_t bytecount;
        uint8_t data[];
    };

    Bytes() : Type(TypeCategory::catBytes)
    {
        m_name = "Bytes";
        m_is_default_constructible = true;
        m_size = sizeof(layout*);
    }

    bool isBinaryCompatibleWithConcrete(Type* other);

    void repr(instance_ptr self, ReprAccumulator& stream);

    template<class buf_t>
    void serialize(instance_ptr self, buf_t& buffer) {
        buffer.write_uint32(count(self));
        buffer.write_bytes(eltPtr(self, 0), count(self));
    }

    template<class buf_t>
    void deserialize(instance_ptr self, buf_t& buffer) {
        int32_t ct = buffer.read_uint32();

        if (ct > buffer.remaining()) {
            throw std::runtime_error("Corrupt data (bytes)");
        }

        constructor(self, ct, nullptr);

        if (ct) {
            buffer.read_bytes(eltPtr(self,0), ct);
        }
    }

    //return an increffed concatenated layout of lhs and rhs
    static layout* concatenate(layout* lhs, layout* rhs);

    //return an increffed bytes object containing a pointer to the requisite bytes
    static layout* createFromPtr(const char* data, int64_t len);

    void _forwardTypesMayHaveChanged() {}

    template<class visitor_type>
    void _visitReferencedTypes(const visitor_type& v) {}

    template<class visitor_type>
    void _visitContainedTypes(const visitor_type& v) {}

    int32_t hash32(instance_ptr left);

    bool cmp(instance_ptr left, instance_ptr right, int pyComparisonOp);

    static Bytes* Make() { static Bytes res; return &res; }

    void constructor(instance_ptr self, int64_t count, const char* data) const;

    instance_ptr eltPtr(instance_ptr self, int64_t i) const;

    int64_t count(instance_ptr self) const;

    void constructor(instance_ptr self) {
        *(layout**)self = 0;
    }

    void destroy(instance_ptr self);

    void copy_constructor(instance_ptr self, instance_ptr other);

    void assign(instance_ptr self, instance_ptr other);
};

