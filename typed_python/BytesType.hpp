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

#pragma once

#include "Type.hpp"
#include "ReprAccumulator.hpp"

class BytesType : public Type {
public:
    class layout {
    public:
        std::atomic<int64_t> refcount;
        int32_t hash_cache;
        int32_t bytecount;
        uint8_t data[];
    };

    BytesType() : Type(TypeCategory::catBytes)
    {
        m_name = "Bytes";
        m_is_default_constructible = true;
        m_size = sizeof(layout*);
    }

    bool isBinaryCompatibleWithConcrete(Type* other);

    void repr(instance_ptr self, ReprAccumulator& stream);

    template<class buf_t>
    void serialize(instance_ptr self, buf_t& buffer, size_t fieldNumber) {
        buffer.writeBeginBytes(fieldNumber, count(self));
        buffer.write_bytes(eltPtr(self, 0), count(self));
    }

    template<class buf_t>
    void deserialize(instance_ptr self, buf_t& buffer, size_t wireType) {
        if (wireType != WireType::BYTES) {
            throw std::runtime_error("Corrupt data (expected BYTES wire type)");
        }

        size_t ct = buffer.readUnsignedVarint();

        if (!buffer.canConsume(ct)) {
            throw std::runtime_error("Corrupt data (not enough data in the stream)");
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
    static char cmpStatic(layout* left, layout* right);

    static BytesType* Make() { static BytesType res; return &res; }

    void constructor(instance_ptr self, int64_t count, const char* data) const;

    instance_ptr eltPtr(instance_ptr self, int64_t i) const;

    int64_t count(instance_ptr self) const;

    void constructor(instance_ptr self) {
        *(layout**)self = 0;
    }

    void destroy(instance_ptr self);

    static void destroyStatic(instance_ptr self);

    void copy_constructor(instance_ptr self, instance_ptr other);

    void assign(instance_ptr self, instance_ptr other);
};

