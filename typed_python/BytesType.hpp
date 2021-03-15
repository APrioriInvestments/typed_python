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

#pragma once

#include "Type.hpp"
#include "ReprAccumulator.hpp"

class BytesType : public Type {
public:
    class layout {
    public:
        std::atomic<int64_t> refcount;
        typed_python_hash_type hash_cache;
        int32_t bytecount;
        uint8_t data[];
    };

    typedef layout* layout_ptr;

    BytesType() : Type(TypeCategory::catBytes)
    {
        m_name = "bytes";
        m_is_default_constructible = true;
        m_size = sizeof(layout*);

        endOfConstructorInitialization(); // finish initializing the type object.
    }

    bool isBinaryCompatibleWithConcrete(Type* other);

    void repr(instance_ptr self, ReprAccumulator& stream, bool isStr);

    template<class buf_t>
    void serialize(instance_ptr self, buf_t& buffer, size_t fieldNumber) {
        buffer.writeBeginBytes(fieldNumber, count(self));
        buffer.write_bytes(eltPtr(self, 0), count(self));
    }

    void deepcopyConcrete(
        instance_ptr dest,
        instance_ptr src,
        DeepcopyContext& context
    ) {
        layout_ptr& destLayout = *(layout**)dest;
        layout_ptr& srcLayout = *(layout**)src;

        if (!srcLayout) {
            destLayout = srcLayout;
            return;
        }

        if (srcLayout->refcount == 1) {
            destLayout = (layout_ptr)context.slab->allocate(sizeof(layout) + srcLayout->bytecount, this);
            destLayout->refcount = 1;
            destLayout->hash_cache = srcLayout->hash_cache;
            destLayout->bytecount = srcLayout->bytecount;
            memcpy(destLayout->data, srcLayout->data, srcLayout->bytecount);
            return;
        }

        auto it = context.alreadyAllocated.find((instance_ptr)srcLayout);
        if (it == context.alreadyAllocated.end()) {
            destLayout = (layout_ptr)context.slab->allocate(sizeof(layout) + srcLayout->bytecount, this);
            destLayout->refcount = 0;
            destLayout->hash_cache = srcLayout->hash_cache;
            destLayout->bytecount = srcLayout->bytecount;
            memcpy(destLayout->data, srcLayout->data, srcLayout->bytecount);

            context.alreadyAllocated[(instance_ptr)srcLayout] = (instance_ptr)destLayout;
        } else {
            destLayout = (layout_ptr)it->second;
        }

        destLayout->refcount++;
    }

    size_t deepBytecountConcrete(instance_ptr instance, std::unordered_set<void*>& alreadyVisited, std::set<Slab*>* outSlabs) {
        layout* l = *(layout**)instance;

        if (!l) {
            return 0;
        }

        if (l->refcount != 1) {
            if (alreadyVisited.find((void*)l) != alreadyVisited.end()) {
                return 0;
            }

            alreadyVisited.insert((void*)l);
        }

        if (outSlabs && Slab::slabForAlloc(l)) {
            outSlabs->insert(Slab::slabForAlloc(l));
            return 0;
        }


        return bytesRequiredForAllocation(l->bytecount + sizeof(layout));
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

    template<class visitor_type>
    void _visitReferencedTypes(const visitor_type& v) {}

    template<class visitor_type>
    void _visitContainedTypes(const visitor_type& v) {}

    typed_python_hash_type hash(instance_ptr left);

    bool cmp(instance_ptr left, instance_ptr right, int pyComparisonOp, bool suppressExceptions);

    static char cmpStatic(layout* left, layout* right);

    static BytesType* Make() {
        static BytesType* res = new BytesType();
        return res;
    }

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

    static bool to_int64(layout* s, int64_t* value);

    static bool to_float64(layout* s, double* value);

    // split 'bytesLayout' depositing results into a ListOf<Bytes> in 'outList'
    // if 'sep' is nullptr, split on whitespace
    // max is the maximum number of splits. If its -1, then split as many times as is necessary
    static void split(ListOfType::layout *outList, layout* bytesLayout, layout* sep, int64_t max);
    static void rsplit(ListOfType::layout *outList, layout* bytesLayout, layout* sep, int64_t max);
    static void splitlines(ListOfType::layout *outList, layout* bytesLayout, bool keepends);

    static void join(BytesType::layout **out, BytesType::layout *separator, ListOfType::layout *toJoin);

    static layout* mult(layout* lhs, int64_t rhs);
    static layout* lower(layout* l);
    static layout* upper(layout* l);
    static layout* capitalize(layout* l);
    static layout* swapcase(layout* l);
    static layout* title(layout* l);
    static layout* strip(layout* l, bool whiteSpace, layout* values, bool fromLeft=true, bool fromRight=true);
    static layout* replace(layout* l, layout* old, layout* the_new, int64_t count);
    static layout* translate(layout* l, layout* table, layout* to_delete);
    static layout* maketrans(layout* from, layout* to);
};
