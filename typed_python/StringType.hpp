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
#include "Unicode.hpp"

class StringType : public Type {
public:
    class layout {
    public:
        std::atomic<int64_t> refcount;
        typed_python_hash_type hash_cache;
        int32_t pointcount;
        int32_t bytes_per_codepoint; //1 implies
        uint8_t data[];

        uint8_t* eltPtr(int64_t i) {
            return data + i * bytes_per_codepoint;
        }
    };

    typedef layout* layout_ptr;

    StringType() : Type(TypeCategory::catString)
    {
        m_name = "str";
        m_is_default_constructible = true;
        m_size = sizeof(void*);

        endOfConstructorInitialization(); // finish initializing the type object.
    }

    static int64_t bytesPerCodepointRequiredForUtf8(const uint8_t* utf8Str, int64_t length);

    //decode a utf8 string. note that 'length' is the number of codepoints, not the number of bytes
    static void decodeUtf8To(uint8_t* target, uint8_t* utf8Str, int64_t bytes_per_codepoint, int64_t length);

    //count the number of codepoints in a utf8 encoded bytestream
    static size_t countUtf8Codepoints(const uint8_t* utfData, size_t bytecount);

    //return an increffed layout containing a copy of lhs at the desired number of codepoints
    static layout* upgradeCodePoints(layout* lhs, int32_t newBytesPerCodepoint);

    //return an increffed concatenated layout of lhs and rhs
    static layout* concatenate(layout* lhs, layout* rhs);

    //return an increffed lowercase conversion layout of l
    static layout* lower(layout *l);

    //return an increffed uppercase conversion layout of l
    static layout* upper(layout *l);
    static layout* capitalize(layout* l);
    static layout* casefold(layout* l);
    static layout* swapcase(layout* l);
    static layout* title(layout* l);

    static layout* strip_whitespace(layout* l, bool fromLeft=true, bool fromRight=true);
    static layout* strip(layout* l, bool whiteSpace=true, layout* values=nullptr, bool fromLeft=true, bool fromRight=true);

    static layout* lstrip(layout *l, bool whiteSpace=true, layout* values=nullptr);

    static layout* rstrip(layout *l, bool whiteSpace=true, layout* values=nullptr);

    //return the lowest index in the string where substring sub is found within l[start, end]
    static int64_t find(layout* l, layout* sub, int64_t start, int64_t end);
    static int64_t rfind(layout* l, layout* sub, int64_t start, int64_t end);
    static int64_t count(layout* l, layout* sub, int64_t start, int64_t end);
    static void split(ListOfType::layout *outList, layout* l, layout* sep, int64_t max);
    static void rsplit(ListOfType::layout *outList, layout* l, layout* sep, int64_t max);
    static void splitlines(ListOfType::layout *outList, layout* l, bool keepends);

    /**
     * It should behave like out = separator.join(toJoin).
     */
    static void join(StringType::layout **out, StringType::layout *separator, ListOfType::layout *toJoin);

    static bool isalpha(layout *l);
    static bool isalnum(layout *l);
    static bool isdecimal(layout *l);
    static bool isdigit(layout *l);
    static bool islower(layout *l);
    static bool isnumeric(layout *l);
    static bool isprintable(layout *l);
    static bool isspace(layout *l);
    static bool istitle(layout *l);
    static bool isupper(layout *l);

    static layout* mult(layout* lhs, int64_t rhs);

    //return an increffed string containing the one codepoint at 'offset'. this function
    //will correctly map negative indices, but performs no other boundschecking.
    static layout* getitem(layout* lhs, int64_t offset);
    static layout* getsubstr(layout* lhs, int64_t start, int64_t stop);

    //return an increffed string containing the data from the utf-encoded string
    static layout* createFromUtf8(const char* utfEncodedString, int64_t len);

    bool isBinaryCompatibleWithConcrete(Type* other) {
        if (other->getTypeCategory() != m_typeCategory) {
            return false;
        }

        return true;
    }

    template<class visitor_type>
    void _visitReferencedTypes(const visitor_type& v) {}

    template<class visitor_type>
    void _visitContainedTypes(const visitor_type& v) {}


    static StringType* Make() { static StringType* res = new StringType(); return res; }

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
            int64_t new_byteCount = srcLayout->pointcount * srcLayout->bytes_per_codepoint;

            destLayout = (layout_ptr)context.slab->allocate(sizeof(layout) + new_byteCount, this);
            destLayout->refcount = 1;
            destLayout->hash_cache = srcLayout->hash_cache;
            destLayout->pointcount = srcLayout->pointcount;
            destLayout->bytes_per_codepoint = srcLayout->bytes_per_codepoint;

            memcpy(destLayout->data, srcLayout->data, new_byteCount);
            return;
        }

        auto it = context.alreadyAllocated.find((instance_ptr)srcLayout);
        if (it == context.alreadyAllocated.end()) {
            int64_t new_byteCount = srcLayout->pointcount * srcLayout->bytes_per_codepoint;

            destLayout = (layout_ptr)context.slab->allocate(sizeof(layout) + new_byteCount, this);
            destLayout->refcount = 0;
            destLayout->hash_cache = srcLayout->hash_cache;
            destLayout->pointcount = srcLayout->pointcount;
            destLayout->bytes_per_codepoint = srcLayout->bytes_per_codepoint;

            memcpy(destLayout->data, srcLayout->data, new_byteCount);

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

        // we don't have to check if we've seen this before if the refcount is 1
        // because there is only one holder.
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

        return bytesRequiredForAllocation(l->pointcount * l->bytes_per_codepoint + sizeof(layout));
    }

    template<class buf_t>
    void serialize(instance_ptr self, buf_t& buffer, size_t fieldNumber) {
        if (bytes_per_codepoint(self) == 1) {
            size_t bytecount = countUtf8BytesRequiredFor((uint8_t*)eltPtr(self, 0), count(self));

            buffer.writeBeginBytes(fieldNumber, bytecount);

            encodeUtf8((uint8_t*)eltPtr(self, 0), count(self), buffer.prepare_bytes(bytecount));
        } else if (bytes_per_codepoint(self) == 2) {
            size_t bytecount = countUtf8BytesRequiredFor((uint16_t*)eltPtr(self, 0), count(self));

            buffer.writeBeginBytes(fieldNumber, bytecount);

            encodeUtf8((uint16_t*)eltPtr(self, 0), count(self), buffer.prepare_bytes(bytecount));
        } else if (bytes_per_codepoint(self) == 4) {
            size_t bytecount = countUtf8BytesRequiredFor((uint32_t*)eltPtr(self, 0), count(self));

            buffer.writeBeginBytes(fieldNumber, bytecount);

            encodeUtf8((uint32_t*)eltPtr(self, 0), count(self), buffer.prepare_bytes(bytecount));
        } else {
            throw std::runtime_error("corrupt bytes-per-codepoint");
        }
    }

    std::string toUtf8String(instance_ptr self) {
        if (!self)
            return std::string();

        std::vector<uint8_t> data;

        if (bytes_per_codepoint(self) == 1) {
            size_t bytecount = countUtf8BytesRequiredFor((uint8_t*)eltPtr(self, 0), count(self));

            data.resize(bytecount);

            encodeUtf8((uint8_t*)eltPtr(self, 0), count(self), &data[0]);
        } else if (bytes_per_codepoint(self) == 2) {
            size_t bytecount = countUtf8BytesRequiredFor((uint16_t*)eltPtr(self, 0), count(self));

            data.resize(bytecount);

            encodeUtf8((uint16_t*)eltPtr(self, 0), count(self), &data[0]);
        } else if (bytes_per_codepoint(self) == 4) {
            size_t bytecount = countUtf8BytesRequiredFor((uint32_t*)eltPtr(self, 0), count(self));

            data.resize(bytecount);

            encodeUtf8((uint32_t*)eltPtr(self, 0), count(self), &data[0]);
        } else {
            throw std::runtime_error("corrupt bytes-per-codepoint");
        }

        return std::string(data.begin(), data.end());
    }

    template<class buf_t>
    void deserialize(instance_ptr self, buf_t& buffer, size_t wireType) {
        assertWireTypesEqual(wireType, WireType::BYTES);

        int32_t ct = buffer.readUnsignedVarint();

        buffer.read_bytes_fun(ct, [&](const uint8_t* bytes) {
            size_t codepointCount = countUtf8Codepoints(bytes, ct);
            *(layout**)self = createFromUtf8((const char*)bytes, codepointCount);
        });
    }

    typed_python_hash_type hash(instance_ptr left) {
        return hash_static(left);
    }

    static typed_python_hash_type hash_static(instance_ptr left) {
        if (!(*(layout**)left)) {
            return 0x12345;
        }

        layout* stringPtr = *(layout**)left;

        if (stringPtr->hash_cache == -1) {
            HashAccumulator acc((int)TypeCategory::catString);

            acc.addBytes(stringPtr->data, stringPtr->bytes_per_codepoint * stringPtr->pointcount);

            stringPtr->hash_cache = acc.get();
            if (stringPtr->hash_cache == -1) {
                stringPtr->hash_cache = -2;
            }
        }

        return (*(layout**)left)->hash_cache;
    }

    static uint32_t getpoint(layout *l, uint64_t i);

    static uint32_t getord(layout *l);

    static layout* singleFromCodepoint(int64_t codePoint);

    bool cmp(instance_ptr left, instance_ptr right, int pyComparisonOp, bool suppressExceptions);

    static char cmpStatic(layout* left, layout* right);
    static char cmpStatic(layout* left, uint8_t* right_data, int right_pointcount, int right_bytes_per_codepoint);

    static bool cmpStaticEq(layout* left, layout* right);

    void constructor(instance_ptr self, int64_t bytes_per_codepoint, int64_t count, const char* data) const;

    void repr(instance_ptr self, ReprAccumulator& stream, bool isStr);

    instance_ptr eltPtr(instance_ptr self, int64_t i) const;

    int64_t bytes_per_codepoint(instance_ptr self) const;

    int64_t count(instance_ptr self) const;

    static int64_t countStatic(StringType::layout* self);

    void constructor(instance_ptr self) {
        *(layout**)self = 0;
    }

    static void destroyStatic(instance_ptr self);

    void destroy(instance_ptr self);

    void copy_constructor(instance_ptr self, instance_ptr other);

    void assign(instance_ptr self, instance_ptr other);

    static bool to_int64(layout* s, int64_t* value);

    static bool to_float64(layout* s, double* value);
};
