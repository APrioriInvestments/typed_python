/******************************************************************************
   Copyright 2017-2020 typed_python Authors

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
#include "Format.hpp"

class TupleOrListOfType : public Type {
public:
    class layout {
    public:
        std::atomic<int64_t> refcount;
        typed_python_hash_type hash_cache;
        int32_t count;
        int32_t reserved;
        uint8_t* data;
    };

    typedef layout* layout_ptr;

    // this is the non-forward clone pathway
    TupleOrListOfType(bool isTuple) :
        Type(isTuple ? TypeCategory::catTupleOf : TypeCategory::catListOf),
        m_element_type(nullptr),
        m_is_tuple(isTuple)
    {
        m_is_default_constructible = true;
    }

    TupleOrListOfType(Type* type, bool isTuple) :
            Type(isTuple ? TypeCategory::catTupleOf : TypeCategory::catListOf),
            m_element_type(type),
            m_is_tuple(isTuple)
    {
        m_is_forward_defined = true;
        m_is_default_constructible = true;

        recomputeName();
    }

    void initializeDuringDeserialization(Type* elementType) {
        m_element_type = elementType;
    }

    template<class visitor_type>
    void _visitContainedTypes(const visitor_type& visitor) {

    }

    template<class visitor_type>
    void _visitReferencedTypes(const visitor_type& visitor) {
        visitor(m_element_type);
    }

    template<class visitor_type>
    void _visitCompilerVisibleInternals(const visitor_type& v) {
        v.visitHash(ShaHash(1, m_typeCategory));
        v.visitTopo(m_element_type);
    }

    //serialize, but don't write a count
    template<class buf_t>
    void serializeStream(instance_ptr self, buf_t& buffer) {
        int32_t ct = count(self);
        m_element_type->check([&](auto& concrete_type) {
            for (long k = 0; k < ct;k++) {
                concrete_type.serialize(this->eltPtr(self,k), buffer, 0);
            }
        });
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

        auto it = context.alreadyAllocated.find((instance_ptr)srcLayout);
        if (it == context.alreadyAllocated.end()) {
            destLayout = (layout_ptr)context.slab->allocate(sizeof(layout), this);
            destLayout->hash_cache = srcLayout->hash_cache;
            destLayout->refcount = 0;

            size_t reserveCount = srcLayout->count;

            if (reserveCount == 0 && !m_is_tuple) {
                reserveCount = 1;
            }

            destLayout->reserved = reserveCount;
            destLayout->count = srcLayout->count;

            if (reserveCount) {
                destLayout->data = (instance_ptr)context.slab->allocate(getEltType()->bytecount() * reserveCount, nullptr);
            } else {
                destLayout->data = nullptr;
            }

            if (destLayout->count) {
                if (getEltType()->isPOD()) {
                    memcpy(destLayout->data, srcLayout->data, getEltType()->bytecount() * srcLayout->count);
                } else {
                    for (long k = 0; k < srcLayout->count; k++) {
                        m_element_type->deepcopy(
                            destLayout->data + k * m_element_type->bytecount(),
                            srcLayout->data + k * m_element_type->bytecount(),
                            context
                        );
                    }
                }
            }

            context.alreadyAllocated[(instance_ptr)srcLayout] = (instance_ptr)destLayout;
        } else {
            destLayout = (layout_ptr)it->second;
        }

        destLayout->refcount++;
    }

    size_t deepBytecountConcrete(instance_ptr instance, std::unordered_set<void*>& alreadyVisited, std::set<Slab*>* outSlabs) {
        layout_ptr& self_layout = *(layout_ptr*)instance;

        if (!self_layout) {
            return 0;
        }

        if (alreadyVisited.find((void*)self_layout) != alreadyVisited.end()) {
            return 0;
        }

        alreadyVisited.insert((void*)self_layout);

        if (outSlabs && Slab::slabForAlloc(self_layout)) {
            outSlabs->insert(Slab::slabForAlloc(self_layout));
            return 0;
        }

        size_t reserveCount = self_layout->count;

        // always allocate something for a list
        if (reserveCount == 0 && !m_is_tuple) {
            reserveCount = 1;
        }

        size_t res = bytesRequiredForAllocation(sizeof(layout));

        if (reserveCount) {
            res += bytesRequiredForAllocation(reserveCount * getEltType()->bytecount());
        }

        if (!getEltType()->isPOD()) {
            for (long k = 0; k < self_layout->count; k++) {
                res += m_element_type->deepBytecount(eltPtr(instance, k), alreadyVisited, outSlabs);
            }
        }

        return res;
    }

    void repr(instance_ptr self, ReprAccumulator& stream, bool isStr);

    typed_python_hash_type hash(instance_ptr left);

    bool cmp(instance_ptr left, instance_ptr right, int pyComparisonOp, bool suppressExceptions);

    Type* getEltType() const {
        return m_element_type;
    }

    instance_ptr eltPtr(layout_ptr self, int64_t i) const {
        return eltPtr((instance_ptr)&self, i);
    }

    instance_ptr eltPtr(instance_ptr self, int64_t i) const {
        if (!(*(layout**)self)) {
            return self;
        }

        return (*(layout**)self)->data + i * m_element_type->bytecount();
    }

    int64_t count(instance_ptr self) const;

    int64_t refcount(instance_ptr self) const;

    //construct a new list at 'selfPtr' with 'count' items, each initialized by calling
    //'allocator(target_object, k)', where 'target_object' is a pointer to the memory location
    //to be filled and 'k' is the index in the list.
    template<class sub_constructor>
    void constructor(instance_ptr selfPtr, int64_t count, const sub_constructor& allocator) {
        layout_ptr& self = *(layout_ptr*)selfPtr;

        if (count == 0 && m_is_tuple) {
            self = nullptr;
            return;
        }

        self = (layout*)tp_malloc(sizeof(layout));

        self->count = count;
        self->refcount = 1;
        self->reserved = std::max<int32_t>(1, count);
        self->hash_cache = -1;
        self->data = (uint8_t*)tp_malloc(getEltType()->bytecount() * self->reserved);

        for (int64_t k = 0; k < count; k++) {
            try {
                allocator(eltPtr(self, k), k);
            } catch(...) {
                if (!m_element_type->isPOD()) {
                    for (long k2 = k-1; k2 >= 0; k2--) {
                        m_element_type->destroy(eltPtr(self,k2));
                    }
                }
                tp_free(self->data);
                tp_free(self);
                throw;
            }
        }
    }
    //construct a new list at 'selfPtr'. We call 'allocator(target_object, k)' repeatedly.
    //we stop when it returns 'false'
    template<class sub_constructor>
    void constructorUnbounded(instance_ptr selfPtr, const sub_constructor& allocator) {
        layout_ptr& self = *(layout_ptr*)selfPtr;

        self = (layout*)tp_malloc(sizeof(layout));

        self->count = 0;
        self->refcount = 1;
        self->reserved = 1;
        self->hash_cache = -1;
        self->data = (uint8_t*)tp_malloc(getEltType()->bytecount() * self->reserved);

        while(true) {
            try {
                if (!allocator(eltPtr(self, self->count), self->count)) {
                    if (m_is_tuple && self->count == 0) {
                        //tuples need to be the nullptr
                        tp_free(self->data);
                        tp_free(self);
                        self = nullptr;
                    }
                    return;
                }

                self->count++;
                if (self->count >= self->reserved) {
                    reserve(selfPtr, self->reserved * 1.25 + 1);
                }
            } catch(...) {
                if (!m_element_type->isPOD()) {
                    for (long k2 = (long)self->count-1; k2 >= 0; k2--) {
                        m_element_type->destroy(eltPtr(self,k2));
                    }
                }
                tp_free(self->data);
                tp_free(self);
                throw;
            }
        }
    }

    void constructor(instance_ptr self);

    void destroy(instance_ptr self);

    void copy_constructor(instance_ptr self, instance_ptr other);

    void assign(instance_ptr self, instance_ptr other);

    void reserve(instance_ptr self, size_t count);

    void setSizeUnsafe(instance_ptr self, size_t count);

    void reverse(instance_ptr self);

    enum class int_block_type {
        sequence=1,
        int8=2,
        int16=3,
        int32=4,
        int64=5,
    };

    // try to serialize a block of integers that fit into the limits of "T"
    template<class buf_t, class T>
    bool trySerializeIntListBlock(int64_t* &ptr, size_t &count, buf_t& buffer, int_block_type blockType, T* nullPtr) {
        if (ptr[0] >= std::numeric_limits<T>::min()
                    && ptr[0] <= std::numeric_limits<T>::max()) {
            long topPt = 1;
            while (topPt < count
                    && topPt < std::numeric_limits<uint8_t>::max()
                    && ptr[topPt] >= std::numeric_limits<T>::min()
                    && ptr[topPt] <= std::numeric_limits<T>::max()) {
                topPt++;
            }

            buffer.writeBeginBytes(
                (uint8_t)blockType,
                topPt * sizeof(T)
            );
            buffer.initialize_bytes(topPt * sizeof(T), [&](uint8_t* output) {
                for (long k = 0; k < topPt; k++) {
                    ((T*)output)[k] = ptr[k];
                }
            });

            count -= topPt;
            ptr += topPt;

            return true;
        }

        return false;
    }

    template<class buf_t>
    void serializeIntList(int64_t* ptr, size_t count, buf_t& buffer) {
        // serialize integers in blocks. each block is either
        //  - a bunch of integers encoded the normal way but with a
        //    possibly reduced bit-size (often we use 8 byte integers
        //    to compress a lot of 2 or 4 byte integers
        //  - a block of integers with a constant difference between them

        while (count > 0) {
            if (    trySerializeIntListBlock(ptr, count, buffer, int_block_type::int8, (int8_t*)nullptr)
                ||  trySerializeIntListBlock(ptr, count, buffer, int_block_type::int16, (int16_t*)nullptr)
                ||  trySerializeIntListBlock(ptr, count, buffer, int_block_type::int32, (int32_t*)nullptr)
                ||  trySerializeIntListBlock(ptr, count, buffer, int_block_type::int64, (int64_t*)nullptr)
            ) {
                // we serialized somehow...
            } else {
                throw std::runtime_error("unreachable code during serializeIntList");
            }
        }
    }

    template<class buf_t, class T>
    size_t deserializeIntListBlock(int64_t* ptr, size_t bytecount, buf_t& buffer, T* nullPtr) {
        size_t count = bytecount / sizeof(T);

        buffer.read_bytes_fun(bytecount,  [&](uint8_t* dataPtr) {
            for (long k = 0; k < count; k++) {
                ptr[k] = ((T*)dataPtr)[k];
            }
        });

        return count;
    }

    template<class buf_t>
    void deserializeIntList(int64_t* ptr, size_t count, buf_t& buffer) {
        while (count) {
            auto fieldAndWireType = buffer.readFieldNumberAndWireType();
            int_block_type kind = (int_block_type)fieldAndWireType.first;

            size_t bytecount = buffer.readUnsignedVarint();
            size_t valuesRead;

            if (kind == int_block_type::int8) {
                valuesRead = deserializeIntListBlock(ptr, bytecount, buffer, (int8_t*)nullptr);
            }
            else if (kind == int_block_type::int16) {
                valuesRead = deserializeIntListBlock(ptr, bytecount, buffer, (int16_t*)nullptr);
            }
            else if (kind == int_block_type::int32) {
                valuesRead = deserializeIntListBlock(ptr, bytecount, buffer, (int32_t*)nullptr);
            }
            else if (kind == int_block_type::int64) {
                valuesRead = deserializeIntListBlock(ptr, bytecount, buffer, (int64_t*)nullptr);
            } else {
                throw std::runtime_error("unreachable code during deserializeIntList");
            }

            count -= valuesRead;
            ptr += valuesRead;
        }
    }

    template<class buf_t>
    void serialize(instance_ptr self, buf_t& buffer, size_t fieldNumber) {
        size_t ct = count(self);

        if (ct == 0 && isTupleOf()) {
            buffer.writeEmpty(fieldNumber);
            return;
        }

        // list-of needs a memo. TupleOf doesn't.
        if (isListOf()) {
            uint32_t id;
            bool isNew;
            std::tie(id, isNew) = buffer.cachePointer(*(void**)self, this);

            if (!isNew) {
                buffer.writeBeginSingle(fieldNumber);
                buffer.writeUnsignedVarintObject(0, id);
                return;
            }

            buffer.writeBeginCompound(fieldNumber);
            buffer.writeUnsignedVarintObject(0, id);
        } else {
            buffer.writeBeginCompound(fieldNumber);
        }

        if (ct && m_element_type->isPOD() && buffer.getContext().serializePodListsInline()) {
            if (m_element_type->bytecount() == 0) {
                // this is for the special case where we are writing
                // zero bytes, in which case the receiving side cannot infer the
                // number of elements from the bytecount.
                buffer.writeUnsignedVarintObject(3, ct);
            }
            else if (m_element_type->getTypeCategory() == TypeCategory::catInt64) {
                buffer.writeUnsignedVarintObject(1, ct);

                serializeIntList(
                    (int64_t*)this->eltPtr(self, 0),
                    ct,
                    buffer
                );
            } else {
                buffer.writeBeginBytes(2, m_element_type->bytecount() * ct);

                buffer.write_bytes(
                    this->eltPtr(self, 0),
                    m_element_type->bytecount() * ct
                );
            }
        } else {
            buffer.writeUnsignedVarintObject(0, ct);

            m_element_type->serializeMulti(
                this->eltPtr(self, 0),
                ct,
                m_element_type->bytecount(),
                buffer,
                0
            );
        }

        buffer.writeEndCompound();
    }

    template<class buf_t>
    void deserialize(instance_ptr self, buf_t& buffer, size_t wireType) {
        if (wireType == WireType::EMPTY && isTupleOf()) {
            *(layout**)self = nullptr;
            return;
        }

        assertNonemptyCompoundWireType(wireType);

        // list-of needs a memo. TupleOf doesn't.
        size_t id = 0;
        if (isListOf()) {
            id = buffer.readUnsignedVarintObject();

            void* ptr = buffer.lookupCachedPointer(id);

            if (ptr) {
                ((layout**)self)[0] = (layout*)ptr;
                ((layout**)self)[0]->refcount++;
                try {
                    buffer.finishCompoundMessage(wireType);
                } catch(...) {
                    throw std::runtime_error("1. Failed finishing: " + name());
                }

                return;
            }
        }

        auto fieldnumAndWireType = buffer.readFieldNumberAndWireType();
        size_t fieldnum = fieldnumAndWireType.first;
        size_t ct = buffer.readUnsignedVarint();

        if (ct == 0) {
            constructor(self);

            if (fieldnum != 0) {
                throw std::runtime_error("Corrupt field num - empty list/tuple count should be 0");
            }

            if (isListOf()) {
                (*(layout**)self)->refcount++;
                buffer.addCachedPointer(id, *((layout**)self), this);
            }
        } else {
            if (fieldnum == 0) {
                constructor(self, ct, [&](instance_ptr tgt, int k) {
                    if (k == 0 && isListOf()) {
                        buffer.addCachedPointer(id, *((layout**)self), this);
                        (*(layout**)self)->refcount++;
                    }

                    auto fieldAndWire = buffer.readFieldNumberAndWireType();
                    if (fieldAndWire.first) {
                        throw std::runtime_error("Corrupt data (count)");
                    }
                    if (fieldAndWire.second == WireType::END_COMPOUND) {
                        throw std::runtime_error("Corrupt data (count)");
                    }

                    m_element_type->deserialize(tgt, buffer, fieldAndWire.second);
                });
            } else
            if (fieldnum == 1) {
                if (m_element_type->getTypeCategory() != TypeCategory::catInt64) {
                    throw std::runtime_error(
                        "Compressed intArray data data makes no sense for " + m_element_type->name()
                    );
                }
                constructor(self, ct, [&](instance_ptr tgt, int k) {});

                if (isListOf()) {
                    (*(layout**)self)->refcount++;
                    buffer.addCachedPointer(id, *((layout**)self), this);
                }

                deserializeIntList(
                    (int64_t*)this->eltPtr(self, 0),
                    ct,
                    buffer
                );
            } else
            if (fieldnum == 2) {
                if (!m_element_type->isPOD()) {
                    throw std::runtime_error(
                        "Compressed POD data makes no sense for " + m_element_type->name()
                    );
                }

                size_t eltCount = ct / m_element_type->bytecount();
                if (eltCount * m_element_type->bytecount() != ct) {
                    throw std::runtime_error("Invalid inline POD data - not a proper multiple");
                }

                constructor(self, eltCount, [&](instance_ptr tgt, int k) {});

                if (isListOf()) {
                    (*(layout**)self)->refcount++;
                    buffer.addCachedPointer(id, *((layout**)self), this);
                }

                buffer.read_bytes(this->eltPtr(self, 0), ct);
            } else
            if (fieldnum == 3) {
                if (!m_element_type->isPOD() || m_element_type->bytecount()) {
                    throw std::runtime_error(
                        "Zero-body POD data makes no sense for " + m_element_type->name()
                    );
                }

                constructor(self, ct, [&](instance_ptr tgt, int k) {});

                if (isListOf()) {
                    (*(layout**)self)->refcount++;
                    buffer.addCachedPointer(id, *((layout**)self), this);
                }
            } else {
                throw std::runtime_error("Corrupt fieldnum for tuple/listof body");
            }
        }

        try {
            buffer.finishCompoundMessage(wireType);
        }
        catch(...) {
            throw std::runtime_error("2. Failed finishing: " + name() + ": " + format(ct));
        }

    }

    Type* cloneForForwardResolutionConcrete();

    void initializeFromConcrete(Type* forwardDefinitionOfSelf);

    void postInitializeConcrete() {
        m_size = sizeof(layout*);
    };

    std::string computeRecursiveNameConcrete(TypeStack& typeStack);

    void updateInternalTypePointersConcrete(
        const std::map<Type*, Type*>& groupMap
    );

protected:
    Type* m_element_type;
    bool m_is_tuple;
};

PyDoc_STRVAR(ListOf_doc,
    "ListOf(T)() -> empty typed list\n"
    "ListOf(T)(lst) -> typed list containing elements of type T, initialized from list lst\n"
    "ListOf(T)(x) -> typed list containing elements of type T, initialized from iterable x\n"
    );

class ListOfType : public TupleOrListOfType {
    friend class TupleOrListOfType;

public:
    ListOfType() : TupleOrListOfType(false)
    {
    }

    ListOfType(Type* type) : TupleOrListOfType(type, false)
    {
    }


    const char* docConcrete() {
        return ListOf_doc;
    }

    static ListOfType* Make(Type* elt);

    void setSizeUnsafe(instance_ptr self, size_t count);

    void append(instance_ptr self, instance_ptr other);

    size_t reserved(instance_ptr self);

    void remove(instance_ptr self, size_t count);

    void resize(instance_ptr self, size_t count);

    void resize(instance_ptr self, size_t count, instance_ptr value);

    void copyListObject(instance_ptr target, instance_ptr src);

    void ensureSpaceFor(instance_ptr self, size_t count);

    template<class initializer>
    void extend(instance_ptr self, size_t count, const initializer& initFun) {
        layout_ptr& self_layout = *(layout_ptr*)self;
        ensureSpaceFor(self, count);

        size_t bytesPer = m_element_type->bytecount();
        instance_ptr base = this->eltPtr(self, this->count(self));

        size_t i = 0;

        try {
            for (; i < count; i++) {
                initFun(base + bytesPer * i, i);
            }

            self_layout->count += i;
        }
        catch(...) {
            self_layout->count += i;
            throw;
        }
    }
};

PyDoc_STRVAR(TupleOf_doc,
    "TupleOf(T)() -> empty typed tuple\n"
    "TupleOf(T)(t) -> typed tuple containing elements of type T, initialized from tuple t\n"
    "TupleOf(T)(x) -> typed tuple containing elements of type T, initialized from iterable x\n"
    );

class TupleOfType : public TupleOrListOfType {
    friend class TupleOrListOfType;

public:
    TupleOfType() : TupleOrListOfType(true)
    {
    }

    // forward form
    TupleOfType(Type* type) : TupleOrListOfType(type, true)
    {
    }

    const char* docConcrete() {
        return TupleOf_doc;
    }

    static TupleOfType* Make(Type* elt);
};
