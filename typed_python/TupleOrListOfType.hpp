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

public:
    TupleOrListOfType(Type* type, bool isTuple) :
            Type(isTuple ? TypeCategory::catTupleOf : TypeCategory::catListOf),
            m_element_type(type),
            m_is_tuple(isTuple)
    {
        m_size = sizeof(void*);
        m_is_default_constructible = true;

        endOfConstructorInitialization(); // finish initializing the type object.
    }

    bool isBinaryCompatibleWithConcrete(Type* other);

    template<class visitor_type>
    void _visitContainedTypes(const visitor_type& visitor) {

    }

    template<class visitor_type>
    void _visitReferencedTypes(const visitor_type& visitor) {
        visitor(m_element_type);
    }

    ShaHash _computeIdentityHash(MutuallyRecursiveTypeGroup* groupHead = nullptr) {
        return ShaHash(1, m_typeCategory) + m_element_type->identityHash(groupHead);
    }

    bool _updateAfterForwardTypesChanged();

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
                for (long k2 = k-1; k2 >= 0; k2--) {
                    m_element_type->destroy(eltPtr(self,k2));
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
                for (long k2 = (long)self->count-1; k2 >= 0; k2--) {
                    m_element_type->destroy(eltPtr(self,k2));
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
public:
    ListOfType(Type* type) : TupleOrListOfType(type, false)
    {
        m_doc = ListOf_doc;
    }

    static ListOfType* Make(Type* elt, ListOfType* knownType=nullptr);

    void _updateTypeMemosAfterForwardResolution() {
        ListOfType::Make(m_element_type, this);
    }

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

    template<class buf_t>
    void serialize(instance_ptr self, buf_t& buffer, size_t fieldNumber) {
        size_t ct = count(self);

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
        buffer.writeUnsignedVarintObject(0, ct);

        m_element_type->check([&](auto& concrete_type) {
            for (long k = 0; k < ct; k++) {
                concrete_type.serialize(this->eltPtr(self,k), buffer, 0);
            }
        });

        buffer.writeEndCompound();
    }

    template<class buf_t>
    void deserialize(instance_ptr self, buf_t& buffer, size_t wireType) {
        assertNonemptyCompoundWireType(wireType);

        size_t id = buffer.readUnsignedVarintObject();

        void* ptr = buffer.lookupCachedPointer(id);

        if (ptr) {
            ((layout**)self)[0] = (layout*)ptr;
            ((layout**)self)[0]->refcount++;
            buffer.finishCompoundMessage(wireType);
            return;
        }

        size_t ct = buffer.readUnsignedVarintObject();

        if (ct == 0) {
            constructor(self);
            (*(layout**)self)->refcount++;
            buffer.addCachedPointer(id, *((layout**)self), this);
        } else {
            constructor(self, ct, [&](instance_ptr tgt, int k) {
                if (k == 0) {
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
        }

        buffer.finishCompoundMessage(wireType);
    }
};

PyDoc_STRVAR(TupleOf_doc,
    "TupleOf(T)() -> empty typed tuple\n"
    "TupleOf(T)(t) -> typed tuple containing elements of type T, initialized from tuple t\n"
    "TupleOf(T)(x) -> typed tuple containing elements of type T, initialized from iterable x\n"
    );

class TupleOfType : public TupleOrListOfType {
public:
    TupleOfType(Type* type) : TupleOrListOfType(type, true)
    {
        m_doc = TupleOf_doc;
    }

    void _updateTypeMemosAfterForwardResolution() {
        TupleOfType::Make(m_element_type, this);
    }

    template<class buf_t>
    void serialize(instance_ptr self, buf_t& buffer, size_t fieldNumber) {
        size_t ct = count(self);

        if (ct == 0) {
            buffer.writeEmpty(fieldNumber);
            return;
        }

        buffer.writeBeginCompound(fieldNumber);

        buffer.writeUnsignedVarintObject(0, ct);

        m_element_type->check([&](auto& concrete_type) {
            for (long k = 0; k < ct; k++) {
                concrete_type.serialize(this->eltPtr(self,k), buffer, 0);
            }
        });

        buffer.writeEndCompound();
    }


    template<class buf_t>
    void deserialize(instance_ptr self, buf_t& buffer, size_t wireType) {
        if (wireType == WireType::EMPTY) {
            *(layout**)self = nullptr;
            return;
        }

        assertNonemptyCompoundWireType(wireType);

        size_t ct = buffer.readUnsignedVarintObject();

        constructor(self, ct, [&](instance_ptr tgt, int k) {
            auto fieldAndWire = buffer.readFieldNumberAndWireType();
            if (fieldAndWire.first) {
                throw std::runtime_error("Corrupt data (count)");
            }
            if (fieldAndWire.second == WireType::END_COMPOUND) {
                throw std::runtime_error("Corrupt data (count)");
            }

            m_element_type->deserialize(tgt, buffer, fieldAndWire.second);
        });

        buffer.finishCompoundMessage(wireType);
    }

    // get a memoized TupleOfType. If 'knownType', then install this type
    // if not already known.
    static TupleOfType* Make(Type* elt, TupleOfType* knownType = nullptr);
};
