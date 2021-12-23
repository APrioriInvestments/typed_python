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

class Instance;

class VTable;

typedef VTable* vtable_ptr;


#define BOTTOM_48_BITS 0xFFFFFFFFFFFF

PyDoc_STRVAR(Class_doc,
    "Class: subclass Class to produce typed python classes.\n"
    "\n"
    "Methods become TypedFunction instances, and support overloading.\n"
    "Define data members with Member.\n"
    );

class Class : public Type {
public:
    class layout {
    public:
        std::atomic<int64_t> refcount;
        vtable_ptr vtable;
        unsigned char data[];
    };

    typedef layout* layout_ptr;

    Class(std::string name, HeldClass* inClass) :
            Type(catClass),
            m_heldClass(inClass)
    {
        m_size = sizeof(layout*);
        m_is_default_constructible = inClass->is_default_constructible();
        m_name = name;
        m_doc = Class_doc;
        m_is_simple = false;

        endOfConstructorInitialization(); // finish initializing the type object.

        inClass->setClassType(this);
    }

    // convert an instance of the class to an actual layout pointer. Because
    // we encode the offset of the dispatch table we're supposed to use for
    // this subclass in the top 16 bits of the pointer, we have to be careful
    // to always use this function to access the layout.
    static layout* instanceToLayout(instance_ptr data) {
        size_t layoutPtrAndDispatchTable = *(size_t*)data;
        return (layout*)(layoutPtrAndDispatchTable & BOTTOM_48_BITS);
    }

    static uint16_t instanceToDispatchTableIndex(instance_ptr data) {
        size_t layoutPtrAndDispatchTable = *(size_t*)data;
        return layoutPtrAndDispatchTable >> 48;
    }

    static layout* initializeInstance(instance_ptr toInit, layout* layoutPtr, uint16_t dispatchIndex) {
        if (dispatchIndex >= 65535) {
            std::cerr << "somehow we got a corrupt class object" << std::endl;

            asm("int3");
        }

        if (((size_t)layoutPtr) >> 48) {
            throw std::runtime_error("Invalid layout pointer encountered.");
        }

        *(size_t*)toInit = ((size_t)layoutPtr) + ((uint64_t)dispatchIndex << 48);

        return layoutPtr;
    }

    // get the actual realized class contained in the instance of 'data'
    static Class* actualTypeForLayout(instance_ptr data) {
        return vtableFor(data)->mType->getClassType();
    }

    static vtable_ptr& vtableFor(instance_ptr self) {
        return instanceToLayout(self)->vtable;
    }

    bool isBinaryCompatibleWithConcrete(Type* other);

    template<class visitor_type>
    void _visitContainedTypes(const visitor_type& visitor) {
    }

    template<class visitor_type>
    void _visitReferencedTypes(const visitor_type& visitor) {
        Type* t = m_heldClass;
        visitor(t);
        assert(t == m_heldClass);
    }

    Type* pickConcreteSubclassConcrete(instance_ptr self) {
        return actualTypeForLayout(self);
    }

    bool _updateAfterForwardTypesChanged();

    static Class* Make(
            std::string inName,
            const std::vector<Class*>& bases,
            bool isFinal,
            const std::vector<MemberDefinition>& members,
            const std::map<std::string, Function*>& memberFunctions,
            const std::map<std::string, Function*>& staticFunctions,
            const std::map<std::string, Function*>& propertyFunctions,
            const std::map<std::string, PyObject*>& classMembers,
            const std::map<std::string, Function*>& classMethods,
            bool isNew
        )
    {
        std::vector<HeldClass*> heldClassBases;

        for (auto c: bases) {
            heldClassBases.push_back(c->getHeldClass());
        }

        return new Class(
            inName,
            HeldClass::Make(
                inName,
                heldClassBases,
                isFinal,
                members,
                memberFunctions,
                staticFunctions,
                propertyFunctions,
                classMembers,
                classMethods,
                isNew
            )
        );
    }

    static const char* pyComparisonOpToMethodName(int pyComparisonOp) {
        switch (pyComparisonOp) {
            case Py_EQ: return "__eq__";
            case Py_NE: return "__ne__";
            case Py_LT: return "__lt__";
            case Py_GT: return "__gt__";
            case Py_LE: return "__le__";
            case Py_GE: return "__ge__";
        }

        return nullptr;
    }

    bool isFinal() {
        return m_heldClass->isFinal();
    }

    const std::vector<HeldClass*> getBases() const {
        return m_heldClass->getBases();
    }

    Class* renamed(std::string newName) {
        return new Class(newName, m_heldClass->renamed("Held(" + newName + ")"));
    }

    instance_ptr eltPtr(instance_ptr self, int64_t ix) const;

    void setAttribute(instance_ptr self, int64_t ix, instance_ptr elt) const;

    void delAttribute(instance_ptr self, int64_t ix) const;

    bool checkInitializationFlag(instance_ptr self, int64_t ix) const;

    ShaHash _computeIdentityHash(MutuallyRecursiveTypeGroup* groupHead = nullptr) {
        return ShaHash(1, m_typeCategory) + m_heldClass->identityHash(groupHead);
    }

    bool cmp(instance_ptr left, instance_ptr right, int pyComparisonOp, bool suppressExceptions);
    static bool cmpStatic(Class* T, instance_ptr left, instance_ptr right, int64_t pyComparisonOp);

    void deepcopyConcrete(
        instance_ptr dest,
        instance_ptr src,
        DeepcopyContext& context
    ) {
        //layout_ptr& destRecordPtr = *(layout**)dest;
        layout_ptr srcRecordPtr = instanceToLayout(src);

        auto it = context.alreadyAllocated.find((instance_ptr)srcRecordPtr);

        if (it == context.alreadyAllocated.end()) {
            // we could have a pointer to a subclass of 'this', in which case
            // the layout could be larger and have more fields than
            // mHeldClass would indicate.
            HeldClass* actualHeldClassType = srcRecordPtr->vtable->mType;

            layout_ptr destRecordPtr = (layout_ptr)context.slab->allocate(
                sizeof(layout) + actualHeldClassType->bytecount(),
                this
            );

            destRecordPtr->refcount = 0;
            destRecordPtr->vtable = srcRecordPtr->vtable;

            actualHeldClassType->deepcopy(
                destRecordPtr->data,
                srcRecordPtr->data,
                context
            );

            context.alreadyAllocated[(instance_ptr)srcRecordPtr] = (instance_ptr)destRecordPtr;
        }

        ((layout_ptr)context.alreadyAllocated[(instance_ptr)srcRecordPtr])->refcount++;

        initializeInstance(
            dest,
            (layout_ptr)context.alreadyAllocated[(instance_ptr)srcRecordPtr],
            instanceToDispatchTableIndex(src)
        );
    }

    size_t deepBytecountConcrete(instance_ptr instance, std::unordered_set<void*>& alreadyVisited, std::set<Slab*>* outSlabs) {
        layout_ptr p = instanceToLayout(instance);

        if (alreadyVisited.find((void*)p) != alreadyVisited.end()) {
            return 0;
        }

        alreadyVisited.insert((void*)p);

        if (outSlabs && Slab::slabForAlloc(p)) {
            outSlabs->insert(Slab::slabForAlloc(p));
            return 0;
        }

        HeldClass* actualHeldClassType = p->vtable->mType;

        return actualHeldClassType->deepBytecount(p->data, alreadyVisited, outSlabs) +
            bytesRequiredForAllocation(sizeof(layout) + actualHeldClassType->bytecount());
    }

    template<class buf_t>
    void deserialize(instance_ptr self, buf_t& buffer, size_t inWireType, bool asIfFinal=false) {
        if (isFinal() || asIfFinal) {
            int64_t id = -1;
            bool hasMemo = false;

            buffer.consumeCompoundMessage(inWireType, [&](size_t fieldNumber, size_t wireType) {
                if (fieldNumber == 0) {
                    assertWireTypesEqual(wireType, WireType::VARINT);
                    id = buffer.readUnsignedVarint();

                    void* ptr = buffer.lookupCachedPointer(id);

                    if (ptr) {
                        hasMemo = true;
                        copy_constructor(self, (instance_ptr)&ptr);
                    }
                }
                if (fieldNumber == 1) {
                    if (id == -1 || hasMemo) {
                        throw std::runtime_error("Corrupt Class instance");
                    }

                    initializeInstance(
                        self,
                        (layout*)tp_malloc(
                            sizeof(layout) + m_heldClass->bytecount()
                        ),
                        0
                    );

                    layout& record = *instanceToLayout(self);
                    record.refcount = 2;
                    record.vtable = m_heldClass->getVTable();

                    buffer.addCachedPointer(id, instanceToLayout(self), this);

                    m_heldClass->deserialize(record.data, buffer, wireType);
                }
            });
        } else {
            Type* actualType = nullptr;
            bool hasBody = false;

            buffer.consumeCompoundMessage(inWireType, [&](size_t fieldNumber, size_t subWireType) {
                if (fieldNumber == 0) {
                    if (actualType) {
                        throw std::runtime_error("Corrupt non-final class instance: multiple type definitions");
                    }

                    actualType = buffer.getContext().deserializeNativeType(buffer, subWireType);
                    if (!actualType || actualType->getTypeCategory() != Type::TypeCategory::catClass) {
                        throw std::runtime_error("Deserialized class type was not a class!");
                    }
                } else if (fieldNumber == 1) {
                    if (hasBody) {
                        throw std::runtime_error("Corrupt non-final class instance: multiple bodies");
                    }
                    if (!actualType) {
                        throw std::runtime_error("Corrupt non-final class instance: body before type");
                    }

                    //recursively call into the serializer for the actual known type
                    ((Class*)actualType)->deserialize(self, buffer, subWireType, true);
                    hasBody = true;

                    // set the dispatch index of the new object
                    int index = ((Class*)actualType)->getHeldClass()->getMroIndex(this->getHeldClass());
                    if (index < 0) {
                        throw std::runtime_error("Corrupt non-final class instance: realized class is not a subclass");
                    }

                    initializeInstance(self, instanceToLayout(self), index);

                    if (instanceToDispatchTableIndex(self) != index) {
                        throw std::runtime_error("failed to set instance index");
                    }
                } else {
                    throw std::runtime_error("Corrupt non-final class instance: invalid field number");
                }
            });

            if (!hasBody) {
                throw std::runtime_error("Corrupt non-final class instance: body not initialized");
            }
        }
    }

    template<class buf_t>
    void serialize(instance_ptr self, buf_t& buffer, size_t fieldNumber, bool asIfFinal=false) {
        if (isFinal() || asIfFinal) {
            layout& l = *instanceToLayout(self);

            uint32_t id;
            bool isNew;
            std::tie(id, isNew) = buffer.cachePointer(&l, this);

            if (!isNew) {
                buffer.writeBeginSingle(fieldNumber);
                buffer.writeUnsignedVarintObject(0, id);
                return;
            }

            buffer.writeBeginCompound(fieldNumber);
            buffer.writeUnsignedVarintObject(0, id);
            m_heldClass->serialize(l.data, buffer, 1);
            buffer.writeEndCompound();
        } else {
            Class* actualType = actualTypeForLayout(self);

            buffer.writeBeginCompound(fieldNumber);

            buffer.getContext().serializeNativeType(actualType, buffer, 0);
            actualType->serialize(self, buffer, 1, true);

            buffer.writeEndCompound();
        }
    }

    void repr(instance_ptr self, ReprAccumulator& stream, bool isStr);

    typed_python_hash_type hash(instance_ptr left);

    // create a new Class instance by initializing the held class using
    // 'initializer(instance_ptr memberData, int memberIx)' for each member
    template<class sub_constructor>
    void constructor(instance_ptr self, const sub_constructor& initializer) const {
        initializeInstance(self, (layout*)tp_malloc(sizeof(layout) + m_heldClass->bytecount()), 0);

        layout& l = *instanceToLayout(self);
        l.refcount = 1;
        l.vtable = m_heldClass->getVTable();

        try {
            m_heldClass->constructor(l.data, initializer);
        } catch (...) {
            tp_free(instanceToLayout(self));
        }
    }

    // create a new Class instance by initializing the held class using
    // 'initializer(instance_ptr data)'
    template<class sub_constructor>
    void constructorInitializingHeld(instance_ptr self, const sub_constructor& initializer) const {
        initializeInstance(self, (layout*)tp_malloc(sizeof(layout) + m_heldClass->bytecount()), 0);

        layout& l = *instanceToLayout(self);
        l.refcount = 1;
        l.vtable = m_heldClass->getVTable();

        try {
            initializer(l.data);
        } catch (...) {
            tp_free(instanceToLayout(self));
        }
    }

    void constructor(instance_ptr self, bool allowEmpty=false);

    void destroy(instance_ptr self);

    int64_t refcount(instance_ptr self);

    void copy_constructor(instance_ptr self, instance_ptr other);

    void assign(instance_ptr self, instance_ptr other);

    bool getMemberIsNonempty(int index) const {
        return m_heldClass->getMemberIsNonempty(index);
    }

    Type* getMemberType(int index) const {
        return m_heldClass->getMemberType(index);
    }

    const std::string& getMemberName(int index) const {
        return m_heldClass->getMemberName(index);
    }

    int getMemberIndex(const char* name) const {
        return m_heldClass->getMemberIndex(name);
    }

    BoundMethod* getMemberFunctionMethodType(const char* name) const {
        return m_heldClass->getMemberFunctionMethodType(name, false);
    }

    const std::vector<MemberDefinition>& getMembers() const {
        return m_heldClass->getMembers();
    }

    const std::map<std::string, Function*>& getMemberFunctions() const {
        return m_heldClass->getMemberFunctions();
    }

    const std::map<std::string, Function*>& getStaticFunctions() const {
        return m_heldClass->getStaticFunctions();
    }

    const std::map<std::string, Function*>& getClassMethods() const {
        return m_heldClass->getClassMethods();
    }

    const std::map<std::string, PyObject*>& getClassMembers() const {
        return m_heldClass->getClassMembers();
    }

    const std::map<std::string, Function*>& getPropertyFunctions() const {
        return m_heldClass->getPropertyFunctions();
    }

    const std::vector<MemberDefinition>& getOwnMembers() const {
        return m_heldClass->getOwnMembers();
    }

    const std::map<std::string, Function*>& getOwnMemberFunctions() const {
        return m_heldClass->getOwnMemberFunctions();
    }

    const std::map<std::string, Function*>& getOwnStaticFunctions() const {
        return m_heldClass->getOwnStaticFunctions();
    }

    const std::map<std::string, PyObject*>& getOwnClassMembers() const {
        return m_heldClass->getOwnClassMembers();
    }

    const std::map<std::string, Function*>& getOwnClassMethods() const {
        return m_heldClass->getOwnClassMethods();
    }

    const std::map<std::string, Function*>& getOwnPropertyFunctions() const {
        return m_heldClass->getOwnPropertyFunctions();
    }

    int memberNamed(const char* c) const {
        return m_heldClass->memberNamed(c);
    }

    HeldClass* getHeldClass() const {
        return m_heldClass;
    }

    bool isSubclassOfConcrete(Type* otherType) {
        if (otherType->getTypeCategory() != Type::TypeCategory::catClass) {
            return false;
        }

        return m_heldClass->isSubclassOfConcrete(((Class*)otherType)->getHeldClass());
    }

private:
    HeldClass* m_heldClass;
};
