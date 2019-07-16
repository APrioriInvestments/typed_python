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

class Instance;

class Class : public Type {
    class layout {
    public:
        std::atomic<int64_t> refcount;
        unsigned char data[];
    };

public:
    Class(HeldClass* inClass) :
            Type(catClass),
            m_heldClass(inClass)
    {
        m_size = sizeof(layout*);
        m_is_default_constructible = inClass->is_default_constructible();
        m_name = m_heldClass->name();
        m_is_simple = false;

        endOfConstructorInitialization(); // finish initializing the type object.

        inClass->setClassType(this);
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

    /*****
    //we should have this, except that natively generated code doesn't know how to write
    //this field yet, so the vtable will be null in those cases.

    Type* pickConcreteSubclassConcrete(instance_ptr self) {
        layout& l = **(layout**)self;

        return m_heldClass->vtableFor(l.data)->mType->getClassType();
    }
    ******/

    bool _updateAfterForwardTypesChanged();

    static Class* Make(
            std::string inName,
            const std::vector<Class*>& bases,
            const std::vector<std::tuple<std::string, Type*, Instance> >& members,
            const std::map<std::string, Function*>& memberFunctions,
            const std::map<std::string, Function*>& staticFunctions,
            const std::map<std::string, Function*>& propertyFunctions,
            const std::map<std::string, PyObject*>& classMembers
            )
    {
        std::vector<HeldClass*> heldClassBases;

        for (auto c: bases) {
            heldClassBases.push_back(c->getHeldClass());
        }

        return new Class(
            HeldClass::Make(
                inName,
                heldClassBases,
                members,
                memberFunctions,
                staticFunctions,
                propertyFunctions,
                classMembers
            )
        );
    }

    Class* renamed(std::string newName) {
        return new Class(m_heldClass->renamed(newName));
    }

    instance_ptr eltPtr(instance_ptr self, int64_t ix) const;

    void setAttribute(instance_ptr self, int64_t ix, instance_ptr elt) const;

    bool checkInitializationFlag(instance_ptr self, int64_t ix) const;

    bool cmp(instance_ptr left, instance_ptr right, int pyComparisonOp, bool suppressExceptions);

    template<class buf_t>
    void deserialize(instance_ptr self, buf_t& buffer, size_t inWireType) {
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

                *(layout**)self = (layout*)malloc(
                    sizeof(layout) + m_heldClass->bytecount()
                    );
                layout& record = **(layout**)self;
                record.refcount = 2;

                buffer.addCachedPointer(id, *((layout**)self), this);

                m_heldClass->deserialize(record.data, buffer, wireType);
            }
        });
    }

    template<class buf_t>
    void serialize(instance_ptr self, buf_t& buffer, size_t fieldNumber) {
        layout& l = **(layout**)self;

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
    }

    void repr(instance_ptr self, ReprAccumulator& stream);

    typed_python_hash_type hash(instance_ptr left);

    template<class sub_constructor>
    void constructor(instance_ptr self, const sub_constructor& initializer) const {
        *(layout**)self = (layout*)malloc(sizeof(layout) + m_heldClass->bytecount());
        layout& l = **(layout**)self;
        l.refcount = 1;

        try {
            m_heldClass->constructor(l.data, initializer);
        } catch (...) {
            free(*(layout**)self);
        }
    }

    void emptyConstructor(instance_ptr self);

    void constructor(instance_ptr self);

    void destroy(instance_ptr self);

    int64_t refcount(instance_ptr self);

    void copy_constructor(instance_ptr self, instance_ptr other);

    void assign(instance_ptr self, instance_ptr other);

    Type* getMemberType(int index) const {
        return m_heldClass->getMemberType(index);
    }

    const std::string& getMemberName(int index) const {
        return m_heldClass->getMemberName(index);
    }

    const std::vector<std::tuple<std::string, Type*, Instance> >& getMembers() const {
        return m_heldClass->getMembers();
    }

    const std::map<std::string, Function*>& getMemberFunctions() const {
        return m_heldClass->getMemberFunctions();
    }

    const std::map<std::string, Function*>& getStaticFunctions() const {
        return m_heldClass->getStaticFunctions();
    }

    const std::map<std::string, PyObject*>& getClassMembers() const {
        return m_heldClass->getClassMembers();
    }

    const std::map<std::string, Function*>& getPropertyFunctions() const {
        return m_heldClass->getPropertyFunctions();
    }

    int memberNamed(const char* c) const {
        return m_heldClass->memberNamed(c);
    }

    HeldClass* getHeldClass() const {
        return m_heldClass;
    }

private:
    HeldClass* m_heldClass;
};

