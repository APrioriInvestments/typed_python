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

#include <unordered_set>

class HeldClass;
class Class;

class VTable {
public:
    VTable(HeldClass* inClass) :
        mType(inClass)
    {
    }

    HeldClass* mType;
};

typedef VTable* vtable_ptr;

//a class held directly inside of another object
class HeldClass : public Type {
public:
    HeldClass(std::string inName,
          const std::vector<HeldClass*>& baseClasses,
          const std::vector<std::tuple<std::string, Type*, Instance> >& members,
          const std::map<std::string, Function*>& memberFunctions,
          const std::map<std::string, Function*>& staticFunctions,
          const std::map<std::string, Function*>& propertyFunctions,
          const std::map<std::string, PyObject*>& classMembers
          ) :
            Type(catHeldClass),
            m_vtable(new VTable(this)),
            m_bases(baseClasses),
            m_classType(nullptr),
            m_own_members(members),
            m_own_memberFunctions(memberFunctions),
            m_own_staticFunctions(staticFunctions),
            m_own_propertyFunctions(propertyFunctions),
            m_own_classMembers(classMembers),
            m_hasComparisonOperators(false)
    {
        m_name = inName;

        initializeMRO();

        if (m_memberFunctions.find("__eq__") != m_memberFunctions.end()) { m_hasComparisonOperators = true; }
        if (m_memberFunctions.find("__ne__") != m_memberFunctions.end()) { m_hasComparisonOperators = true; }
        if (m_memberFunctions.find("__lt__") != m_memberFunctions.end()) { m_hasComparisonOperators = true; }
        if (m_memberFunctions.find("__gt__") != m_memberFunctions.end()) { m_hasComparisonOperators = true; }
        if (m_memberFunctions.find("__le__") != m_memberFunctions.end()) { m_hasComparisonOperators = true; }
        if (m_memberFunctions.find("__ge__") != m_memberFunctions.end()) { m_hasComparisonOperators = true; }

        endOfConstructorInitialization(); // finish initializing the type object.
    }

    bool isBinaryCompatibleWithConcrete(Type* other);

    template<class visitor_type>
    void _visitContainedTypes(const visitor_type& visitor) {
        for (auto& o: m_members) {
            visitor(std::get<1>(o));
        }
    }

    template<class visitor_type>
    void _visitReferencedTypes(const visitor_type& visitor) {
        for (auto& o: m_members) {
            visitor(std::get<1>(o));
        }
        for (auto& o: m_memberFunctions) {
            Type* t = std::get<1>(o);
            visitor(t);
            assert(t == std::get<1>(o));
        }
        for (auto& o: m_staticFunctions) {
            Type* t = std::get<1>(o);
            visitor(t);
            assert(t == std::get<1>(o));
        }
    }

    bool _updateAfterForwardTypesChanged();

    static HeldClass* Make(
            std::string inName,
            const std::vector<HeldClass*>& bases,
            const std::vector<std::tuple<std::string, Type*, Instance> >& members,
            const std::map<std::string, Function*>& memberFunctions,
            const std::map<std::string, Function*>& staticFunctions,
            const std::map<std::string, Function*>& propertyFunctions,
            const std::map<std::string, PyObject*>& classMembers
            )
    {
        return new HeldClass(inName, bases, members, memberFunctions, staticFunctions, propertyFunctions, classMembers);
    }

    // this gets called by Class. These types are always produced in pairs.
    void setClassType(Class* inClass) {
        if (m_classType) {
            throw std::runtime_error("Class is already set.");
        }
        m_classType = inClass;
    }

    Class* getClassType() const {
        return m_classType;
    }

    HeldClass* renamed(std::string newName) {
        return Make(
            newName,
            m_bases,
            m_members,
            m_memberFunctions,
            m_staticFunctions,
            m_propertyFunctions,
            m_classMembers
        );
    }

    // HeldClass is laid out as a VTable ptr, a set of member initialization fields, and then
    // the actual members.
    vtable_ptr& vtableFor(instance_ptr self) const {
        return *(vtable_ptr*)self;
    }

    instance_ptr eltPtr(instance_ptr self, int64_t ix) const {
        return self + m_byte_offsets[ix];
    }

    bool cmp(instance_ptr left, instance_ptr right, int pyComparisonOp, bool suppressExceptions);

    template<class buf_t>
    void deserialize(instance_ptr self, buf_t& buffer, size_t wireType) {
        vtableFor(self) = m_vtable;

        for (long k = 0; k < m_members.size();k++) {
            clearInitializationFlag(self, k);
        }

        buffer.consumeCompoundMessage(wireType, [&](size_t fieldNumber, size_t subWireType) {
            if (fieldNumber < m_members.size()) {
                getMemberType(fieldNumber)->deserialize(eltPtr(self,fieldNumber), buffer, subWireType);
                setInitializationFlag(self, fieldNumber);
            } else {
                buffer.finishReadingMessageAndDiscard(subWireType);
            }
        });
    }

    template<class buf_t>
    void serialize(instance_ptr self, buf_t& buffer, size_t fieldNumber) {
        buffer.writeBeginCompound(fieldNumber);

        for (long k = 0; k < m_members.size();k++) {
            bool isInitialized = checkInitializationFlag(self, k);
            if (isInitialized) {
                std::get<1>(m_members[k])->serialize(eltPtr(self,k),buffer, k);
            }
        }

        buffer.writeEndCompound();
    }

    void repr(instance_ptr self, ReprAccumulator& stream);

    typed_python_hash_type hash(instance_ptr left);

    template<class sub_constructor>
    void constructor(instance_ptr self, const sub_constructor& initializer) const {
        vtableFor(self) = m_vtable;

        for (int64_t k = 0; k < m_members.size(); k++) {
            try {
                initializer(eltPtr(self, k), k);
                setInitializationFlag(self, k);
            } catch(...) {
                for (long k2 = k-1; k2 >= 0; k2--) {
                    std::get<1>(m_members[k2])->destroy(eltPtr(self,k2));
                }
                throw;
            }
        }
    }

    void setAttribute(instance_ptr self, int memberIndex, instance_ptr other) const;

    void emptyConstructor(instance_ptr self);

    //don't default construct classes
    static bool wantsToDefaultConstruct(Type* t) {
        return t->is_default_constructible() && t->getTypeCategory() != TypeCategory::catClass;
    }

    void constructor(instance_ptr self);

    void destroy(instance_ptr self);

    void copy_constructor(instance_ptr self, instance_ptr other);

    void assign(instance_ptr self, instance_ptr other);

    bool checkInitializationFlag(instance_ptr self, int memberIndex) const {
        int byte = memberIndex / 8 + sizeof(vtable_ptr);
        int bit = memberIndex % 8;
        return bool( ((uint8_t*)self)[byte] & (1 << bit) );
    }

    void setInitializationFlag(instance_ptr self, int memberIndex) const;

    void clearInitializationFlag(instance_ptr self, int memberIndex) const;

    Type* getMemberType(int index) const {
        return std::get<1>(m_members[index]);
    }

    const std::string& getMemberName(int index) const {
        return std::get<0>(m_members[index]);
    }

    bool memberHasDefaultValue(int index) const {
        return std::get<2>(m_members[index]).type()->getTypeCategory() != TypeCategory::catNone;
    }

    const Instance& getMemberDefaultValue(int index) const {
        return std::get<2>(m_members[index]);
    }

    const std::vector<HeldClass*> getBases() const {
        return m_bases;
    }

    const std::vector<std::tuple<std::string, Type*, Instance> >& getMembers() const {
        return m_members;
    }

    const std::map<std::string, Function*>& getMemberFunctions() const {
        return m_memberFunctions;
    }

    const std::map<std::string, Function*>& getStaticFunctions() const {
        return m_staticFunctions;
    }

    const std::map<std::string, PyObject*>& getClassMembers() const {
        return m_classMembers;
    }

    const std::map<std::string, Function*>& getPropertyFunctions() const {
        return m_propertyFunctions;
    }

    const std::vector<size_t>& getOffsets() const {
        return m_byte_offsets;
    }

    int memberNamed(const char* c) const;

    bool hasAnyComparisonOperators() const {
        return m_hasComparisonOperators;
    }

    void initializeMRO() {
        // this is not how the MRO actually works, but we have yet to actually
        // code it correctly.
        std::set<HeldClass*> seen;

        std::function<void (HeldClass*)> visit = [&](HeldClass* cls) {
            if (seen.find(cls) == seen.end()) {
                seen.insert(cls);
                m_mro.push_back(cls);
                m_bases_as_set.insert(cls);

                for (auto b: cls->getBases()) {
                    visit(b);
                }
            }
        };

        visit(this);

        std::set<std::string> membersSoFar;

        // build our own method resolution table directly from our parents.
        for (HeldClass* base: m_mro) {
            for (auto nameAndObj: base->m_own_classMembers) {
                if (m_classMembers.find(nameAndObj.first) == m_classMembers.end()) {
                    m_classMembers[nameAndObj.first] = incref(nameAndObj.second);
                }
            }

            for (auto nameAndType: base->m_own_members) {
                if (membersSoFar.find(std::get<0>(nameAndType)) != membersSoFar.end()) {
                    throw std::runtime_error("Can't redefine member named " + std::get<0>(nameAndType));
                }
                membersSoFar.insert(std::get<0>(nameAndType));

                m_members.push_back(nameAndType);
            }

            mergeInto(m_memberFunctions, base->m_own_memberFunctions);
            mergeInto(m_staticFunctions, base->m_own_staticFunctions);
            mergeInto(m_propertyFunctions, base->m_own_propertyFunctions);
        }
    }

    // given some function defininitions by name, add them to a target dictionary.
    // if the function is new just add it. Otherwise merge it in to the existing set
    // of method specializations.
    static void mergeInto(
            std::map<std::string, Function*>& target,
            const std::map<std::string, Function*>& source
            )
    {
        for (auto nameAndFunc: source) {
            auto it = target.find(nameAndFunc.first);
            if (it == target.end()) {
                target[nameAndFunc.first] = nameAndFunc.second;
            } else {
                target[nameAndFunc.first] = Function::merge(target[nameAndFunc.first], nameAndFunc.second);
            }
        }
    }


    bool isSubclassOfConcrete(Type* otherType) {
        return m_bases_as_set.find((HeldClass*)otherType) != m_bases_as_set.end();
    }

    VTable* getVTable() const {
        return m_vtable;
    }

private:
    std::vector<size_t> m_byte_offsets;

    std::vector<HeldClass*> m_bases;

    // equivalent to python's method resolution order, so we can
    // search for methods at runtime.
    std::vector<HeldClass*> m_mro;

    std::unordered_set<HeldClass*> m_bases_as_set;

    VTable* m_vtable;

    Class* m_classType; //the non-held version of this class

    //the members we
    std::vector<std::tuple<std::string, Type*, Instance> > m_members;

    std::map<std::string, Function*> m_memberFunctions;

    std::map<std::string, Function*> m_staticFunctions;

    std::map<std::string, Function*> m_propertyFunctions;

    std::map<std::string, PyObject*> m_classMembers;

    // the original members we were provided with
    std::vector<std::tuple<std::string, Type*, Instance> > m_own_members;

    std::map<std::string, Function*> m_own_memberFunctions;

    std::map<std::string, Function*> m_own_staticFunctions;

    std::map<std::string, Function*> m_own_propertyFunctions;

    std::map<std::string, PyObject*> m_own_classMembers;

    bool m_hasComparisonOperators;
};

