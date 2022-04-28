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

#include <unordered_set>
#include <unordered_map>

class HeldClass;
class RefTo;
class Function;
class Class;

//this takes an instance_ptr for a Class object (not the HeldClass)
typedef void (*destructor_fun_type)(void* inst);

// represents a concrete call signature with positional arguments packed
// into the tuple and the named arguments packed into the named tuple.
// we'll end up with a each distinct function call signat
typedef std::tuple<Type*, Tuple*, NamedTuple*> function_call_signature_type;

typedef std::pair<std::string, function_call_signature_type> method_call_signature_type;

typedef void* untyped_function_ptr;

class ConstCharPtrsAreEqual {
public:
    bool operator()(const char* x, const char* y) const {
        return strcmp(x, y) == 0;
    }
};

class HashConstCharPtr {
public:
    int operator()(const char* str) const {
        int seed = 131;
        int hash = 0;

        while (*str) {
            hash = (hash * seed) + (*str);
            str++;
        }

        return hash & (0x7FFFFFFF);
    }
};

// models a single class member definition
class MemberDefinition {
public:
    MemberDefinition(
        const std::string& inName,
        Type* inType,
        const Instance& inDefaultValue,
        bool nonempty
    ) :
        mName(inName),
        mType(inType),
        mDefaultValue(inDefaultValue),
        mIsNonempty(nonempty)
    {}

    const std::string& getName() const {
        return mName;
    }

    Type*& getType() {
        return mType;
    }

    Type* getType() const {
        return mType;
    }

    const Instance& getDefaultValue() const {
        return mDefaultValue;
    }

    bool getIsNonempty() const {
        return mIsNonempty;
    }

private:
    std::string mName;
    Type* mType;
    Instance mDefaultValue;
    bool mIsNonempty;
};


/****
ClassDispatchTable

Contains the dispatch pointers for each entrypoint to compiled code. Each instance represents
everything we know about dispatching 'Subclass as Class', where an instance of 'Subclass' needs
to masquerade as an instance of 'Class' in compiled code.

At every call site for things that look like 'Class' we have an integer representing that
dispatch.  Every 'Subclass as Class' dispatch table will need to have an entry for that id,
so that compiled code can find the function pointer.
*****/


class ClassDispatchTable {
public:
    ClassDispatchTable(HeldClass* implementingClass, HeldClass* interfaceClass) :
        mImplementingClass(implementingClass),
        mInterfaceClass(interfaceClass),
        mFuncPtrs(nullptr),
        mFuncPtrsUsed(0),
        mFuncPtrsAllocated(0),
        // these members are explicitly leaked so that the layout of the C class is
        // comprehensible to the llvm code layer. maps and sets have a nontrivial layout,
        // and so it's more stable to just hold them as pointers.
        mDispatchIndices(*new std::map<method_call_signature_type, size_t>()),
        mDispatchDefinitions(*new std::map<size_t, method_call_signature_type>()),
        mIndicesNeedingDefinition(*new std::set<size_t>())
    {
    }

    // initialize this ClassDispatchTable given the table for the interface we're implementing.
    // If we are 'Subclass' implementing 'Base', 'baseAsBase' will be the table for 'Base' implementing
    // 'Base', which we need to see because it will contain an entry for every dispatch that's currently
    // known for 'Base'.
    void initialize(ClassDispatchTable* baseAsBase) {
        mDispatchIndices = baseAsBase->mDispatchIndices;
        mDispatchDefinitions = baseAsBase->mDispatchDefinitions;

        mFuncPtrsUsed = baseAsBase->mFuncPtrsUsed;
        mFuncPtrsAllocated = baseAsBase->mFuncPtrsAllocated;

        mFuncPtrs = (untyped_function_ptr*)tp_malloc(sizeof(untyped_function_ptr) * mFuncPtrsAllocated);

        for (long k = 0; k < mFuncPtrsUsed; k++) {
            mFuncPtrs[k] = nullptr;
            mIndicesNeedingDefinition.insert(k);
            globalPointersNeedingCompile().insert(std::make_pair(this, k));
        }

        allocateUpcastDispatchTables();
    }

    size_t allocateMethodDispatch(std::string funcName, const function_call_signature_type& signature) {
        assertHoldingTheGil();

        auto it = mDispatchIndices.find(method_call_signature_type(funcName, signature));
        if (it != mDispatchIndices.end()) {
            return it->second;
        }

        size_t newIndex = mDispatchIndices.size();

        mDispatchIndices[method_call_signature_type(funcName, signature)] = newIndex;
        mDispatchDefinitions[newIndex] = method_call_signature_type(funcName, signature);

        // check if we need to allocate a bigger function pointer table. If we do,
        // we must leave the existing one in place, because compiled code may be
        // reading from it concurrently. This functino is holding the GIL, but
        // compiled code doesn't have to do that.
        if (mFuncPtrsUsed >= mFuncPtrsAllocated) {
            mFuncPtrsAllocated = (mFuncPtrsAllocated + 1) * 2;
            untyped_function_ptr* newTable = (untyped_function_ptr*)tp_malloc(sizeof(untyped_function_ptr) * mFuncPtrsAllocated);
            for (long k = 0; k < mFuncPtrsUsed; k++) {
                newTable[k] = mFuncPtrs[k];
            }

            //TODO: don't just leak this. Put it in a queue that we can clean up in
            //the background after we are certain that any compiled code that was
            //reading from the old table will have seen this.
            mFuncPtrs = newTable;
        }

        mFuncPtrsUsed += 1;

        if (mFuncPtrsUsed != mDispatchIndices.size()) {
            throw std::runtime_error("Somehow we lost track of how many function pointers we're using.");
        }

        mFuncPtrs[newIndex] = nullptr;

        mIndicesNeedingDefinition.insert(newIndex);
        globalPointersNeedingCompile().insert(std::make_pair(this, newIndex));

        return newIndex;
    }

    void define(size_t index, untyped_function_ptr ptr) {
        if (!ptr) {
            throw std::runtime_error("Tried to define a function pointer in a VTable.ClassDispatchTable to be null.");
        }

        if (mFuncPtrs[index] == ptr) {
            return;
        }

        auto it = mIndicesNeedingDefinition.find(index);
        if (it == mIndicesNeedingDefinition.end()) {
            throw std::runtime_error("Tried to define a function pointer in a VTable.ClassDispatchTable twice.");
        }

        mIndicesNeedingDefinition.erase(it);

        mFuncPtrs[index] = ptr;
    }

    // a set of slots for function pointers that need to be compiled. We only add to this from
    // this code. Clients of this object pop these off and compile them.
    static std::set<std::pair<ClassDispatchTable*, size_t> >& globalPointersNeedingCompile() {
        static std::set<std::pair<ClassDispatchTable*, size_t> > pointers;

        return pointers;
    }

    HeldClass* getImplementingClass() const {
        return mImplementingClass;
    }

    HeldClass* getInterfaceClass() const {
        return mInterfaceClass;
    }

    method_call_signature_type dispatchDefinitionForSlot(size_t slotIx) const {
        auto it = mDispatchDefinitions.find(slotIx);
        if (it == mDispatchDefinitions.end()) {
            throw std::runtime_error("Invalid slot " + format(slotIx));
        }
        return it->second;
    }

    void allocateUpcastDispatchTables();

    untyped_function_ptr get(size_t slot) {
        return mFuncPtrs[slot];
    }

private:
    // the class actually represented by this instance
    HeldClass* mImplementingClass;

    // the class we're pretending to be
    HeldClass* mInterfaceClass;

    untyped_function_ptr* mFuncPtrs;

    // for each base class of mInterfaceClass, what is the MRO index of that class
    // in the implementing class. We need this to allow compiled code to upcast
    // an already-upcast class pointer. For instance if we have Base, Child, ChildChild,
    // and we know an instance of ChildChild as 'Child', and we want to cast it as 'Base',
    // we need to find out that we're using index '2' for Base, because we have ChildChild's
    // vtable.
    uint16_t* mUpcastDispatches;

    size_t mFuncPtrsAllocated;

    size_t mFuncPtrsUsed;

    std::map<method_call_signature_type, size_t>& mDispatchIndices;

    std::map<size_t, method_call_signature_type>& mDispatchDefinitions;

    std::set<size_t>& mIndicesNeedingDefinition;
};

class VTable {
public:
    VTable(HeldClass* inClass) :
        mType(inClass),
        mCompiledDestructorFun(nullptr),
        mDispatchTables(nullptr)
    {
    }

    /*****
    When VTable is constructed, the Class type object itself isn't complete. This function
    is responsible for completing initialization by creating dispatch tables for all of the
    base classes we might masquerade as.
    *****/

    void finalize(ClassDispatchTable* dispatchers, long inInitializationBitByteCount, long count) {
        mDispatchTables = dispatchers;
        mInitializationBitByteCount = inInitializationBitByteCount;
    }

    void installDestructor(destructor_fun_type fun) {
        if (mCompiledDestructorFun == fun) {
            return;
        }
        if (mCompiledDestructorFun) {
            throw std::runtime_error("Can't change the compiled destructor!");
        }

        mCompiledDestructorFun = fun;
    }

    HeldClass* mType;

    // null, unless we've compiled a destructor for this function in which case we can just use that.

    destructor_fun_type mCompiledDestructorFun;

    // for each base class, we have a dispatch table we use when we are interacting with the class from
    // code that wants to view the child class as if it were the base class. we encode which base class
    // by index using the top 16 bits of the class pointer.

    ClassDispatchTable* mDispatchTables;

    // the number of bytes of initialization bits. We have to keep track of how many
    // members are in this particular layout, so that we know how far ahead to look in
    // the object when we are looking up a particular data member.
    int64_t mInitializationBitByteCount;

    Type* mClassType;
};

//a class held directly inside of another object
class HeldClass : public Type {
public:
    HeldClass(std::string inName,
          const std::vector<HeldClass*>& baseClasses,
          bool isFinal,
          const std::vector<MemberDefinition>& members,
          const std::map<std::string, Function*>& memberFunctions,
          const std::map<std::string, Function*>& staticFunctions,
          const std::map<std::string, Function*>& propertyFunctions,
          const std::map<std::string, PyObject*>& classMembers,
          const std::map<std::string, Function*>& classMethods,
          // set to True if this is the first time the class is being created
          // and we need to make copies of all the function objects so that
          // they know who their methods are.
          bool isNew
          ) :
            Type(catHeldClass),
            m_vtable(new VTable(this)),
            m_bases(baseClasses),
            m_classType(nullptr),
            m_is_final(isFinal),
            m_own_members(members),
            m_own_memberFunctions(memberFunctions),
            m_own_staticFunctions(staticFunctions),
            m_own_propertyFunctions(propertyFunctions),
            m_own_classMembers(classMembers),
            m_own_classMethods(classMethods),
            m_hasComparisonOperators(false),
            m_hasGetAttributeMagicMethod(false),
            m_hasGetAttrMagicMethod(false),
            m_hasSetAttrMagicMethod(false),
            m_hasDelAttrMagicMethod(false),
            m_refToType(nullptr)
    {
        m_name = inName;

        if (isNew) {
            for (auto& nameAndF: m_own_memberFunctions) {
                nameAndF.second = nameAndF.second->withMethodOf(this);
            }

            for (auto& nameAndF: m_own_classMethods) {
                nameAndF.second = nameAndF.second->withMethodOf(this);
            }

            for (auto& nameAndF: m_own_staticFunctions) {
                nameAndF.second = nameAndF.second->withMethodOf(this);
            }
        }
    }

    ShaHash _computeIdentityHash(MutuallyRecursiveTypeGroup* groupHead = nullptr) {
        ShaHash res = ShaHash(1, m_typeCategory) + ShaHash(m_name);

        res += ShaHash(0);
        for (auto b: m_bases) {
            res += b->identityHash(groupHead);
        }

        res += ShaHash(1);
        for (auto tup: m_own_members) {
            res += ShaHash(tup.getName());
            res += ShaHash(tup.getType()->identityHash(groupHead));
            res += MutuallyRecursiveTypeGroup::tpInstanceShaHash(tup.getDefaultValue(), groupHead);
            res += ShaHash(tup.getIsNonempty());
        }

        res += ShaHash(2);
        for (auto nameAndFun: m_own_memberFunctions) {
            res += ShaHash(nameAndFun.first);
            res += nameAndFun.second->identityHash(groupHead);
        }

        res += ShaHash(3);
        for (auto nameAndFun: m_own_staticFunctions) {
            res += ShaHash(nameAndFun.first);
            res += nameAndFun.second->identityHash(groupHead);
        }

        res += ShaHash(4);
        for (auto nameAndFun: m_own_propertyFunctions) {
            res += ShaHash(nameAndFun.first);
            res += nameAndFun.second->identityHash(groupHead);
        }

        res += ShaHash(5);
        for (auto nameAndFun: m_own_classMembers) {
            res += ShaHash(nameAndFun.first);
            res += MutuallyRecursiveTypeGroup::pyObjectShaHash(nameAndFun.second, groupHead);
        }

        res += ShaHash(6);
        for (auto nameAndFun: m_own_classMethods) {
            res += ShaHash(nameAndFun.first);
            res += nameAndFun.second->identityHash(groupHead);
        }

        return res;
    }

    template<class visitor_type>
    void _visitCompilerVisibleInstances(const visitor_type& visitor) {
        for (auto tup: m_own_members) {
            visitor(tup.getDefaultValue());
        }
    }

    bool isBinaryCompatibleWithConcrete(Type* other);

    template<class visitor_type>
    void _visitContainedTypes(const visitor_type& visitor) {
        for (auto& o: m_members) {
            visitor(o.getType());
        }
    }

    template<class visitor_type>
    void _visitReferencedTypes(const visitor_type& visitor) {
        for (auto& b: m_bases) {
            Type* baseT = b;
            visitor(baseT);
            if (b != baseT) {
                throw std::runtime_error("Somehow, we modified the base type of a HeldClass?");
            }
        }
        for (auto& o: m_own_members) {
            // this is expected to actually modify the type
            // if its a forward.
            visitor(o.getType());
        }
        for (auto& o: m_own_memberFunctions) {
            Type* t = o.second;
            visitor(t);
            assert(t == o.second);
        }
        for (auto& o: m_own_staticFunctions) {
            Type* t = o.second;
            visitor(t);
            assert(t == o.second);
        }
        for (auto& o: m_own_classMethods) {
            Type* t = o.second;
            visitor(t);
            assert(t == o.second);
        }
        for (auto& o: m_own_propertyFunctions) {
            Type* t = o.second;
            visitor(t);
            assert(t == o.second);
        }
        for (auto& o: m_members) {
            // this is expected to actually modify the type
            // if its a forward.
            visitor(o.getType());
        }
        for (auto& o: m_memberFunctions) {
            Type* t = o.second;
            visitor(t);
            assert(t == o.second);
        }
        for (auto& o: m_staticFunctions) {
            Type* t = o.second;
            visitor(t);
            assert(t == o.second);
        }
        for (auto& o: m_propertyFunctions) {
            Type* t = o.second;
            visitor(t);
            assert(t == o.second);
        }
    }

    template<class visitor_type>
    void _visitCompilerVisiblePythonObjects(const visitor_type& visitor) {
        for (auto nameAndFun: m_own_classMembers) {
            visitor(nameAndFun.second);
        }
    }

    bool _updateAfterForwardTypesChanged();

    static HeldClass* Make(
        std::string inName,
        const std::vector<HeldClass*>& bases,
        bool isFinal,
        const std::vector<MemberDefinition>& members,
        const std::map<std::string, Function*>& memberFunctions,
        const std::map<std::string, Function*>& staticFunctions,
        const std::map<std::string, Function*>& propertyFunctions,
        const std::map<std::string, PyObject*>& classMembers,
        const std::map<std::string, Function*>& classMethods,
        bool isNew
    );

    // this gets called by Class. These types are always produced in pairs.
    void setClassType(Class* inClass) {
        if (m_classType) {
            throw std::runtime_error("Class is already set.");
        }
        m_classType = inClass;
        m_vtable->mClassType = (Type*)inClass;
    }

    Class* getClassType() const {
        return m_classType;
    }

    RefTo* getRefToType();

    HeldClass* renamed(std::string newName) {
        return Make(
            newName,
            m_bases,
            m_is_final,
            m_own_members,
            m_own_memberFunctions,
            m_own_staticFunctions,
            m_own_propertyFunctions,
            m_own_classMembers,
            m_own_classMethods,
            true
        );
    }

    // HeldClass is laid out as a set of member initialization fields, and then
    // the actual members.
    instance_ptr eltPtr(instance_ptr self, int64_t ix) const {
        return self + m_byte_offsets[ix];
    }

    bool cmp(instance_ptr left, instance_ptr right, int pyComparisonOp, bool suppressExceptions);

    void deepcopyConcrete(
        instance_ptr dest,
        instance_ptr src,
        DeepcopyContext& context
    ) {
        for (int64_t k = 0; k < m_members.size(); k++) {
            if (checkInitializationFlag(src, k)) {
                m_members[k].getType()->deepcopy(
                    eltPtr(dest, k),
                    eltPtr(src, k),
                    context
                );
                setInitializationFlag(dest, k);
            }
        }
    }

    size_t deepBytecountConcrete(instance_ptr instance, std::unordered_set<void*>& alreadyVisited, std::set<Slab*>* outSlabs) {
        size_t res = 0;

        for (long k = 0; k < m_members.size(); k++) {
            res += m_members[k].getType()->deepBytecount(eltPtr(instance, k), alreadyVisited, outSlabs);
        }

        return res;
    }

    template<class buf_t>
    void deserialize(instance_ptr self, buf_t& buffer, size_t wireType) {
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
                m_members[k].getType()->serialize(eltPtr(self,k),buffer, k);
            }
        }

        buffer.writeEndCompound();
    }

    void repr(instance_ptr self, ReprAccumulator& stream, bool isStr, bool isClassNotHeldClass=false);

    typed_python_hash_type hash(instance_ptr left);

    template<class sub_constructor>
    void constructor(instance_ptr self, const sub_constructor& initializer) const {
        for (int64_t k = 0; k < m_members.size(); k++) {
            try {
                initializer(eltPtr(self, k), k);
                setInitializationFlag(self, k);
            } catch(...) {
                for (long k2 = k-1; k2 >= 0; k2--) {
                    m_members[k2].getType()->destroy(eltPtr(self,k2));
                }
                throw;
            }
        }
    }

    void delAttribute(instance_ptr self, int memberIndex) const;

    void setAttribute(instance_ptr self, int memberIndex, instance_ptr other) const;

    //don't default construct classes
    static bool wantsToDefaultConstruct(Type* t) {
        return t->is_default_constructible() && t->getTypeCategory() != TypeCategory::catClass;
    }

    void constructor(instance_ptr self, bool allowEmpty=false);

    void destroy(instance_ptr self);

    void copy_constructor(instance_ptr self, instance_ptr other);

    void assign(instance_ptr self, instance_ptr other);

    bool fieldGuaranteedInitialized(int ix) const {
        return m_members[ix].getIsNonempty();
    }

    bool checkInitializationFlag(instance_ptr self, int memberIndex) const;

    bool isFinal() {
        return m_is_final;
    }

    void setInitializationFlag(instance_ptr self, int memberIndex) const;

    void clearInitializationFlag(instance_ptr self, int memberIndex) const;

    bool getMemberIsNonempty(int index) const {
        return m_members[index].getIsNonempty();
    }

    Type* getMemberType(int index) const {
        return m_members[index].getType();
    }

    const std::string& getMemberName(int index) const {
        return m_members[index].getName();
    }

    bool memberHasDefaultValue(int index) const {
        return m_members[index].getDefaultValue().type()->getTypeCategory() != TypeCategory::catNone;
    }

    const Instance& getMemberDefaultValue(int index) const {
        return m_members[index].getDefaultValue();
    }

    const std::vector<HeldClass*> getBases() const {
        return m_bases;
    }

    const std::vector<MemberDefinition>& getMembers() const {
        return m_members;
    }

    const std::vector<MemberDefinition>& getOwnMembers() const {
        return m_own_members;
    }

    int getMemberIndex(const char* attrName) const {
        auto it = m_membersByName.find(attrName);

        if (it == m_membersByName.end()) {
            return -1;
        }

        return it->second;
    }

    const std::map<std::string, Function*>& getMemberFunctions() const {
        return m_memberFunctions;
    }

    const std::map<std::string, Function*>& getStaticFunctions() const {
        return m_staticFunctions;
    }

    const std::map<std::string, Function*>& getClassMethods() const {
        return m_classMethods;
    }

    const std::map<std::string, PyObject*>& getClassMembers() const {
        return m_classMembers;
    }

    const std::map<std::string, Function*>& getPropertyFunctions() const {
        return m_propertyFunctions;
    }

    const std::map<std::string, Function*>& getOwnMemberFunctions() const {
        return m_own_memberFunctions;
    }

    const std::map<std::string, Function*>& getOwnStaticFunctions() const {
        return m_own_staticFunctions;
    }

    const std::map<std::string, Function*>& getOwnClassMethods() const {
        return m_own_classMethods;
    }

    const std::map<std::string, PyObject*>& getOwnClassMembers() const {
        return m_own_classMembers;
    }

    const std::map<std::string, Function*>& getOwnPropertyFunctions() const {
        return m_own_propertyFunctions;
    }

    const std::vector<size_t>& getOffsets() const {
        return m_byte_offsets;
    }

    int memberNamed(const char* c) const;

    bool hasAnyComparisonOperators() const {
        return m_hasComparisonOperators;
    }

    bool hasGetAttributeMagicMethod() const {
        return m_hasGetAttributeMagicMethod;
    }

    bool hasGetAttrMagicMethod() const {
        return m_hasGetAttrMagicMethod;
    }

    bool hasSetAttrMagicMethod() const {
        return m_hasSetAttrMagicMethod;
    }

    bool hasDelAttrMagicMethod() const {
        return m_hasDelAttrMagicMethod;
    }

    // fill out m_mro according to C3 precedence rules
    void _computeMroSequence() {
        m_mro.push_back(this);

        std::vector<std::vector<HeldClass*> > chains;
        std::vector<std::set<HeldClass*> > chainTails;

        for (HeldClass* base: m_bases) {
            chains.push_back(base->getMro());
        }

        chains.push_back(m_bases);
        chainTails.resize(chains.size());

        for (size_t i = 0; i < chains.size(); i++) {
            for (size_t j = 1; j < chains[i].size(); j++) {
                chainTails[i].insert(chains[i][j]);
            }
        }

        // check if 'h' is in one of the linearization tails. If it is
        // then we can't use it
        auto isValid = [&](HeldClass* h) {
            for (auto& ct: chainTails) {
                if (ct.find(h) != ct.end()) {
                    return false;
                }
            }

            return true;
        };

        auto pickCls = [&]() {
            bool anyPopulated = false;

            for (auto& chain: chains) {
                if (chain.size() && isValid(chain[0])) {
                    return chain[0];
                }

                if (chain.size()) {
                    anyPopulated = true;
                }
            }

            if (anyPopulated) {
                throw std::runtime_error(
                    "Class MRO order for " + name() + " couldn't be determined. No "
                    "valid linearization exists."
                );
            }

            return (HeldClass*)nullptr;
        };

        while (true) {
            HeldClass* hc = pickCls();

            if (!hc) {
                return;
            }

            m_mro.push_back(hc);

            for (size_t i = 0; i < chains.size(); i++) {
                if (chains[i].size() && chains[i][0] == hc) {
                    chains[i].erase(chains[i].begin(), chains[i].begin() + 1);
                    chainTails[i].erase(hc);

                    if (chains[i].size()) {
                        chainTails[i].erase(chains[i][0]);
                    }
                }
            }
        }
    }

    // every HeldClass has a set of functions that are defined within that specific
    // class body, and then they have the fully formed versions of those functions that
    // inherit and overload the versions defined in their base classes.
    // this function builds the versions we actually call from our own and our parent
    // versions.

    // this function can be called multiple times, which happens after the class is
    // deserialized, since we may be rebuilding the function globals.
    void mergeOwnFunctionsIntoInheritanceTree() {
        m_classMembers.clear();
        m_memberFunctions.clear();
        m_classMethods.clear();
        m_propertyFunctions.clear();

        for (HeldClass* base: m_mro) {
            for (auto nameAndObj: base->m_own_classMembers) {
                if (m_classMembers.find(nameAndObj.first) == m_classMembers.end()) {
                    m_classMembers[nameAndObj.first] = incref(nameAndObj.second);
                }
            }

            mergeInto(m_memberFunctions, base->m_own_memberFunctions);
            mergeInto(m_staticFunctions, base->m_own_staticFunctions);
            mergeInto(m_classMethods, base->m_own_classMethods);
            mergeInto(m_propertyFunctions, base->m_own_propertyFunctions);
        }
    }

    void initializeMRO() {
        _computeMroSequence();

        for (size_t i = 0; i < m_mro.size(); i++) {
            m_ancestor_to_mro_index[m_mro[i]] = i;
        }

        if (m_ancestor_to_mro_index.find(this) == m_ancestor_to_mro_index.end() ||
                m_ancestor_to_mro_index.find(this)->second != 0) {
            throw std::runtime_error("Somehow " + m_name + " doesn't have itself as MRO 0");
        }

        // build our own method resolution table directly from our parents.
        mergeOwnFunctionsIntoInheritanceTree();

        std::set<std::string> membersSoFar;

        //only one base class can have members
        for (auto base: m_bases) {
            if (base->m_members.size()) {
                m_members = base->m_members;
            }
        }

        for (auto nameAndType: m_members) {
            membersSoFar.insert(nameAndType.getName());
        }

        for (auto nameAndType: m_own_members) {
            if (membersSoFar.find(nameAndType.getName()) != membersSoFar.end()) {
                throw std::runtime_error("Can't redefine member named " + nameAndType.getName());
            }

            membersSoFar.insert(nameAndType.getName());

            m_members.push_back(nameAndType);
        }

        for (long k = 0; k < m_members.size(); k++) {
            // note that we explicitly leak the string so that the refcount on c_str
            // stays active. I'm sure there's a better way to do this, but types are
            // permanent, so we would never have cleaned this up anyways.
            m_membersByName[(new std::string(m_members[k].getName()))->c_str()] = k;
        }

        for (HeldClass* ancestor: m_mro) {
            mClassDispatchTables.push_back(ClassDispatchTable(this, ancestor));
        }

        for (HeldClass* ancestor: m_mro) {
            ancestor->m_implementors.insert(this);
        }

        updateBytesOfInitBits();

        m_vtable->finalize(
            &mClassDispatchTables[0],
            mBytesOfInitializationBits,
            mClassDispatchTables.size()
        );

        // make sure that, for every interface we can take on, we have slots allocated
        // that the compiler can come along and compile.
        for (HeldClass* ancestor: m_mro) {
            dispatchTableAs(ancestor)->initialize(ancestor->dispatchTableAs(ancestor));
        }

        setMagicMethodExistConstants();
    }

    void updateBytesOfInitBits();

    void setMagicMethodExistConstants() {
        if (m_memberFunctions.find("__eq__") != m_memberFunctions.end()) {
            m_hasComparisonOperators = true;
        }
        if (m_memberFunctions.find("__ne__") != m_memberFunctions.end()) {
            m_hasComparisonOperators = true;
        }
        if (m_memberFunctions.find("__lt__") != m_memberFunctions.end()) {
            m_hasComparisonOperators = true;
        }
        if (m_memberFunctions.find("__gt__") != m_memberFunctions.end()) {
            m_hasComparisonOperators = true;
        }
        if (m_memberFunctions.find("__le__") != m_memberFunctions.end()) {
            m_hasComparisonOperators = true;
        }
        if (m_memberFunctions.find("__ge__") != m_memberFunctions.end()) {
            m_hasComparisonOperators = true;
        }
        if (m_memberFunctions.find("__getattribute__") != m_memberFunctions.end()) {
            m_hasGetAttributeMagicMethod = true;
        }
        if (m_memberFunctions.find("__delattr__") != m_memberFunctions.end()) {
            m_hasDelAttrMagicMethod = true;
        }
        if (m_memberFunctions.find("__setattr__") != m_memberFunctions.end()) {
            m_hasSetAttrMagicMethod = true;
        }
        if (m_memberFunctions.find("__getattr__") != m_memberFunctions.end()) {
            m_hasGetAttrMagicMethod = true;
        }
    }

    size_t allocateMethodDispatch(std::string funcName, function_call_signature_type signature) {
        size_t result = dispatchTableAs(this)->allocateMethodDispatch(funcName, signature);

        // make sure we add this dispatch to every child that implements us as an interface
        for (HeldClass* child: m_implementors) {
            if (result != child->dispatchTableAs(this)->allocateMethodDispatch(funcName, signature)) {
                throw std::runtime_error("Corrupted Dispatch Tables!");
            }
        }

        return result;
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
        for (auto t: m_mro) {
            if (Type::typesEquivalent(t, otherType)) {
                return true;
            }
        }

        return false;
    }

    VTable* getVTable() const {
        return m_vtable;
    }

    const std::vector<HeldClass*>& getMro() const {
        return m_mro;
    }

    int64_t getMroIndex(HeldClass* ancestor) const {
        auto it = m_ancestor_to_mro_index.find(ancestor);

        if (it != m_ancestor_to_mro_index.end()) {
            return it->second;
        }

        // check if an 'equivalent' class is floating around in there.
        for (long k = 0; k < m_mro.size(); k++) {
            if (Type::typesEquivalent(m_mro[k], ancestor)) {
                return k;
            }
        }

        return -1;
    }

    ClassDispatchTable* dispatchTableAs(HeldClass* interface) {
        int64_t offset = getMroIndex(interface);
        if (offset < 0) {
            throw std::runtime_error("Interface is not an ancestor.");
        }

        return &mClassDispatchTables[offset];
    }

    // get a BoundMethod type object representing this method on instances
    // of this class. If 'forHeld', then 'self' will be bound as a RefTo(Held(T))
    // instead of a T
    BoundMethod* getMemberFunctionMethodType(const char* attr, bool forHeld);

private:
    std::vector<size_t> m_byte_offsets;

    std::vector<HeldClass*> m_bases;

    size_t mBytesOfInitializationBits;

    // if final, we can't subclass this class
    bool m_is_final;

    // equivalent to python's method resolution order, so we can
    // search for methods at runtime.
    std::vector<HeldClass*> m_mro;

    std::set<HeldClass*> m_implementors; //all classes that implement this interface

    std::unordered_map<HeldClass*, int> m_ancestor_to_mro_index;

    VTable* m_vtable;

    Class* m_classType; //the non-held version of this class

    RefTo* m_refToType; //a cache for the ref-to-type

    std::vector<ClassDispatchTable> mClassDispatchTables;

    //the members we
    std::vector<MemberDefinition> m_members;

    std::map<std::string, Function*> m_memberFunctions;

    std::map<std::string, Function*> m_staticFunctions;

    std::map<std::string, Function*> m_propertyFunctions;

    std::map<std::string, Function*> m_classMethods;

    std::map<std::string, PyObject*> m_classMembers;

    // the original members we were provided with
    std::vector<MemberDefinition> m_own_members;

    std::map<std::string, Function*> m_own_memberFunctions;

    std::map<std::string, Function*> m_own_staticFunctions;

    std::map<std::string, Function*> m_own_propertyFunctions;

    std::map<std::string, PyObject*> m_own_classMembers;

    std::map<std::string, Function*> m_own_classMethods;

    std::unordered_map<const char*, size_t, HashConstCharPtr, ConstCharPtrsAreEqual> m_membersByName;

    std::unordered_map<const char*, BoundMethod*, HashConstCharPtr, ConstCharPtrsAreEqual> m_memberFunctionMethodTypes[2];

    bool m_hasComparisonOperators;
    bool m_hasGetAttributeMagicMethod;
    bool m_hasGetAttrMagicMethod;
    bool m_hasSetAttrMagicMethod;
    bool m_hasDelAttrMagicMethod;
};
