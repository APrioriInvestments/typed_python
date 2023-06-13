/******************************************************************************
   Copyright 2017-2023 typed_python Authors

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

#include "Memory.hpp"
#include <Python.h>
#include <string>
#include <vector>
#include <algorithm>
#include <functional>
#include <unordered_set>
#include <mutex>
#include <set>
#include <map>
#include <tuple>
#include <utility>
#include <atomic>
#include <iostream>
#include "ReprAccumulator.hpp"
#include "SerializationContext.hpp"
#include "HashAccumulator.hpp"
#include "ShaHash.hpp"
#include "TypeOrPyobj.hpp"
#include "util.hpp"
#include "MutuallyRecursiveTypeGroup.hpp"
#include "Slab.hpp"
#include "DeepcopyContext.hpp"
#include "TypeStack.hpp"

class SerializationBuffer;
class DeserializationBuffer;

class Type;
class NoneType;
class Bool;
class UInt8;
class UInt16;
class UInt32;
class UInt64;
class Int8;
class Int16;
class Int32;
class Int64;
class Float32;
class Float64;
class StringType;
class BytesType;
class OneOfType;
class Value;
class TupleOfType;
class PointerTo;
class RefTo;
class ListOfType;
class NamedTuple;
class Tuple;
class DictType;
class ConstDictType;
class Alternative;
class ConcreteAlternative;
class AlternativeMatcher;
class PythonObjectOfType;
class SubclassOfType;
class Class;
class HeldClass;
class Function;
class BoundMethod;
class Forward;
class EmbeddedMessageType;
class SetType;
class TypedCellType;
class PyCellType;

typedef uint8_t* instance_ptr;

typedef void (*compiled_code_entrypoint)(instance_ptr, instance_ptr*);

void updateTypeRepForType(Type* t, PyTypeObject* pyType);

PyObject* getOrSetTypeResolver(PyObject* module = nullptr, PyObject* args = nullptr);

enum class Maybe {
    True,
    False,
    Maybe
};


class Type {
public:
    //the indices are part of the binary serialization format (except for 'Forward'),
    //so don't change them.
    enum TypeCategory {
        catNone = 0,
        catBool = 1,
        catUInt8 = 2,
        catUInt16 = 3,
        catUInt32 = 4,
        catUInt64 = 5,
        catInt8 = 6,
        catInt16 = 7,
        catInt32 = 8,
        catInt64 = 9,
        catString = 10,
        catBytes = 11,
        catFloat32 = 12,
        catFloat64 = 13,
        catValue = 14,
        catOneOf = 15,
        catTupleOf = 16,
        catPointerTo = 17,
        catListOf = 18,
        catNamedTuple = 19,
        catTuple = 20,
        catDict = 21,
        catConstDict = 22,
        catAlternative = 23,
        catConcreteAlternative = 24, //concrete Alternative subclass
        catPythonObjectOfType = 26, //a python object that matches 'isinstance' on a particular type
        catBoundMethod = 27,
        catAlternativeMatcher = 37,
        catClass = 28,
        catHeldClass = 29,
        catFunction = 30,
        catForward = 31,
        catEmbeddedMessage = 32,
        catSet = 33,
        catTypedCell = 34,
        catPyCell = 35,
        catRefTo = 36,
        catSubclassOf = 38, // holds a Type* which is a subclass of a given set of Type instances
    };

    bool isTuple() const {
        return m_typeCategory == catTuple;
    }

    bool isString() const {
        return m_typeCategory == catString;
    }

    bool isBytes() const {
        return m_typeCategory == catBytes;
    }

    bool isListOf() const {
        return m_typeCategory == catListOf;
    }

    bool isTupleOrListOf() const {
        return m_typeCategory == catListOf || m_typeCategory == catTupleOf;
    }

    bool isTupleOf() const {
        return m_typeCategory == catTupleOf;
    }

    bool isDict() const {
        return m_typeCategory == catDict;
    }

    bool isSet() const {
        return m_typeCategory == catSet;
    }

    bool isPointerTo() const {
        return m_typeCategory == catPointerTo;
    }

    bool isConstDict() const {
        return m_typeCategory == catConstDict;
    }

    bool isAlternativeMatcher() const {
        return m_typeCategory == catAlternativeMatcher;
    }

    bool isBoundMethod() const {
        return m_typeCategory == catBoundMethod;
    }

    bool isNamedTuple() const {
        return m_typeCategory == catNamedTuple;
    }

    bool isNone() const {
        return m_typeCategory == catNone;
    }

    bool isFunction() const {
        return m_typeCategory == catFunction;
    }

    bool isPythonObjectOfType() const {
        return m_typeCategory == catPythonObjectOfType;
    }

    bool isPyCell() const {
        return m_typeCategory == catPyCell;
    }

    bool isClass() const {
        return m_typeCategory == catClass;
    }

    bool isHeldClass() const {
        return m_typeCategory == catHeldClass;
    }

    bool isAlternative() const {
        return m_typeCategory == catAlternative;
    }

    bool isConcreteAlternative() const {
        return m_typeCategory == catConcreteAlternative;
    }

    bool isTypedCell() const {
        return m_typeCategory == catTypedCell;
    }

    bool isForward() const {
        return m_typeCategory == catForward;
    }

    bool isOneOf() const {
        return m_typeCategory == catOneOf;
    }

    bool isSubclassOf() const {
        return m_typeCategory == catSubclassOf;
    }

    bool isValue() const {
        return m_typeCategory == catValue;
    }

    bool isBool() const {
        return (
            m_typeCategory == catBool
        );
    }

    bool isRegister() const {
        return (
            m_typeCategory == catBool
            || m_typeCategory == catUInt8
            || m_typeCategory == catUInt16
            || m_typeCategory == catUInt32
            || m_typeCategory == catUInt64
            || m_typeCategory == catInt8
            || m_typeCategory == catInt16
            || m_typeCategory == catInt32
            || m_typeCategory == catInt64
            || m_typeCategory == catFloat32
            || m_typeCategory == catFloat64
        );
    }

    bool isInteger() const {
        return (
               m_typeCategory == catUInt8
            || m_typeCategory == catUInt16
            || m_typeCategory == catUInt32
            || m_typeCategory == catUInt64
            || m_typeCategory == catInt8
            || m_typeCategory == catInt16
            || m_typeCategory == catInt32
            || m_typeCategory == catInt64
        );
    }

    bool isRecursive() {
        return getRecursiveTypeGroupMembers().size() != 1;
    }

    bool isRefTo() const {
        return m_typeCategory == catRefTo;
    }

    // is it legal (and will it always succeed) to cast values from
    // fromType to toType?
    static bool isValidUpcastType(Type* fromType, Type* toType);

    virtual ~Type() {
        throw std::runtime_error("Types should never get deleted.");
    }

    TypeCategory getTypeCategory() const {
        return m_typeCategory;
    }

    std::string getTypeCategoryString() const {
        return categoryToString(m_typeCategory);
    }

    bool isComposite() const {
        return (
            m_typeCategory == catTuple ||
            m_typeCategory == catNamedTuple
            );
    }

    const std::string& name(bool stripQualname=false) const {
        if (stripQualname) {
            if (!m_stripped_name.size()) {
                m_stripped_name = qualname_to_name(m_name);
            }

            return m_stripped_name;
        }
        return m_name;
    }

    std::string moduleName() {
        return this->check([&](auto& subtype) {
            return subtype.moduleNameConcrete();
        });
    }

    std::string moduleNameConcrete() {
        return "builtins";
    }

    std::string nameWithModule() {
        return this->check([&](auto& subtype) {
            return subtype.nameWithModuleConcrete();
        });
    }

    std::string nameWithModuleConcrete() {
        return name();
    }

    const char* doc() {
        return this->check([&](auto& subtype) {
            return subtype.docConcrete();
        });
    }

    const char* docConcrete() {
        throw std::runtime_error(
            "No docstring provided for " + name() + " of category " + getTypeCategoryString()
        );
    }

    size_t bytecount() const {
        return m_size;
    }

    Type* pickConcreteSubclass(instance_ptr data) {
        assertForwardsResolvedSufficientlyToInstantiate();

        return this->check([&](auto& subtype) {
            return subtype.pickConcreteSubclassConcrete(data);
        });
    }

    Type* pickConcreteSubclassConcrete(instance_ptr data) {
        return this;
    }

    void repr(instance_ptr self, ReprAccumulator& out, bool isStr);

    std::string toString(instance_ptr self, bool isStr) {
        std::ostringstream s;

        {
            ReprAccumulator acc(s);
            repr(self, acc, isStr);
        }

        return s.str();
    }

    /* compare two types as closely as possible to how python would.

    If 'suppressExceptions', then don't generate exceptions when python doesn't have an ordering.
    Instead, sort their type names, and compare pointers for objects that have the same type. This
    allows us to use this function to produce a strict ordering.
    */
    bool cmp(instance_ptr left, instance_ptr right, int pyComparisonOp, bool suppressExceptions = false);

    typed_python_hash_type hash(instance_ptr left);

    void deepcopy(
        instance_ptr dest,
        instance_ptr src,
        DeepcopyContext& context
    );

    void deepcopyConcrete(
        instance_ptr dest,
        instance_ptr src,
        DeepcopyContext& context
    ) {
        throw std::runtime_error(
            "deepcopyConcrete not implemented for " + name() +
            " of category " + categoryToString(getTypeCategory())
        );
    }

    // compute the allocated bytecount of an object, including python objects (but not walking
    // into types, modules, or functions). Don't include storage for 'instance' itself - just storage
    // that's allocated on the heap by this object. If 'outSlabs' is not the nullpointer, then if we
    // hit an allocation that's mapped to a slab, just mark the slab and return.
    size_t deepBytecount(instance_ptr instance, std::unordered_set<void*>& alreadyVisited, std::set<Slab*>* outSlabs) {
        assertForwardsResolvedSufficientlyToInstantiate();

        return this->check([&](auto& subtype) {
            return subtype.deepBytecountConcrete(instance, alreadyVisited, outSlabs);
        });
    }

    size_t deepBytecountConcrete(
        instance_ptr instance,
        std::unordered_set<void*>& alreadyVisited,
        std::set<Slab*>* outSlabs
    ) {
        throw std::runtime_error(
            "deepBytecountConcrete not implemented for " + name() +
            " of category " + categoryToString(getTypeCategory())
        );
    }

    template<class buf_t>
    void serialize(instance_ptr left, buf_t& buffer, size_t fieldNumber) {
        assertForwardsResolvedSufficientlyToInstantiate();

        return this->check([&](auto& subtype) {
            return subtype.serialize(left, buffer, fieldNumber);
        });
    }

    template<class buf_t>
    void serializeMulti(instance_ptr left, size_t count, size_t stride, buf_t& buffer, size_t fieldNumber) {
        assertForwardsResolvedSufficientlyToInstantiate();

        return this->check([&](auto& subtype) {
            return subtype.serializeMultiConcrete(left, count, stride, buffer, fieldNumber);
        });
    }

    template<class buf_t>
    void serializeMultiConcrete(instance_ptr left, size_t count, size_t stride, buf_t& buffer, size_t fieldNumber) {
        for (long k = 0; k < count; k++) {
            serialize(left + stride * k, buffer, fieldNumber);
        }
    }

    template<class buf_t>
    void deserialize(instance_ptr left, buf_t& buffer, size_t wireType) {
        assertForwardsResolvedSufficientlyToInstantiate();

        return this->check([&](auto& subtype) {
            return subtype.deserialize(left, buffer, wireType);
        });
    }

    // swap the contents of 'left' and 'right'.  The values should be valid.
    void swap(instance_ptr left, instance_ptr right);

    // initialize 'dest' from 'src', and destroy 'src'.
    void move(instance_ptr dest, instance_ptr src);

    static char byteCompare(uint8_t* l, uint8_t* r, size_t count);

    static std::string categoryToString(TypeCategory category) {
        if (category == Type::TypeCategory::catNone) { return "None"; }
        if (category == Type::TypeCategory::catBool) { return "Bool"; }
        if (category == Type::TypeCategory::catUInt8) { return "UInt8"; }
        if (category == Type::TypeCategory::catUInt16) { return "UInt16"; }
        if (category == Type::TypeCategory::catUInt32) { return "UInt32"; }
        if (category == Type::TypeCategory::catUInt64) { return "UInt64"; }
        if (category == Type::TypeCategory::catInt8) { return "Int8"; }
        if (category == Type::TypeCategory::catInt16) { return "Int16"; }
        if (category == Type::TypeCategory::catInt32) { return "Int32"; }
        if (category == Type::TypeCategory::catInt64) { return "int"; }
        if (category == Type::TypeCategory::catString) { return "str"; }
        if (category == Type::TypeCategory::catBytes) { return "bytes"; }
        if (category == Type::TypeCategory::catFloat32) { return "Float32"; }
        if (category == Type::TypeCategory::catFloat64) { return "float"; }
        if (category == Type::TypeCategory::catValue) { return "Value"; }
        if (category == Type::TypeCategory::catOneOf) { return "OneOf"; }
        if (category == Type::TypeCategory::catTupleOf) { return "TupleOf"; }
        if (category == Type::TypeCategory::catPointerTo) { return "PointerTo"; }
        if (category == Type::TypeCategory::catRefTo) { return "RefTo"; }
        if (category == Type::TypeCategory::catListOf) { return "ListOf"; }
        if (category == Type::TypeCategory::catNamedTuple) { return "NamedTuple"; }
        if (category == Type::TypeCategory::catTuple) { return "Tuple"; }
        if (category == Type::TypeCategory::catSet) { return "Set"; }
        if (category == Type::TypeCategory::catDict) { return "Dict"; }
        if (category == Type::TypeCategory::catConstDict) { return "ConstDict"; }
        if (category == Type::TypeCategory::catAlternative) { return "Alternative"; }
        if (category == Type::TypeCategory::catConcreteAlternative) { return "ConcreteAlternative"; }
        if (category == Type::TypeCategory::catBoundMethod) { return "BoundMethod"; }
        if (category == Type::TypeCategory::catAlternativeMatcher) { return "AlternativeMatcher"; }
        if (category == Type::TypeCategory::catClass) { return "Class"; }
        if (category == Type::TypeCategory::catHeldClass) { return "HeldClass"; }
        if (category == Type::TypeCategory::catFunction) { return "Function"; }
        if (category == Type::TypeCategory::catForward) { return "Forward"; }
        if (category == Type::TypeCategory::catEmbeddedMessage) { return "EmbeddedMessage"; }
        if (category == Type::TypeCategory::catPythonObjectOfType) { return "PythonObjectOfType"; }
        if (category == Type::TypeCategory::catPyCell) { return "PyCell"; }
        if (category == Type::TypeCategory::catTypedCell) { return "TypedCell"; }
        if (category == Type::TypeCategory::catSubclassOf) { return "SubclassOf"; }

        return "Unknown";
    }

    template<class T>
    auto check(const T& f) -> decltype(f(*this)) {
        switch (m_typeCategory) {
            case catNone:
                return f(*(NoneType*)this);
            case catBool:
                return f(*(Bool*)this);
            case catUInt8:
                return f(*(UInt8*)this);
            case catUInt16:
                return f(*(UInt16*)this);
            case catUInt32:
                return f(*(UInt32*)this);
            case catUInt64:
                return f(*(UInt64*)this);
            case catInt8:
                return f(*(Int8*)this);
            case catInt16:
                return f(*(Int16*)this);
            case catInt32:
                return f(*(Int32*)this);
            case catInt64:
                return f(*(Int64*)this);
            case catString:
                return f(*(StringType*)this);
            case catBytes:
                return f(*(BytesType*)this);
            case catFloat32:
                return f(*(Float32*)this);
            case catFloat64:
                return f(*(Float64*)this);
            case catValue:
                return f(*(Value*)this);
            case catOneOf:
                return f(*(OneOfType*)this);
            case catTupleOf:
                return f(*(TupleOfType*)this);
            case catPointerTo:
                return f(*(PointerTo*)this);
            case catRefTo:
                return f(*(RefTo*)this);
            case catListOf:
                return f(*(ListOfType*)this);
            case catNamedTuple:
                return f(*(NamedTuple*)this);
            case catTuple:
                return f(*(Tuple*)this);
            case catSet:
                return f(*(SetType*)this);
            case catDict:
                return f(*(DictType*)this);
            case catConstDict:
                return f(*(ConstDictType*)this);
            case catAlternative:
                return f(*(Alternative*)this);
            case catConcreteAlternative:
                return f(*(ConcreteAlternative*)this);
            case catPythonObjectOfType:
                return f(*(PythonObjectOfType*)this);
            case catClass:
                return f(*(Class*)this);
            case catHeldClass:
                return f(*(HeldClass*)this);
            case catFunction:
                return f(*(Function*)this);
            case catBoundMethod:
                return f(*(BoundMethod*)this);
            case catAlternativeMatcher:
                return f(*(AlternativeMatcher*)this);
            case catForward:
                return f(*(Forward*)this);
            case catEmbeddedMessage:
                return f(*(EmbeddedMessageType*)this);
            case catPyCell:
                return f(*(PyCellType*)this);
            case catTypedCell:
                return f(*(TypedCellType*)this);
            case catSubclassOf:
                return f(*(SubclassOfType*)this);
            default:
                throw std::runtime_error("Invalid type found");
        }
    }

    template<class concrete_type, class visitor_type>
    static void adaptTypeVisitor(const visitor_type& vis, concrete_type*& ioTypePtr) {
        Type* t = ioTypePtr;
        vis(t);
        if (t != (Type*)ioTypePtr) {
            if (t->getTypeCategory() != ioTypePtr->getTypeCategory() && !ioTypePtr->isForward()) {
                throw std::runtime_error(
                    "Somehow, visitor changed the type category from "
                    + ioTypePtr->getTypeCategoryString() + " to " + t->getTypeCategoryString()
                );
            }

            ioTypePtr = (concrete_type*)t;
        }
    }

    template<class T>
    static void updateTypeRefFromGroupMap(T*& toUpdate, const std::map<Type*, Type*>& groupMap) {
        auto it = groupMap.find(toUpdate);

        if (it != groupMap.end()) {
            toUpdate = (T*)it->second;
        }
    }

    Type* getBaseType() const {
        return m_base;
    }

    // are these types equivalent up to identity? This should be
    // preferred over t1 == t2 for comparing types in cases where
    // we don't need _exact_ identity equiality, because there are
    // some pathways where the identical type object can get created
    // (usually through deserialization)
    static bool typesEquivalent(Type* t1, Type* t2) {
        return t1 == t2;
    }

    //this checks _strict_ subclass. X is not a subclass of itself.
    //note we are NOT holding the GIL.
    bool isSubclassOf(Type* otherType) {
        if (Type::typesEquivalent(otherType, this)) {
            return false;
        }

        if (Type::typesEquivalent(otherType, m_base)) {
            return true;
        }

        return this->check([&](auto& subtype) {
            return subtype.isSubclassOfConcrete(otherType);
        });
    }

    bool isSubclassOfConcrete(Type* otherType) {
        return false;
    }

    // walk the MRO type hierarchy - 'v' gets called with 'this'
    // and all base classes in the order of method resolution
    template<class visitor_type>
    void visitMRO(const visitor_type& v) {
        this->check([&](auto& subtype) {
            subtype.visitMROConcrete(v);
        });
    }

    template<class visitor_type>
    void visitMROConcrete(const visitor_type& v) {
        v(this);
    }

    // if isPOD (Plain Old Data), then there is no constructor / destructor semantic.
    // Instances of the type can just be bitcopied blindly without leaking any memory.
    bool isPOD() {
        return this->check([&](auto& subtype) {
            return subtype.isPODConcrete();
        });
    }

    bool isPODConcrete() {
        return false;
    }

    void assertForwardsResolved() const {
        if (m_is_forward_defined) {
            throw std::logic_error("Type " + m_name + " is a forward");
        }
    }

    void assertForwardsResolvedSufficientlyToInstantiate() {
        this->check([&](auto& subtype) {
            subtype.assertForwardsResolvedSufficientlyToInstantiateConcrete();
        });
    }

    void assertForwardsResolvedSufficientlyToInstantiateConcrete() const {
        if (m_is_forward_defined) {
            throw std::logic_error(
                "Type " + m_name + " of cat "
                + this->getTypeCategoryString() + " is a forward"
            );
        }
    }

    template<class visitor_type>
    void _visitReferencedTypes(const visitor_type& v) {}

    template<class visitor_type>
    void _visitContainedTypes(const visitor_type& v) {}

    // visit the internals of the type in a stable order. Two types that are different
    // should have different visit orders - all data must be hashed into this.
    template<class visitor_type>
    void _visitCompilerVisibleInternals(const visitor_type& v) {
        throw std::runtime_error(
            "No _visitCompilerVisibleInternals implemented for " + name()
        );
    }

    // call subtype.constructor
    void constructor(instance_ptr self);

    template<class ptr_func>
    void constructor(int64_t count, const ptr_func& ptrToChild) {
        assertForwardsResolvedSufficientlyToInstantiate();

        this->check([&](auto& subtype) {
            for (long k = 0; k < count; k++) {
                subtype.constructor(ptrToChild(k));
            }
        });
    }

    // call subtype.destroy
    void destroy(instance_ptr self);

    template<class ptr_func>
    void destroy(int64_t count, const ptr_func& ptrToChild) {
        this->check([&](auto& subtype) {
            for (long k = 0; k < count; k++) {
                subtype.destroy(ptrToChild(k));
            }
        });
    }

    template<class visitor_type>
    void visitContainedTypes(const visitor_type& v) {
        this->check([&](auto& subtype) {
            subtype._visitContainedTypes(v);
        });
    }

    template<class visitor_type>
    void visitReferencedTypes(const visitor_type& v) {
        this->check([&](auto& subtype) {
            subtype._visitReferencedTypes(v);
        });
    }

    template<class visitor_type>
    void visitCompilerVisibleInternals(const visitor_type& v) {
        this->check([&](auto& subtype) {
            subtype._visitCompilerVisibleInternals(v);
        });
    }

    // call subtype.copy_constructor
    void copy_constructor(instance_ptr self, instance_ptr other);

    template<class ptr_func_dest, class ptr_func_src>
    void copy_constructor(int64_t count, const ptr_func_dest& ptrToTarget, const ptr_func_src& ptrToSrc) {
        assertForwardsResolvedSufficientlyToInstantiate();

        this->check([&](auto& subtype) {
            for (long k = 0; k < count; k++) {
                subtype.copy_constructor(ptrToTarget(k), ptrToSrc(k));
            }
        });
    }

    // call subtype.assign
    void assign(instance_ptr self, instance_ptr other);

    PyTypeObject* getTypeRep() const {
        return mTypeRep;
    }

    void setTypeRep(PyTypeObject* o) {
        mTypeRep = o;
    }

    bool is_default_constructible() const {
        return m_is_default_constructible;
    }

    // are we guaranteed we can convert to this other type at the 'Signature' level
    bool canConvertToTrivially(Type* otherType);

    /*****
        determine the full group of types that this type is a part of as far
        as our compiler is concerned. For instance, two functions 'f' and 'g'
        that are mutually recursive at the module level will see each other
        in the compiler, even if their individual types just know that they
        see 'f' and 'g' through their globals.

        This involves tracing function, class, and module references
    *****/
    MutuallyRecursiveTypeGroup* getRecursiveTypeGroup() {
        if (!mTypeGroup) {
            MutuallyRecursiveTypeGroup::ensureRecursiveTypeGroup(
                this,
                VisibilityType::Identity
            );

            if (!mTypeGroup) {
                throw std::runtime_error("Somehow, this type is not in a type group!");
            }
        }

        return mTypeGroup;
    }

    bool hasTypeGroup() const {
        return mTypeGroup != nullptr;
    }

    const std::map<int32_t, TypeOrPyobj>& getRecursiveTypeGroupMembers() {
        return getRecursiveTypeGroup()->getIndexToObject();
    }

    /*****
        compute our identity hash, which is a sha hash that uniquely identifies
        this type and all the code it can reach. This hash should be precise enough
        that any binaries produced that work with two implementions with the same hash
        will produce the same result.

        Practically, we have two modes: for nonrecursive types, we simply sha-hash
        the hashes of the type parameters we depend on, along with some constants (
        e.g. the typeCategory) to uniquely id the type.

        For recursive types, we have to compute a 'MutuallyRecursiveTypeGroup', which
        is the set of types and python objects that are mutually visible to each other
        at the compiler level.  We then hash all the types in the group, but are careful
        to simply hash the type's index within the group when we're hashing a parameter
        type that's in our group. This gives us a unique signature for the group. The
        final hash is then that hash plus our id within the group.
    ******/
    ShaHash identityHash() {
        // if we've never initialized our hash
        if (mIdentityHash == ShaHash()) {
            MutuallyRecursiveTypeGroup::ensureRecursiveTypeGroup(
                this,
                VisibilityType::Identity
            );

            mIdentityHash = MutuallyRecursiveTypeGroup::shaHash(
                this,
                VisibilityType::Identity
            );
        }

        return mIdentityHash;
    }

    void setRecursiveTypeGroup(MutuallyRecursiveTypeGroup* group, int32_t index) {
        if (group->visibilityType() != VisibilityType::Identity) {
            throw std::runtime_error("A Type's MutuallyRecursiveTypeGroup needs to be an Identity group");
        }
        mTypeGroup = group;
        mRecursiveTypeGroupIndex = index;
    }

    int64_t getRecursiveTypeGroupIndex() const {
        return mRecursiveTypeGroupIndex;
    }

    bool isForwardDefined() const {
        return m_is_forward_defined;
    }

    bool isResolved() const {
        if (!m_is_forward_defined) {
            return true;
        }

        return m_forward_resolves_to != nullptr;
    }

    bool looksResolvable(bool unambiguously);

    Type* forwardResolvesTo() {
        if (!m_is_forward_defined) {
            return this;
        }

        if (!m_forward_resolves_to) {
            attemptToResolve();
        }

        return m_forward_resolves_to;
    }

protected:
    Type(TypeCategory in_typeCategory) :
            m_typeCategory(in_typeCategory),
            m_size(0),
            m_is_default_constructible(false),
            m_name("Undefined"),
            mTypeRep(nullptr),
            m_base(nullptr),
            mTypeGroup(nullptr),
            mRecursiveTypeGroupIndex(-1),
            m_is_forward_defined(false),
            m_forward_resolves_to(nullptr),
            m_is_redundant(false)
        {}

    TypeCategory m_typeCategory;

    size_t m_size;

    bool m_is_default_constructible;

    std::string m_recursive_name;

    std::string m_name;

    mutable std::string m_stripped_name;

    PyTypeObject* mTypeRep;

    Type* m_base;

    enum BinaryCompatibilityCategory { Incompatible, Checking, Compatible };

    // a sha-hash that uniquely identifies this type. If this value is
    // the same for two types, then they should be indistinguishable except
    // for pointers values.
    ShaHash mIdentityHash;

    // a pointer to our type group
    MutuallyRecursiveTypeGroup* mTypeGroup;

    int32_t mRecursiveTypeGroupIndex;

    // is there a Forward reachable in our object graph? This is a permanent feature of
    // the type object
    bool m_is_forward_defined;

    // if we are a Forward, what do we resolve to? This will be null if we have not
    // fully resolved ourselves and will, for valid types, become not-null pointing to
    // the actual type we resolve to.
    Type* m_forward_resolves_to;

    // have we been made 'redundant'?
    bool m_is_redundant;

    // try to resolve this forward type. If we can't, we'll throw an exception. On exit,
    // we will have thrown, or m_forward_resolves_to will be populated.
    void attemptToResolve();

    // this is a fully-resolved type that we created but aren't using.
    // it should just leak and never be used again.
    void markRedundant() {
        m_is_redundant = true;
    }

    bool isRedundant() {
        return m_is_redundant;
    }

    void initializeFromConcrete(Type* forwardDefinitionOfSelf) {
        throw std::runtime_error("Type " + name() + " didn't define initializeFromConcrete");
    }

    // update internal Type pointers to point into a new group
    void updateInternalTypePointers(const std::map<Type*, Type*>& groupMap) {
        this->check([&](auto& subtype) {
            return subtype.updateInternalTypePointersConcrete(groupMap);
        });
    }

    void updateInternalTypePointersConcrete(
        const std::map<Type*, Type*>& groupMap
    ) {
        throw std::runtime_error("Type " + name() + " didn't define updateInternalTypePointersConcrete");
    }

    // produce a copy of ourself
    Type* cloneForForwardResolution() {
        return this->check([&](auto& subtype) {
            return subtype.cloneForForwardResolutionConcrete();
        });
    }

    Type* cloneForForwardResolutionConcrete() {
        throw std::runtime_error("Type " + name() + " didn't define cloneForForwardResolutionConcrete");
    }

    void postInitializeConcrete() {
        throw std::runtime_error(
            "Type " + name() + " of cat " + getTypeCategoryString() + " didn't implement postInitializeConcrete"
        );
    }

    void finalizeTypeConcrete() {}

    std::string computeRecursiveNameConcrete(TypeStack& typeStack) {
        return m_name;
    }

    static std::map<ShaHash, Type*> mInternalizedIdentityHashToType;

public:
    void recomputeName() {
        TypeStack stack;
        m_name = computeRecursiveName(stack);
        m_stripped_name = "";
    }

    void internalize();

    // initialize ourself as a copy of 'forwardDefinitionOfSelf' where none of the types
    // will have a forward definition reference. The instance must have been constructed
    // using 'cloneForForwardResolution'. If the recursiveNameOverride is not None, then
    // use that name for the type instead of the computed name.
    void initializeFrom(Type* forwardDefinitionOfSelf) {
        this->check([&](auto& subtype) {
            return subtype.initializeFromConcrete(forwardDefinitionOfSelf);
        });
    }

    // finish initializing the type assuming no forward types are reachable
    bool postInitialize() {
        size_t oldSize = m_size;
        bool oldIsDefaultInit = m_is_default_constructible;

        this->check([&](auto& subtype) {
            subtype.postInitializeConcrete();
        });

        // return whether we changed
        return m_size != oldSize || m_is_default_constructible != oldIsDefaultInit;
    }

    // update any caches in the type after we've walked over it
    void finalizeType() {
        this->check([&](auto& subtype) {
            subtype.finalizeTypeConcrete();
        });
    }

    std::string computeRecursiveName(TypeStack& stack) {
        long index = stack.indexOf(this);

        if (index != -1) {
            return "^" + format(index);
        }

        PushTypeStack addSelf(stack, this);

        return this->check([&](auto& subtype) {
            return subtype.computeRecursiveNameConcrete(stack);
        });
    }
};
