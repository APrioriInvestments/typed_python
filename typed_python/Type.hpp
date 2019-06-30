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

#include <Python.h>
#include <string>
#include <vector>
#include <algorithm>
#include <functional>
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
#include "util.hpp"

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
class ListOfType;
class NamedTuple;
class Tuple;
class DictType;
class ConstDictType;
class Alternative;
class ConcreteAlternative;
class PythonSubclass;
class PythonObjectOfType;
class Class;
class HeldClass;
class Function;
class BoundMethod;
class Forward;
class EmbeddedMessageType;

typedef uint8_t* instance_ptr;

typedef void (*compiled_code_entrypoint)(instance_ptr, instance_ptr*);

void updateTypeRepForType(Type* t, PyTypeObject* pyType);

PyObject* getOrSetTypeResolver(PyObject* module = nullptr, PyObject* args = nullptr);

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
        catPythonSubclass = 25, //subclass of a nativepython type
        catPythonObjectOfType = 26, //a python object that matches 'isinstance' on a particular type
        catBoundMethod = 27,
        catClass = 28,
        catHeldClass = 29,
        catFunction = 30,
        catForward = 31,
        catEmbeddedMessage = 32
    };

    TypeCategory getTypeCategory() const {
        return m_typeCategory;
    }

    bool isComposite() const {
        return (
            m_typeCategory == catTuple ||
            m_typeCategory == catNamedTuple
            );
    }

    const std::string& name() const {
        return m_name;
    }

    size_t bytecount() const {
        return m_size;
    }

    Type* pickConcreteSubclass(instance_ptr data) {
        assertForwardsResolved();

        return this->check([&](auto& subtype) {
            return subtype.pickConcreteSubclassConcrete(data);
        });
    }

    Type* pickConcreteSubclassConcrete(instance_ptr data) {
        return this;
    }

    void repr(instance_ptr self, ReprAccumulator& out);

    bool cmp(instance_ptr left, instance_ptr right, int pyComparisonOp);

    typed_python_hash_type hash64(instance_ptr left);

    template<class buf_t>
    void serialize(instance_ptr left, buf_t& buffer, size_t fieldNumber) {
        assertForwardsResolved();

        return this->check([&](auto& subtype) {
            return subtype.serialize(left, buffer, fieldNumber);
        });
    }

    template<class buf_t>
    void deserialize(instance_ptr left, buf_t& buffer, size_t wireType) {
        assertForwardsResolved();

        return this->check([&](auto& subtype) {
            return subtype.deserialize(left, buffer, wireType);
        });
    }

    // swap the contents of 'left' and 'right'.  The values should be valid.
    void swap(instance_ptr left, instance_ptr right);

    // initialize 'dest' from 'src', and destroy 'src'.
    void move(instance_ptr dest, instance_ptr src);

    static char byteCompare(uint8_t* l, uint8_t* r, size_t count);

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
            case catListOf:
                return f(*(ListOfType*)this);
            case catNamedTuple:
                return f(*(NamedTuple*)this);
            case catTuple:
                return f(*(Tuple*)this);
            case catDict:
                return f(*(DictType*)this);
            case catConstDict:
                return f(*(ConstDictType*)this);
            case catAlternative:
                return f(*(Alternative*)this);
            case catConcreteAlternative:
                return f(*(ConcreteAlternative*)this);
            case catPythonSubclass:
                return f(*(PythonSubclass*)this);
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
            case catForward:
                return f(*(Forward*)this);
            case catEmbeddedMessage:
                return f(*(EmbeddedMessageType*)this);
            default:
                throw std::runtime_error("Invalid type found");
        }
    }

    Type* getBaseType() const {
        return m_base;
    }

    void assertForwardsResolved() const {
        if (!m_resolved) {
            throw std::logic_error("Type " + m_name + " has unresolved forwards.");
        }
    }

    template<class visitor_type>
    void _visitReferencedTypes(const visitor_type& v) {}

    template<class visitor_type>
    void _visitContainedTypes(const visitor_type& v) {}

    // call subtype.constructor
    void constructor(instance_ptr self);

    template<class ptr_func>
    void constructor(int64_t count, const ptr_func& ptrToChild) {
        assertForwardsResolved();

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
        assertForwardsResolved();

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

    // subtype-specific calculation
    bool _updateAfterForwardTypesChanged() { return false; }

    // called when a downstream type has changed in some way.
    // this may recalculate our on-disk size, our name, or some other
    // feature of the type that depends on the forward delcarations
    // below us.
    void forwardTypesAreResolved();

    // called after each type has initialized its internals
    void endOfConstructorInitialization();

    // call subtype.copy_constructor
    void copy_constructor(instance_ptr self, instance_ptr other);

    template<class ptr_func_dest, class ptr_func_src>
    void copy_constructor(int64_t count, const ptr_func_dest& ptrToTarget, const ptr_func_src& ptrToSrc) {
        assertForwardsResolved();

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

    bool resolved() const {
        return m_resolved;
    }

    bool isBinaryCompatibleWith(Type* other);

    bool isBinaryCompatibleWithConcrete(Type* other) {
        return false;
    }

    bool isSimple() const {
        return m_is_simple;
    }

    const std::set<Forward*>& getReferencedForwards() const {
        return m_referenced_forwards;
    }

    const std::set<Forward*>& getContainedForwards() const {
        return m_contained_forwards;
    }

    void forwardResolvedTo(Forward* forward, Type* resolvedTo);

    void setNameForRecursiveType(std::string nameOverride) {
        m_recursive_name = nameOverride;
        m_is_recursive = true;
    }

protected:
    Type(TypeCategory in_typeCategory) :
            m_typeCategory(in_typeCategory),
            m_size(0),
            m_is_default_constructible(false),
            m_name("Undefined"),
            mTypeRep(nullptr),
            m_base(nullptr),
            m_is_simple(true),
            m_resolved(false),
            m_is_recursive(false)
        {}

    TypeCategory m_typeCategory;

    size_t m_size;

    bool m_is_default_constructible;

    std::string m_recursive_name;

    std::string m_name;

    PyTypeObject* mTypeRep;

    Type* m_base;

    // 'simple' types are those that have no reference to the python interpreter
    bool m_is_simple;

    bool m_resolved;

    bool m_is_recursive;

    enum BinaryCompatibilityCategory { Incompatible, Checking, Compatible };

    std::map<Type*, BinaryCompatibilityCategory> mIsBinaryCompatible;

    // a set of forward types that we need to be resolved before we
    // could be resolved
    std::set<Forward*> m_referenced_forwards;

    // a subset of m_referenced_forwards that we directly contain
    std::set<Forward*> m_contained_forwards;
};

