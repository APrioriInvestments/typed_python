#pragma once

#include <Python.h>
#include <string>
#include <vector>
#include <algorithm>
#include <functional>
#include <mutex>
#include <set>
#include <map>
#include <utility>
#include <atomic>
#include <iostream>
#include <sstream>
#include "SerializationContext.hpp"
#include "HashAccumulator.hpp"
#include "SerializationBuffer.hpp"
#include "DeserializationBuffer.hpp"

class Type;

class None;

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

class String;
class Bytes;
class OneOf;
class Value;
class TupleOf;
class NamedTuple;
class Tuple;
class ConstDict;
class Alternative;
class ConcreteAlternative;
class PythonSubclass;
class PythonObjectOfType;
class Class;
class HeldClass;
class Function;
class BoundMethod;
class Forward;

typedef uint8_t* instance_ptr;

typedef void (*compiled_code_entrypoint)(instance_ptr, instance_ptr*);

void updateTypeRepForType(Type* t, PyTypeObject* pyType);

class Type {
public:
    enum TypeCategory {
        catNone,
        catBool,
        catUInt8,
        catUInt16,
        catUInt32,
        catUInt64,
        catInt8,
        catInt16,
        catInt32,
        catInt64,
        catString,
        catBytes,
        catFloat32,
        catFloat64,
        catValue,
        catOneOf,
        catTupleOf,
        catNamedTuple,
        catTuple,
        catConstDict,
        catAlternative,
        catConcreteAlternative, //concrete Alternative subclass
        catPythonSubclass, //subclass of a nativepython type
        catPythonObjectOfType, //a python object that matches 'isinstance' on a particular type
        catBoundMethod,
        catClass,
        catHeldClass,
        catFunction,
        catForward
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

    void repr(instance_ptr self, std::ostringstream& out);

    char cmp(instance_ptr left, instance_ptr right);

    int32_t hash32(instance_ptr left);

    template<class buf_t>
    void serialize(instance_ptr left, buf_t& buffer) {
        assertForwardsResolved();

        return this->check([&](auto& subtype) {
            return subtype.serialize(left, buffer);
        });
    }

    template<class buf_t>
    void deserialize(instance_ptr left, buf_t& buffer) {
        assertForwardsResolved();

        return this->check([&](auto& subtype) {
            return subtype.deserialize(left, buffer);
        });
    }

    void swap(instance_ptr left, instance_ptr right);

    static char byteCompare(uint8_t* l, uint8_t* r, size_t count);

    template<class T>
    auto check(const T& f) -> decltype(f(*this)) {
        switch (m_typeCategory) {
            case catNone:
                return f(*(None*)this);
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
                return f(*(String*)this);
            case catBytes:
                return f(*(Bytes*)this);
            case catFloat32:
                return f(*(Float32*)this);
            case catFloat64:
                return f(*(Float64*)this);
            case catValue:
                return f(*(Value*)this);
            case catOneOf:
                return f(*(OneOf*)this);
            case catTupleOf:
                return f(*(TupleOf*)this);
            case catNamedTuple:
                return f(*(NamedTuple*)this);
            case catTuple:
                return f(*(Tuple*)this);
            case catConstDict:
                return f(*(ConstDict*)this);
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
            default:
                throw std::runtime_error("Invalid type found");
        }
    }

    Type* getBaseType() const {
        return m_base;
    }

    void assertForwardsResolved() const {
        if (m_references_unresolved_forwards || m_failed_resolution) {
            throw std::logic_error("Type has unresolved forwards.");
        }
    }

    //this MUST be called while holding the GIL
    template<class resolve_py_callable_to_type>
    Type* guaranteeForwardsResolved(const resolve_py_callable_to_type& resolver) {
        if (m_failed_resolution) {
            throw std::runtime_error("Type failed to resolve the first time it was triggered.");
        }

        if (m_checking_for_references_unresolved_forwards) {
            //it's ok to bail early. the call stack will recurse
            //back to this point, and during the unwind will ensure
            //that we check that any forwards are resolved.
            return this;
        }

        if (m_references_unresolved_forwards) {
            m_checking_for_references_unresolved_forwards = true;
            m_references_unresolved_forwards = false;

            Type* res;

            try {
                res = this->check([&](auto& subtype) {
                    return subtype.guaranteeForwardsResolvedConcrete(resolver);
                });
            } catch(...) {
                m_checking_for_references_unresolved_forwards = false;
                m_failed_resolution = true;
                throw;
            }

            m_checking_for_references_unresolved_forwards = false;

            forwardTypesMayHaveChanged();

            if (res != this) {
                return res->guaranteeForwardsResolved(resolver);
            }

            return res;
        }

        return this;
    }

    template<class resolve_py_callable_to_type>
    Type* guaranteeForwardsResolvedConcrete(resolve_py_callable_to_type& resolver) {
        typedef Type* type_ptr;

        bool didSomething = false;

        visitReferencedTypes([&](type_ptr& t) {
            t = t->guaranteeForwardsResolved(resolver);
        });

        forwardTypesMayHaveChanged();

        return this;
    }

    void _forwardTypesMayHaveChanged() {}

    template<class visitor_type>
    void _visitReferencedTypes(const visitor_type& v) {}

    template<class visitor_type>
    void _visitContainedTypes(const visitor_type& v) {}


    // call subtype.constructor
    void constructor(instance_ptr self);

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

    void forwardTypesMayHaveChanged();

    // call subtype.copy_constructor
    void copy_constructor(instance_ptr self, instance_ptr other);

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

    bool references_unresolved_forwards() const {
        return m_references_unresolved_forwards;
    }

    bool isBinaryCompatibleWith(Type* other);

    bool isBinaryCompatibleWithConcrete(Type* other) {
        return false;
    }

protected:
    Type(TypeCategory in_typeCategory) :
            m_typeCategory(in_typeCategory),
            m_size(0),
            m_is_default_constructible(false),
            m_name("Undefined"),
            mTypeRep(nullptr),
            m_base(nullptr),
            m_references_unresolved_forwards(false),
            m_checking_for_references_unresolved_forwards(false),
            m_failed_resolution(false)
        {}

    TypeCategory m_typeCategory;

    size_t m_size;

    bool m_is_default_constructible;

    std::string m_name;

    PyTypeObject* mTypeRep;

    Type* m_base;

    bool m_references_unresolved_forwards;

    bool m_checking_for_references_unresolved_forwards;

    bool m_failed_resolution;

    std::set<Type*> mUses;

    enum BinaryCompatibilityCategory { Incompatible, Checking, Compatible };

    std::map<Type*, BinaryCompatibilityCategory> mIsBinaryCompatible;

};

#include "RegisterTypes.hpp"

// forward types are never actually used - they must be removed from the graph before
// any types that contain them can be used.
class Forward : public Type {
public:
    Forward(PyObject* deferredDefinition, std::string name) :
        Type(TypeCategory::catForward),
        mTarget(nullptr),
        mDefinition(deferredDefinition)
    {
        m_references_unresolved_forwards = true;
        m_name = name;
    }

    Type* getTarget() const {
        return mTarget;
    }

    template<class resolve_py_callable_to_type>
    Type* guaranteeForwardsResolvedConcrete(resolve_py_callable_to_type& resolver) {
        if (mTarget) {
            return mTarget;
        }

        Type* t = resolver(mDefinition);

        if (!t) {
            m_failed_resolution = true;
        }

        mTarget = t;

        if (mTarget) {
            return mTarget;
        }

        return this;
    }

    template<class visitor_type>
    void _visitContainedTypes(const visitor_type& v) {
        v(mTarget);
    }

    template<class visitor_type>
    void _visitReferencedTypes(const visitor_type& v) {
        v(mTarget);
    }

    void _forwardTypesMayHaveChanged() {
    }

    void resolveDuringSerialization(Type* newTarget) {
        if (mTarget && mTarget != newTarget) {
            throw std::runtime_error("can't resolve a forward type to a new value.");
        }

        mTarget = newTarget;
    }

private:
    Type* mTarget;
    PyObject* mDefinition;
};

class OneOf : public Type {
public:
    OneOf(const std::vector<Type*>& types) :
                    Type(TypeCategory::catOneOf),
                    m_types(types)
    {
        if (m_types.size() > 255) {
            throw std::runtime_error("OneOf types are limited to 255 alternatives in this implementation");
        }

        forwardTypesMayHaveChanged();
    }

    bool isBinaryCompatibleWithConcrete(Type* other);

    template<class visitor_type>
    void _visitContainedTypes(const visitor_type& visitor) {
        for (auto& typePtr: m_types) {
            visitor(typePtr);
        }
    }

    template<class visitor_type>
    void _visitReferencedTypes(const visitor_type& visitor) {
        _visitContainedTypes(visitor);
    }

    void _forwardTypesMayHaveChanged();

    std::string computeName() const;

    template<class buf_t>
    void deserialize(instance_ptr self, buf_t& buffer) {
        uint8_t which = buffer.read_uint8();
        if (which >= m_types.size()) {
            throw std::runtime_error("Corrupt OneOf");
        }
        *(uint8_t*)self = which;
        m_types[which]->deserialize(self+1, buffer);
    }

    template<class buf_t>
    void serialize(instance_ptr self, buf_t& buffer) {
        buffer.write_uint8(*(uint8_t*)self);
        m_types[*((uint8_t*)self)]->serialize(self+1, buffer);
    }

    void repr(instance_ptr self, std::ostringstream& stream);

    int32_t hash32(instance_ptr left);

    char cmp(instance_ptr left, instance_ptr right);

    std::pair<Type*, instance_ptr> unwrap(instance_ptr self) {
        return std::make_pair(m_types[*(uint8_t*)self], self+1);
    }

    size_t computeBytecount() const;

    void constructor(instance_ptr self);

    void destroy(instance_ptr self);

    void copy_constructor(instance_ptr self, instance_ptr other);

    void assign(instance_ptr self, instance_ptr other);

    const std::vector<Type*>& getTypes() const {
        return m_types;
    }

    static OneOf* Make(const std::vector<Type*>& types);

private:
    std::vector<Type*> m_types;
};

class CompositeType : public Type {
public:
    CompositeType(
                TypeCategory in_typeCategory,
                const std::vector<Type*>& types,
                const std::vector<std::string>& names
                ) :
            Type(in_typeCategory),
            m_types(types),
            m_names(names)
    {
    }

    bool isBinaryCompatibleWithConcrete(Type* other);

    template<class visitor_type>
    void _visitContainedTypes(const visitor_type& visitor) {
        for (auto& typePtr: m_types) {
            visitor(typePtr);
        }
    }

    template<class visitor_type>
    void _visitReferencedTypes(const visitor_type& visitor) {
        _visitContainedTypes(visitor);
    }

    void _forwardTypesMayHaveChanged();

    instance_ptr eltPtr(instance_ptr self, int64_t ix) const {
        return self + m_byte_offsets[ix];
    }

    char cmp(instance_ptr left, instance_ptr right);

    template<class buf_t>
    void deserialize(instance_ptr self, buf_t& buffer) {
        for (long k = 0; k < getTypes().size();k++) {
            getTypes()[k]->deserialize(eltPtr(self,k),buffer);
        }
    }

    template<class buf_t>
    void serialize(instance_ptr self, buf_t& buffer) {
        for (long k = 0; k < getTypes().size();k++) {
            getTypes()[k]->serialize(eltPtr(self,k),buffer);
        }
    }

    void repr(instance_ptr self, std::ostringstream& stream);

    int32_t hash32(instance_ptr left);

    template<class sub_constructor>
    void constructor(instance_ptr self, const sub_constructor& initializer) {
        for (int64_t k = 0; k < getTypes().size(); k++) {
            try {
                initializer(eltPtr(self, k), k);
            } catch(...) {
                for (long k2 = k-1; k2 >= 0; k2--) {
                    m_types[k2]->destroy(eltPtr(self,k2));
                }
                throw;
            }
        }
    }

    void constructor(instance_ptr self);

    void destroy(instance_ptr self);

    void copy_constructor(instance_ptr self, instance_ptr other);

    void assign(instance_ptr self, instance_ptr other);

    const std::vector<Type*>& getTypes() const {
        return m_types;
    }
    const std::vector<size_t>& getOffsets() const {
        return m_byte_offsets;
    }
    const std::vector<std::string>& getNames() const {
        return m_names;
    }
protected:
    template<class subtype>
    static subtype* MakeSubtype(const std::vector<Type*>& types, const std::vector<std::string>& names) {
        static std::mutex guard;

        std::lock_guard<std::mutex> lock(guard);

        typedef std::pair<const std::vector<Type*>, const std::vector<std::string> > keytype;

        static std::map<keytype, subtype*> m;

        auto it = m.find(keytype(types,names));
        if (it == m.end()) {
            it = m.insert(std::make_pair(keytype(types,names), new subtype(types,names))).first;
        }

        return it->second;
    }

    std::vector<Type*> m_types;
    std::vector<size_t> m_byte_offsets;
    std::vector<std::string> m_names;
};

class NamedTuple : public CompositeType {
public:
    NamedTuple(const std::vector<Type*>& types, const std::vector<std::string>& names) :
            CompositeType(TypeCategory::catNamedTuple, types, names)
    {
        assert(types.size() == names.size());
        forwardTypesMayHaveChanged();
    }

    void _forwardTypesMayHaveChanged();

    static NamedTuple* Make(const std::vector<Type*>& types, const std::vector<std::string>& names) {
        return MakeSubtype<NamedTuple>(types,names);
    }
};

class Tuple : public CompositeType {
public:
    Tuple(const std::vector<Type*>& types, const std::vector<std::string>& names) :
            CompositeType(TypeCategory::catTuple, types, names)
    {
        forwardTypesMayHaveChanged();
    }

    void _forwardTypesMayHaveChanged();

    static Tuple* Make(const std::vector<Type*>& types) {
        return MakeSubtype<Tuple>(types, std::vector<std::string>());
    }
};

class TupleOf : public Type {
    class layout {
    public:
        std::atomic<int64_t> refcount;
        int32_t hash_cache;
        int32_t count;
        uint8_t data[];
    };

public:
    TupleOf(Type* type) :
            Type(TypeCategory::catTupleOf),
            m_element_type(type)
    {
        m_size = sizeof(void*);
        m_is_default_constructible = true;

        forwardTypesMayHaveChanged();
    }

    bool isBinaryCompatibleWithConcrete(Type* other);

    template<class visitor_type>
    void _visitContainedTypes(const visitor_type& visitor) {

    }

    template<class visitor_type>
    void _visitReferencedTypes(const visitor_type& visitor) {
        visitor(m_element_type);
    }

    void _forwardTypesMayHaveChanged() {
        m_name = "TupleOf(" + m_element_type->name() + ")";
    }

    template<class buf_t>
    void serialize(instance_ptr self, buf_t& buffer) {
        int32_t ct = count(self);
        buffer.write_uint32(ct);
        for (long k = 0; k < ct;k++) {
            m_element_type->serialize(eltPtr(self,k),buffer);
        }
    }

    template<class buf_t>
    void deserialize(instance_ptr self, buf_t& buffer) {
        int32_t ct = buffer.read_uint32();

        if (ct > buffer.remaining() && m_element_type->bytecount()) {
            throw std::runtime_error("Corrupt data (tuplecount)");
        }

        constructor(self, ct, [&](instance_ptr tgt, int k) {
            m_element_type->deserialize(tgt, buffer);
        });
    }

    void repr(instance_ptr self, std::ostringstream& stream);

    int32_t hash32(instance_ptr left);

    char cmp(instance_ptr left, instance_ptr right);

    Type* getEltType() const {
        return m_element_type;
    }

    static TupleOf* Make(Type* elt);

    instance_ptr eltPtr(instance_ptr self, int64_t i) const;

    int64_t count(instance_ptr self) const;

    int64_t refcount(instance_ptr self) const;

    template<class sub_constructor>
    void constructor(instance_ptr self, int64_t count, const sub_constructor& allocator) {
        if (count == 0) {
            (*(layout**)self) = nullptr;
            return;
        }

        (*(layout**)self) = (layout*)malloc(sizeof(layout) + getEltType()->bytecount() * count);

        (*(layout**)self)->count = count;
        (*(layout**)self)->refcount = 1;
        (*(layout**)self)->hash_cache = -1;

        for (int64_t k = 0; k < count; k++) {
            try {
                allocator(eltPtr(self, k), k);
            } catch(...) {
                for (long k2 = k-1; k2 >= 0; k2--) {
                    m_element_type->destroy(eltPtr(self,k2));
                }
                free(*(layout**)self);
                throw;
            }
        }
    }

    void constructor(instance_ptr self);

    void destroy(instance_ptr self);

    void copy_constructor(instance_ptr self, instance_ptr other);

    void assign(instance_ptr self, instance_ptr other);

private:
    Type* m_element_type;
};



class ConstDict : public Type {
    class layout {
    public:
        std::atomic<int64_t> refcount;
        int32_t hash_cache;
        int32_t count;
        int32_t subpointers; //if 0, then all values are inline as pairs of (key,value)
                             //otherwise, its an array of '(key, ConstDict(key,value))'
        uint8_t data[];
    };

public:
    ConstDict(Type* key, Type* value) :
            Type(TypeCategory::catConstDict),
            m_key(key),
            m_value(value)
    {
        forwardTypesMayHaveChanged();
    }

    template<class visitor_type>
    void _visitContainedTypes(const visitor_type& visitor) {
    }

    template<class visitor_type>
    void _visitReferencedTypes(const visitor_type& visitor) {
        visitor(m_key);
        visitor(m_value);
    }

    void _forwardTypesMayHaveChanged();

    bool isBinaryCompatibleWithConcrete(Type* other);

    static ConstDict* Make(Type* key, Type* value);

    template<class buf_t>
    void serialize(instance_ptr self, buf_t& buffer) {
        int32_t ct = count(self);
        buffer.write_uint32(ct);
        for (long k = 0; k < ct;k++) {
            m_key->serialize(kvPairPtrKey(self,k),buffer);
            m_value->serialize(kvPairPtrValue(self,k),buffer);
        }
    }

    template<class buf_t>
    void deserialize(instance_ptr self, buf_t& buffer) {
        int32_t ct = buffer.read_uint32();

        if (ct > buffer.remaining() && m_bytes_per_key_value_pair) {
            throw std::runtime_error("Corrupt data (dictcount)");
        }

        constructor(self, ct, false);

        for (long k = 0; k < ct;k++) {
            m_key->deserialize(kvPairPtrKey(self,k),buffer);
            m_value->deserialize(kvPairPtrValue(self,k),buffer);
        }

        incKvPairCount(self, ct);
    }

    void repr(instance_ptr self, std::ostringstream& stream);

    int32_t hash32(instance_ptr left);

    //to make this fast(er), we do dict size comparison first, then keys, then values
    char cmp(instance_ptr left, instance_ptr right);

    void addDicts(instance_ptr lhs, instance_ptr rhs, instance_ptr output) const;

    TupleOf* tupleOfKeysType() const {
        return TupleOf::Make(m_key);
    }

    void subtractTupleOfKeysFromDict(instance_ptr lhs, instance_ptr rhs, instance_ptr output) const;

    instance_ptr kvPairPtrKey(instance_ptr self, int64_t i) const;

    instance_ptr kvPairPtrValue(instance_ptr self, int64_t i) const;

    void incKvPairCount(instance_ptr self, int by = 1) const;

    void sortKvPairs(instance_ptr self) const;

    instance_ptr keyTreePtr(instance_ptr self, int64_t i) const;

    bool instanceIsSubtrees(instance_ptr self) const;

    int64_t count(instance_ptr self) const;

    int64_t size(instance_ptr self) const;

    int64_t lookupIndexByKey(instance_ptr self, instance_ptr key) const;

    instance_ptr lookupValueByKey(instance_ptr self, instance_ptr key) const;

    void constructor(instance_ptr self, int64_t space, bool isPointerTree) const;

    void constructor(instance_ptr self);

    void destroy(instance_ptr self);

    void copy_constructor(instance_ptr self, instance_ptr other);

    void assign(instance_ptr self, instance_ptr other);


    Type* keyValuePairType() const { return m_key_value_pair_type; }
    Type* keyType() const { return m_key; }
    Type* valueType() const { return m_value; }

private:
    Type* m_key;
    Type* m_value;
    Type* m_key_value_pair_type;
    size_t m_bytes_per_key;
    size_t m_bytes_per_key_value_pair;
    size_t m_bytes_per_key_subtree_pair;
};

class None : public Type {
public:
    None() : Type(TypeCategory::catNone)
    {
        m_name = "NoneType";
        m_size = 0;
        m_is_default_constructible = true;
    }

    bool isBinaryCompatibleWithConcrete(Type* other) {
        if (other->getTypeCategory() != m_typeCategory) {
            return false;
        }

        return true;
    }

    void _forwardTypesMayHaveChanged() {}

    template<class visitor_type>
    void _visitReferencedTypes(const visitor_type& v) {}

    template<class visitor_type>
    void _visitContainedTypes(const visitor_type& v) {}


    char cmp(instance_ptr left, instance_ptr right) {
        return 0;
    }

    int32_t hash32(instance_ptr left) {
        return (int)getTypeCategory();
    }

    void constructor(instance_ptr self) {}

    void destroy(instance_ptr self) {}

    void copy_constructor(instance_ptr self, instance_ptr other) {}

    void assign(instance_ptr self, instance_ptr other) {}

    static None* Make() { static None res; return &res; }

    template<class buf_t>
    void deserialize(instance_ptr self, buf_t& buffer) {
    }

    template<class buf_t>
    void serialize(instance_ptr self, buf_t& buffer) {
    }

    void repr(instance_ptr self, std::ostringstream& stream) {
        stream << "None";
    }
};

#include "Instance.hpp"

class String : public Type {
public:
    class layout {
    public:
        std::atomic<int64_t> refcount;
        int32_t hash_cache;
        int32_t pointcount;
        int32_t bytes_per_codepoint; //1 implies
        uint8_t data[];

        uint8_t* eltPtr(int64_t i) {
            return data + i * bytes_per_codepoint;
        }
    };

    String() : Type(TypeCategory::catString)
    {
        m_name = "String";
        m_is_default_constructible = true;
        m_size = sizeof(void*);
    }

    bool isBinaryCompatibleWithConcrete(Type* other) {
        if (other->getTypeCategory() != m_typeCategory) {
            return false;
        }

        return true;
    }

    void _forwardTypesMayHaveChanged() {}

    template<class visitor_type>
    void _visitReferencedTypes(const visitor_type& v) {}

    template<class visitor_type>
    void _visitContainedTypes(const visitor_type& v) {}


    static String* Make() { static String res; return &res; }

    template<class buf_t>
    void serialize(instance_ptr self, buf_t& buffer) {
        buffer.write_uint32(count(self));
        buffer.write_uint8(bytes_per_codepoint(self));
        buffer.write_bytes(eltPtr(self,0), bytes_per_codepoint(self) * count(self));
    }

    template<class buf_t>
    void deserialize(instance_ptr self, buf_t& buffer) {
        int32_t ct = buffer.read_uint32();
        uint8_t bytes_per = buffer.read_uint8();

        if (bytes_per != 1 && bytes_per != 2 && bytes_per != 4) {
            throw std::runtime_error("Corrupt data (bytes per unicode character): "
                + std::to_string(bytes_per) + " " + std::to_string(ct) + ". pos is " + std::to_string(buffer.pos()));
        }

        if (ct > buffer.remaining()) {
            throw std::runtime_error("Corrupt data (stringsize)");
        }

        constructor(self, bytes_per, ct, nullptr);

        if (ct) {
            buffer.read_bytes(eltPtr(self,0), bytes_per * ct);
        }
    }

    int32_t hash32(instance_ptr left);

    char cmp(instance_ptr left, instance_ptr right);

    void constructor(instance_ptr self, int64_t bytes_per_codepoint, int64_t count, const char* data) const;

    void repr(instance_ptr self, std::ostringstream& stream);

    instance_ptr eltPtr(instance_ptr self, int64_t i) const;

    int64_t bytes_per_codepoint(instance_ptr self) const;

    int64_t count(instance_ptr self) const;

    void constructor(instance_ptr self) {
        *(layout**)self = 0;
    }

    void destroy(instance_ptr self);

    void copy_constructor(instance_ptr self, instance_ptr other);

    void assign(instance_ptr self, instance_ptr other);
};

class Bytes : public Type {
public:
    class layout {
    public:
        std::atomic<int64_t> refcount;
        int32_t hash_cache;
        int32_t bytecount;
        uint8_t data[];
    };

    Bytes() : Type(TypeCategory::catBytes)
    {
        m_name = "Bytes";
        m_is_default_constructible = true;
        m_size = sizeof(layout*);
    }

    bool isBinaryCompatibleWithConcrete(Type* other);

    void repr(instance_ptr self, std::ostringstream& stream);

    template<class buf_t>
    void serialize(instance_ptr self, buf_t& buffer) {
        buffer.write_uint32(count(self));
        buffer.write_bytes(eltPtr(self, 0), count(self));
    }

    template<class buf_t>
    void deserialize(instance_ptr self, buf_t& buffer) {
        int32_t ct = buffer.read_uint32();

        if (ct > buffer.remaining()) {
            throw std::runtime_error("Corrupt data (bytes)");
        }

        constructor(self, ct, nullptr);

        if (ct) {
            buffer.read_bytes(eltPtr(self,0), ct);
        }
    }

    void _forwardTypesMayHaveChanged() {}

    template<class visitor_type>
    void _visitReferencedTypes(const visitor_type& v) {}

    template<class visitor_type>
    void _visitContainedTypes(const visitor_type& v) {}

    int32_t hash32(instance_ptr left);

    char cmp(instance_ptr left, instance_ptr right);

    static Bytes* Make() { static Bytes res; return &res; }

    void constructor(instance_ptr self, int64_t count, const char* data) const;

    instance_ptr eltPtr(instance_ptr self, int64_t i) const;

    int64_t count(instance_ptr self) const;

    void constructor(instance_ptr self) {
        *(layout**)self = 0;
    }

    void destroy(instance_ptr self);

    void copy_constructor(instance_ptr self, instance_ptr other);

    void assign(instance_ptr self, instance_ptr other);
};

class Value : public Type {
public:
    Value(Instance instance) :
            Type(TypeCategory::catValue),
            mInstance(instance)
    {
        m_size = 0;
        m_is_default_constructible = true;
        m_name = mInstance.repr();
    }

    bool isBinaryCompatibleWithConcrete(Type* other) {
        return this == other;
    }

    template<class visitor_type>
    void _visitContainedTypes(const visitor_type& visitor) {
    }

    template<class visitor_type>
    void _visitReferencedTypes(const visitor_type& visitor) {
    }

    void _forwardTypesMayHaveChanged() {
    }

    char cmp(instance_ptr left, instance_ptr right) {
        return 0;
    }

    int32_t hash32(instance_ptr left) {
        return mInstance.hash32();
    }

    void repr(instance_ptr self, std::ostringstream& stream) {
        mInstance.type()->repr(mInstance.data(), stream);
    }

    template<class buf_t>
    void serialize(instance_ptr self, buf_t& buffer) {
    }

    template<class buf_t>
    void deserialize(instance_ptr self, buf_t& buffer) {
    }

    void constructor(instance_ptr self) {}

    void destroy(instance_ptr self) {}

    void copy_constructor(instance_ptr self, instance_ptr other) {}

    void assign(instance_ptr self, instance_ptr other) {}

    const Instance& value() const {
        return mInstance;
    }

    static Type* Make(Instance i) {
        static std::mutex guard;

        std::lock_guard<std::mutex> lock(guard);

        static std::map<Instance, Value*> m;

        auto it = m.find(i);

        if (it == m.end()) {
            it = m.insert(std::make_pair(i, new Value(i))).first;
        }

        return it->second;
    }

    static Type* MakeInt64(int64_t i) {
        return Make(Instance::create(Int64::Make(), (instance_ptr)&i));
    }
    static Type* MakeFloat64(double i) {
        return Make(Instance::create(Float64::Make(), (instance_ptr)&i));
    }
    static Type* MakeBool(bool i) {
        return Make(Instance::create(Bool::Make(), (instance_ptr)&i));
    }

    static Type* MakeBytes(char* data, size_t count) {
        return Make(Instance::createAndInitialize(Bytes::Make(), [&](instance_ptr i) {
            Bytes::Make()->constructor(i, count, data);
        }));
    }

    static Type* MakeString(size_t bytesPerCodepoint, size_t count, char* data) {
        return Make(Instance::createAndInitialize(String::Make(), [&](instance_ptr i) {
            String::Make()->constructor(i, bytesPerCodepoint, count, data);
        }));
    }

private:
    Instance mInstance;
};

class Alternative : public Type {
public:
    class layout {
    public:
        std::atomic<int64_t> refcount;

        int64_t which;
        uint8_t data[];
    };

    Alternative(std::string name,
                const std::vector<std::pair<std::string, NamedTuple*> >& subtypes,
                const std::map<std::string, Function*>& methods
                ) :
            Type(TypeCategory::catAlternative),
            m_default_construction_ix(0),
            m_default_construction_type(nullptr),
            m_subtypes(subtypes),
            m_methods(methods)
    {
        m_name = name;

        if (m_subtypes.size() > 255) {
            throw std::runtime_error("Can't have an alternative with more than 255 subelements");
        }

        forwardTypesMayHaveChanged();
    }

    bool isBinaryCompatibleWithConcrete(Type* other);

    template<class visitor_type>
    void _visitContainedTypes(const visitor_type& visitor) {
    }

    template<class visitor_type>
    void _visitReferencedTypes(const visitor_type& visitor) {
        for (auto& subtype_pair: m_subtypes) {
            Type* t = subtype_pair.second;
            visitor(t);
            assert(t == subtype_pair.second);
        }
        for (auto& method_pair: m_methods) {
            Type* t = method_pair.second;
            visitor(t);
            assert(t == method_pair.second);
        }
    }

    void _forwardTypesMayHaveChanged();

    char cmp(instance_ptr left, instance_ptr right);

    template<class buf_t>
    void serialize(instance_ptr self, buf_t& buffer) {
        buffer.write_uint8(which(self));
        m_subtypes[which(self)].second->serialize(eltPtr(self), buffer);
    }

    template<class buf_t>
    void deserialize(instance_ptr self, buf_t& buffer) {
        uint8_t w = buffer.read_uint8();
        if (w >= m_subtypes.size()) {
            throw std::runtime_error("Corrupt data (alt which)");
        }

        if (m_all_alternatives_empty) {
            *(uint8_t*)self = w;
            return;
        }

        *(layout**)self = (layout*)malloc(
            sizeof(layout) +
            m_subtypes[w].second->bytecount()
            );

        layout& record = **(layout**)self;

        record.refcount = 1;
        record.which = w;

        m_subtypes[w].second->deserialize(record.data, buffer);
    }

    void repr(instance_ptr self, std::ostringstream& stream);

    int32_t hash32(instance_ptr left);

    instance_ptr eltPtr(instance_ptr self) const;

    int64_t which(instance_ptr self) const;

    void constructor(instance_ptr self);

    void destroy(instance_ptr self);

    void copy_constructor(instance_ptr self, instance_ptr other);

    void assign(instance_ptr self, instance_ptr other);

    static Alternative* Make(std::string name,
                         const std::vector<std::pair<std::string, NamedTuple*> >& types,
                         const std::map<std::string, Function*>& methods //methods preclude us from being in the memo
                         );

    const std::vector<std::pair<std::string, NamedTuple*> >& subtypes() const {
        return m_subtypes;
    }

    bool all_alternatives_empty() const {
        return m_all_alternatives_empty;
    }

    Type* pickConcreteSubclassConcrete(instance_ptr data);

    const std::map<std::string, Function*>& getMethods() const {
        return m_methods;
    }

private:
    bool m_all_alternatives_empty;

    int m_default_construction_ix;

    Type* m_default_construction_type;

    std::vector<std::pair<std::string, NamedTuple*> > m_subtypes;

    std::map<std::string, Function*> m_methods;

    std::map<std::string, int> m_arg_positions;
};

class ConcreteAlternative : public Type {
public:
    typedef Alternative::layout layout;

    ConcreteAlternative(Alternative* m_alternative, int64_t which) :
            Type(TypeCategory::catConcreteAlternative),
            m_alternative(m_alternative),
            m_which(which)
    {
        forwardTypesMayHaveChanged();
    }

    bool isBinaryCompatibleWithConcrete(Type* other);

    template<class visitor_type>
    void _visitContainedTypes(const visitor_type& visitor) {
        Type* t = m_alternative;
        visitor(t);
        assert(t == m_alternative);
    }

    template<class visitor_type>
    void _visitReferencedTypes(const visitor_type& visitor) {
        Type* t = m_alternative;
        visitor(t);
        assert(t == m_alternative);
    }

    void _forwardTypesMayHaveChanged();

    int32_t hash32(instance_ptr left) {
        return m_alternative->hash32(left);
    }

    void repr(instance_ptr self, std::ostringstream& stream) {
        m_alternative->repr(self,stream);
    }

    template<class buf_t>
    void deserialize(instance_ptr self, buf_t& buffer) {
        m_alternative->deserialize(self,buffer);
    }

    template<class buf_t>
    void serialize(instance_ptr self, buf_t& buffer) {
        m_alternative->serialize(self,buffer);
    }

    char cmp(instance_ptr left, instance_ptr right) {
        return m_alternative->cmp(left,right);
    }

    void constructor(instance_ptr self);

    //returns an uninitialized object of type-index 'which'
    template<class subconstructor>
    void constructor(instance_ptr self, const subconstructor& s) const {
        if (m_alternative->all_alternatives_empty()) {
            *(uint8_t*)self = m_which;
            s(self);
        } else {
            *(layout**)self = (layout*)malloc(
                sizeof(layout) +
                elementType()->bytecount()
                );

            layout& record = **(layout**)self;
            record.refcount = 1;
            record.which = m_which;
            try {
                s(record.data);
            } catch(...) {
                free(*(layout**)self);
                throw;
            }
        }
    }

    void destroy(instance_ptr self) {
        m_alternative->destroy(self);
    }

    void copy_constructor(instance_ptr self, instance_ptr other) {
        m_alternative->copy_constructor(self, other);
    }

    void assign(instance_ptr self, instance_ptr other) {
        m_alternative->assign(self, other);
    }

    static ConcreteAlternative* Make(Alternative* alt, int64_t which);

    Type* elementType() const {
        return m_alternative->subtypes()[m_which].second;
    }

    Alternative* getAlternative() const {
        return m_alternative;
    }

    int64_t which() const {
        return m_which;
    }

private:
    Alternative* m_alternative;

    int64_t m_which;
};

class PythonSubclass : public Type {
public:
    PythonSubclass(Type* base, PyTypeObject* typePtr) :
            Type(TypeCategory::catPythonSubclass)
    {
        m_base = base;
        mTypeRep = typePtr;
        m_name = typePtr->tp_name;

        forwardTypesMayHaveChanged();
    }

    bool isBinaryCompatibleWithConcrete(Type* other);

    template<class visitor_type>
    void _visitContainedTypes(const visitor_type& visitor) {
        visitor(m_base);
    }

    template<class visitor_type>
    void _visitReferencedTypes(const visitor_type& visitor) {
        visitor(m_base);
    }

    void _forwardTypesMayHaveChanged() {
        m_size = m_base->bytecount();
        m_is_default_constructible = m_base->is_default_constructible();
    }

    int32_t hash32(instance_ptr left) {
        return m_base->hash32(left);
    }

    template<class buf_t>
    void serialize(instance_ptr self, buf_t& buffer) {
        m_base->serialize(self,buffer);
    }

    template<class buf_t>
    void deserialize(instance_ptr self, buf_t& buffer) {
        m_base->deserialize(self,buffer);
    }

    void repr(instance_ptr self, std::ostringstream& stream) {
        m_base->repr(self,stream);
    }

    char cmp(instance_ptr left, instance_ptr right) {
        return m_base->cmp(left,right);
    }

    void constructor(instance_ptr self) {
        m_base->constructor(self);
    }

    void destroy(instance_ptr self) {
        m_base->destroy(self);
    }

    void copy_constructor(instance_ptr self, instance_ptr other) {
        m_base->copy_constructor(self, other);
    }

    void assign(instance_ptr self, instance_ptr other) {
        m_base->assign(self, other);
    }

    static PythonSubclass* Make(Type* base, PyTypeObject* pyType);

    Type* baseType() const {
        return m_base;
    }

    PyTypeObject* pyType() const {
        return mTypeRep;
    }
};

//wraps an actual python instance. Note that we assume we're holding the GIL whenever
//we interact with actual python objects. Compiled code needs to treat these objects
//with extreme care...
class PythonObjectOfType : public Type {
public:
    PythonObjectOfType(PyTypeObject* typePtr) :
            Type(TypeCategory::catPythonObjectOfType)
    {
        mPyTypePtr = typePtr;
        m_name = typePtr->tp_name;

        forwardTypesMayHaveChanged();
    }

    bool isBinaryCompatibleWithConcrete(Type* other) {
        return other == this;
    }

    template<class visitor_type>
    void _visitContainedTypes(const visitor_type& visitor) {
    }

    template<class visitor_type>
    void _visitReferencedTypes(const visitor_type& visitor) {
    }

    void _forwardTypesMayHaveChanged() {
        m_size = sizeof(PyObject*);

        int isinst = PyObject_IsInstance(Py_None, (PyObject*)mPyTypePtr);
        if (isinst == -1) {
            isinst = 0;
            PyErr_Clear();
        }

        m_is_default_constructible = isinst != 0;
    }

    int32_t hash32(instance_ptr left) {
        PyObject* p = *(PyObject**)left;

        return PyObject_Hash(p);
    }

    template<class buf_t>
    void serialize(instance_ptr self, buf_t& buffer) {
        PyObject* p = *(PyObject**)self;
        buffer.getContext().serializePythonObject(p, buffer);
    }

    template<class buf_t>
    void deserialize(instance_ptr self, buf_t& buffer) {
         *(PyObject**)self = buffer.getContext().deserializePythonObject(buffer);
    }

    void repr(instance_ptr self, std::ostringstream& stream);

    char cmp(instance_ptr left, instance_ptr right);

    void constructor(instance_ptr self) {
        *(PyObject**)self = Py_None;
        Py_INCREF(Py_None);
    }

    void destroy(instance_ptr self) {
        Py_DECREF(*(PyObject**)self);
    }

    void copy_constructor(instance_ptr self, instance_ptr other) {
        Py_INCREF(*(PyObject**)other);
        *(PyObject**)self = *(PyObject**)other;
    }

    void assign(instance_ptr self, instance_ptr other) {
        Py_INCREF(*(PyObject**)other);
        Py_DECREF(*(PyObject**)self);
        *(PyObject**)self = *(PyObject**)other;
    }

    static PythonObjectOfType* Make(PyTypeObject* pyType);

    PyTypeObject* pyType() const {
        return mPyTypePtr;
    }

private:
    PyTypeObject* mPyTypePtr;
};

class Function : public Type {
public:
    class FunctionArg {
    public:
        FunctionArg(std::string name, Type* typeFilterOrNull, PyObject* defaultValue, bool isStarArg, bool isKwarg) :
            m_name(name),
            m_typeFilter(typeFilterOrNull),
            m_defaultValue(defaultValue),
            m_isStarArg(isStarArg),
            m_isKwarg(isKwarg)
        {
            assert(!(isStarArg && isKwarg));
        }

        std::string getName() const {
            return m_name;
        }

        PyObject* getDefaultValue() const {
            return m_defaultValue;
        }

        Type* getTypeFilter() const {
            return m_typeFilter;
        }

        bool getIsStarArg() const {
            return m_isStarArg;
        }

        bool getIsKwarg() const {
            return m_isKwarg;
        }

        bool getIsNormalArg() const {
            return !m_isKwarg && !m_isStarArg;
        }

        template<class visitor_type>
        void _visitReferencedTypes(const visitor_type& visitor) {
            if (m_typeFilter) {
                visitor(m_typeFilter);
            }
        }

    private:
        std::string m_name;
        Type* m_typeFilter;
        PyObject* m_defaultValue;
        bool m_isStarArg;
        bool m_isKwarg;
    };

    class Overload {
    public:
        Overload(
            PyFunctionObject* functionObj,
            Type* returnType,
            const std::vector<FunctionArg>& args
            ) :
                mFunctionObj(functionObj),
                mReturnType(returnType),
                mArgs(args),
                mCompiledCodePtr(nullptr)
        {
        }

        PyFunctionObject* getFunctionObj() const {
            return mFunctionObj;
        }

        Type* getReturnType() const {
            return mReturnType;
        }

        const std::vector<FunctionArg>& getArgs() const {
            return mArgs;
        }

        template<class visitor_type>
        void _visitReferencedTypes(const visitor_type& visitor) {
            if (mReturnType) {
                visitor(mReturnType);
            }
            for (auto& a: mArgs) {
                a._visitReferencedTypes(visitor);
            }
        }

        compiled_code_entrypoint getEntrypoint() const {
            return mCompiledCodePtr;
        }

        void setEntrypoint(compiled_code_entrypoint e) {
            if (mCompiledCodePtr) {
                throw std::runtime_error("Can't redefine a function entrypoint");
            }

            mCompiledCodePtr = e;
        }

    private:
        PyFunctionObject* mFunctionObj;
        Type* mReturnType;
        std::vector<FunctionArg> mArgs;
        compiled_code_entrypoint mCompiledCodePtr; //accepts a pointer to packed arguments and another pointer with the return value
    };

    class Matcher {
    public:
        Matcher(const Overload& overload) :
                mOverload(overload),
                mArgs(overload.getArgs())
        {
            m_used.resize(overload.getArgs().size());
            m_matches = true;
        }

        bool stillMatches() const {
            return m_matches;
        }

        //called at the end to see if this was a valid match
        bool definitelyMatches() const {
            if (!m_matches) {
                return false;
            }

            for (long k = 0; k < m_used.size(); k++) {
                if (!m_used[k] && !mArgs[k].getDefaultValue() && mArgs[k].getIsNormalArg()) {
                    return false;
                }
            }

            return true;
        }

        //push the state machine forward.
        Type* requiredTypeForArg(const char* name) {
            if (!name) {
                for (long k = 0; k < m_used.size(); k++) {
                    if (!m_used[k]) {
                        if (mArgs[k].getIsNormalArg()) {
                            m_used[k] = true;
                            return mArgs[k].getTypeFilter();
                        }
                        else if (mArgs[k].getIsStarArg()) {
                            //this doesn't consume the star arg
                            return mArgs[k].getTypeFilter();
                        }
                        else {
                            //this is a kwarg, but we didn't give a name.
                            m_matches = false;
                            return nullptr;
                        }
                    }
                }
            }
            else if (name) {
                for (long k = 0; k < m_used.size(); k++) {
                    if (!m_used[k]) {
                        if (mArgs[k].getIsNormalArg() && mArgs[k].getName() == name) {
                            m_used[k] = true;
                            return mArgs[k].getTypeFilter();
                        }
                        else if (mArgs[k].getIsNormalArg()) {
                            //just keep going
                        }
                        else if (mArgs[k].getIsStarArg()) {
                            //just keep going
                        } else {
                            //this is a kwarg
                            return mArgs[k].getTypeFilter();
                        }
                    }
                }
            }

            m_matches = false;
            return nullptr;
        }

    private:
        const Overload& mOverload;
        const std::vector<FunctionArg>& mArgs;
        std::vector<char> m_used;
        bool m_matches;
    };

    Function(std::string inName,
            const std::vector<Overload>& overloads
            ) :
        Type(catFunction),
        mOverloads(overloads)
    {
        m_name = inName;
        m_is_default_constructible = true;
        m_size = 0;

        forwardTypesMayHaveChanged();
    }

    template<class visitor_type>
    void _visitContainedTypes(const visitor_type& visitor) {
    }

    template<class visitor_type>
    void _visitReferencedTypes(const visitor_type& visitor) {
        for (auto& o: mOverloads) {
            o._visitReferencedTypes(visitor);
        }
    }

    void _forwardTypesMayHaveChanged() {
    }

    static Function* merge(Function* f1, Function* f2) {
        std::vector<Overload> overloads(f1->mOverloads);
        for (auto o: f2->mOverloads) {
            overloads.push_back(o);
        }
        return new Function(f1->m_name, overloads);
    }

    char cmp(instance_ptr left, instance_ptr right) {
        return 0;
    }

    template<class buf_t>
    void deserialize(instance_ptr self, buf_t& buffer) {
    }

    template<class buf_t>
    void serialize(instance_ptr self, buf_t& buffer) {
    }

    void repr(instance_ptr self, std::ostringstream& stream) {
        stream << "<function " << m_name << ">";
    }

    int32_t hash32(instance_ptr left) {
        Hash32Accumulator acc((int)getTypeCategory());

        acc.addRegister((uint64_t)mPyFunc);

        return acc.get();
    }

    void constructor(instance_ptr self) {
    }

    void destroy(instance_ptr self) {
    }

    void copy_constructor(instance_ptr self, instance_ptr other) {
    }

    void assign(instance_ptr self, instance_ptr other) {
    }

    const PyFunctionObject* getPyFunc() const {
        return mPyFunc;
    }

    const std::vector<Overload>& getOverloads() const {
        return mOverloads;
    }

    void setEntrypoint(long whichOverload, compiled_code_entrypoint entrypoint) {
        if (whichOverload < 0 || whichOverload >= mOverloads.size()) {
            throw std::runtime_error("Invalid overload index.");
        }

        mOverloads[whichOverload].setEntrypoint(entrypoint);
    }

private:
    PyFunctionObject* mPyFunc;
    std::vector<Overload> mOverloads;
};

//a class held directly inside of another object
class HeldClass : public Type {
public:
    HeldClass(std::string inName,
          const std::vector<std::pair<std::string, Type*> >& members,
          const std::map<std::string, Function*>& memberFunctions,
          const std::map<std::string, Function*>& staticFunctions,
          const std::map<std::string, PyObject*>& classMembers
          ) :
            Type(catHeldClass),
            m_members(members),
            m_memberFunctions(memberFunctions),
            m_staticFunctions(staticFunctions),
            m_classMembers(classMembers)
    {
        m_name = inName;
        forwardTypesMayHaveChanged();
    }

    bool isBinaryCompatibleWithConcrete(Type* other);

    template<class visitor_type>
    void _visitContainedTypes(const visitor_type& visitor) {
        for (auto& o: m_members) {
            visitor(o.second);
        }
    }

    template<class visitor_type>
    void _visitReferencedTypes(const visitor_type& visitor) {
        for (auto& o: m_members) {
            visitor(o.second);
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
    }

    void _forwardTypesMayHaveChanged();

    static HeldClass* Make(
            std::string inName,
            const std::vector<std::pair<std::string, Type*> >& members,
            const std::map<std::string, Function*>& memberFunctions,
            const std::map<std::string, Function*>& staticFunctions,
            const std::map<std::string, PyObject*>& classMembers
            )
    {
        return new HeldClass(inName, members, memberFunctions, staticFunctions, classMembers);
    }

    instance_ptr eltPtr(instance_ptr self, int64_t ix) const {
        return self + m_byte_offsets[ix];
    }

    char cmp(instance_ptr left, instance_ptr right);

    template<class buf_t>
    void deserialize(instance_ptr self, buf_t& buffer) {
        for (long k = 0; k < m_members.size();k++) {
            bool isInitialized = buffer.read_uint8();
            if (isInitialized) {
                m_members[k].second->deserialize(eltPtr(self,k),buffer);
                setInitializationFlag(self, k);
            }
        }
    }

    template<class buf_t>
    void serialize(instance_ptr self, buf_t& buffer) {
        for (long k = 0; k < m_members.size();k++) {
            bool isInitialized = checkInitializationFlag(self, k);
            if (isInitialized) {
                buffer.write_uint8(true);
                m_members[k].second->serialize(eltPtr(self,k),buffer);
            } else {
                buffer.write_uint8(false);
            }
        }
    }

    void repr(instance_ptr self, std::ostringstream& stream);

    int32_t hash32(instance_ptr left);

    template<class sub_constructor>
    void constructor(instance_ptr self, const sub_constructor& initializer) const {
        for (int64_t k = 0; k < m_members.size(); k++) {
            try {
                initializer(eltPtr(self, k), k);
                setInitializationFlag(self, k);
            } catch(...) {
                for (long k2 = k-1; k2 >= 0; k2--) {
                    m_members[k2].second->destroy(eltPtr(self,k2));
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
        int byte = memberIndex / 8;
        int bit = memberIndex % 8;
        return bool( ((uint8_t*)self)[byte] & (1 << bit) );
    }

    void setInitializationFlag(instance_ptr self, int memberIndex, bool value=true) const;

    const std::vector<std::pair<std::string, Type*> >& getMembers() const {
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

    const std::vector<size_t>& getOffsets() const {
        return m_byte_offsets;
    }

    int memberNamed(const char* c) const;

private:
    std::vector<size_t> m_byte_offsets;

    std::vector<std::pair<std::string, Type*> > m_members;

    std::map<std::string, Function*> m_memberFunctions;
    std::map<std::string, Function*> m_staticFunctions;
    std::map<std::string, PyObject*> m_classMembers;
};

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

        forwardTypesMayHaveChanged();
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

    void _forwardTypesMayHaveChanged();

    static Class* Make(
            std::string inName,
            const std::vector<std::pair<std::string, Type*> >& members,
            const std::map<std::string, Function*>& memberFunctions,
            const std::map<std::string, Function*>& staticFunctions,
            const std::map<std::string, PyObject*>& classMembers
            )
    {
        return new Class(HeldClass::Make(inName, members, memberFunctions, staticFunctions, classMembers));
    }

    instance_ptr eltPtr(instance_ptr self, int64_t ix) const;

    void setAttribute(instance_ptr self, int64_t ix, instance_ptr elt) const;

    bool checkInitializationFlag(instance_ptr self, int64_t ix) const;

    char cmp(instance_ptr left, instance_ptr right);

    template<class buf_t>
    void deserialize(instance_ptr self, buf_t& buffer) {
        *(layout**)self = (layout*)malloc(
            sizeof(layout) + m_heldClass->bytecount()
            );

        layout& record = **(layout**)self;
        record.refcount = 1;

        m_heldClass->deserialize(record.data, buffer);
    }

    template<class buf_t>
    void serialize(instance_ptr self, buf_t& buffer) {
        layout& l = **(layout**)self;
        m_heldClass->serialize(l.data, buffer);
    }

    void repr(instance_ptr self, std::ostringstream& stream);

    int32_t hash32(instance_ptr left);

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

    void copy_constructor(instance_ptr self, instance_ptr other);

    void assign(instance_ptr self, instance_ptr other);

    const std::vector<std::pair<std::string, Type*> >& getMembers() const {
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

    int memberNamed(const char* c) const {
        return m_heldClass->memberNamed(c);
    }

    HeldClass* getHeldClass() const {
        return m_heldClass;
    }



private:
    HeldClass* m_heldClass;
};

class BoundMethod : public Class {
public:
    BoundMethod(Class* inClass, Function* inFunc) : Class(inClass->getHeldClass())
    {
        m_typeCategory = TypeCategory::catBoundMethod;
        m_function = inFunc;
        m_class = inClass;
        forwardTypesMayHaveChanged();
    }

    template<class visitor_type>
    void _visitContainedTypes(const visitor_type& visitor) {
        visitor(m_class);
    }

    template<class visitor_type>
    void _visitReferencedTypes(const visitor_type& visitor) {
        Type* c = m_class;
        Type* f = m_function;

        visitor(c);
        visitor(f);

        assert(c == m_class);
        assert(f == m_function);
    }

    void _forwardTypesMayHaveChanged() {
        m_name = "BoundMethod(" + m_class->name() + "." + m_function->name() + ")";
    }

    static BoundMethod* Make(Class* c, Function* f) {
        static std::mutex guard;

        std::lock_guard<std::mutex> lock(guard);

        typedef std::pair<Class*, Function*> keytype;

        static std::map<keytype, BoundMethod*> m;

        auto it = m.find(keytype(c,f));

        if (it == m.end()) {
            it = m.insert(
                std::make_pair(keytype(c,f), new BoundMethod(c, f))
                ).first;
        }

        return it->second;
    }

    void repr(instance_ptr self, std::ostringstream& stream) {
        stream << m_name;
    }

    Class* getClass() const {
        return m_class;
    }

    Function* getFunction() const {
        return m_function;
    }

private:
    Function* m_function;
    Class* m_class;
};



