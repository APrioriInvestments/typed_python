#pragma once

#include <Python.h>
#include <string>
#include <vector>
#include <mutex>
#include <set>
#include <utility>
#include <atomic>

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
class String;
class Bytes;
class Float32;
class Float64;
class OneOf;
class Value;
class TupleOf;
class NamedTuple;
class Tuple;
class ConstDict;
class Alternative;
class Class;
class PackedArray;
class Pointer;

typedef uint8_t* instance_ptr;

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
        catClass,
        catPackedArray,
        catPointer
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

    template<class T>
    auto check(const T& f) const -> decltype(f(*this)) {
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
            case catClass:
                return f(*(Class*)this);
            case catPackedArray:
                return f(*(PackedArray*)this);
            case catPointer:
                return f(*(Pointer*)this);
            default:
                throw std::runtime_error("Invalid type found");
        }
    }

    void constructor(instance_ptr self) const {
        this->check([&](auto& subtype) { subtype.constructor(self); } );
    }

    void destroy(instance_ptr self) const {
        this->check([&](auto& subtype) { subtype.destroy(self); } );
    }

    template<class ptr_func>
    void destroy(int64_t count, const ptr_func& ptrToChild) const {
        this->check([&](auto& subtype) { 
            for (long k = 0; k < count; k++) {
                subtype.destroy(ptrToChild(k)); 
            }
        });
    }

    void copy_constructor(instance_ptr self, instance_ptr other) const {
        this->check([&](auto& subtype) { subtype.copy_constructor(self, other); } );
    }

    void assign(instance_ptr self, instance_ptr other) const {
        this->check([&](auto& subtype) { subtype.assign(self, other); } );
    }

    PyTypeObject* getTypeRep() const {
        return mTypeRep;
    }

    void setTypeRep(PyTypeObject* o) {
        mTypeRep = o;
    }

    bool is_default_constructible() const {
        return m_is_default_constructible;
    }

protected:
    Type(TypeCategory in_typeCategory) : 
            m_typeCategory(in_typeCategory),
            m_size(0),
            m_is_default_constructible(false),
            m_name("Undefined"),
            mTypeRep(nullptr)
        {}

    TypeCategory m_typeCategory;
    
    size_t m_size;

    bool m_is_default_constructible;

    std::string m_name;

    PyTypeObject* mTypeRep;
};

class OneOf : public Type {
public:
    OneOf(const std::vector<Type*>& types) : 
                    Type(TypeCategory::catOneOf),
                    m_types(types)
    {   
        if (m_types.size() > 255) {
            throw std::runtime_error("Cant make a OneOf with more than 255 types.");
        }

        m_size = computeBytecount();
        
        m_name = "OneOf(...)";
        
        m_is_default_constructible = false;

        for (auto typePtr: m_types) {
            if (typePtr->is_default_constructible()) {
                m_is_default_constructible = true;
                break;
            }
        }
    }

    std::pair<Type*, instance_ptr> unwrap(instance_ptr self) const {
        return std::make_pair(m_types[*(uint8_t*)self], self+1);
    }

    size_t computeBytecount() const { 
        size_t res = 0;
        
        for (auto t: m_types)
            res = std::max(res, t->bytecount());

        return res + 1;
    }

    void constructor(instance_ptr self) const {
        if (!m_is_default_constructible) {
            throw std::runtime_error(m_name + " is not default-constructible");
        }

        for (size_t k = 0; k < m_types.size(); k++) {
            if (m_types[k]->is_default_constructible()) {
                *(uint8_t*)self = k;
                m_types[k]->constructor(self+1);
            }
        }
    }

    void destroy(instance_ptr self) const {
        uint8_t which = *(uint8_t*)(self);
        m_types[which]->destroy(self+1);
    }

    void copy_constructor(instance_ptr self, instance_ptr other) const {
        uint8_t which = *(uint8_t*)self = *(uint8_t*)other;
        m_types[which]->copy_constructor(self+1, other+1);
    }

    void assign(instance_ptr self, instance_ptr other) const {
        uint8_t which = *(uint8_t*)self;
        if (which == *(uint8_t*)other) {
            m_types[which]->assign(self+1,other+1);
        } else {
            m_types[which]->destroy(self+1);

            uint8_t otherWhich = *(uint8_t*)other;
            *(uint8_t*)self = otherWhich;
            m_types[otherWhich]->copy_constructor(self+1,other+1);
        }
    }
    
    const std::vector<Type*>& getTypes() const {
        return m_types;
    }

    static OneOf* Make(const std::vector<Type*>& types) {
        std::vector<Type*> flat_typelist;
        std::set<Type*> seen;

        //make sure we only get each type once and don't have any other 'OneOf' in there...
        std::function<void (const std::vector<Type*>)> visit = [&](const std::vector<Type*>& subvec) {
            for (auto t: subvec) {
                if (t->getTypeCategory() == catOneOf) {
                    visit( ((OneOf*)t)->getTypes() );
                } else if (seen.find(t) == seen.end()) {
                    flat_typelist.push_back(t);
                    seen.insert(t);
                }
            }
        };

        visit(types);

        static std::mutex guard;

        std::lock_guard<std::mutex> lock(guard);

        typedef const std::vector<Type*> keytype;

        static std::map<keytype, OneOf*> m;

        auto it = m.find(types);
        if (it == m.end()) {
            it = m.insert(std::make_pair(types, new OneOf(types))).first;
        }

        return it->second;
    }

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
        m_is_default_constructible = true;

        m_size = 0;
        for (auto t: m_types) {
            m_byte_offsets.push_back(m_size);
            m_size += t->bytecount();
        }

        for (auto t: m_types) {
            if (!t->is_default_constructible()) {
                m_is_default_constructible = false;
            }
        }
    }

    instance_ptr eltPtr(instance_ptr self, int64_t ix) const {
        return self + m_byte_offsets[ix];
    }

    template<class sub_constructor>
    void constructor(instance_ptr self, const sub_constructor& initializer) const {
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

    void constructor(instance_ptr self) const {
        if (!m_is_default_constructible) {
            throw std::runtime_error(m_name + " is not default-constructible");
        }

        for (size_t k = 0; k < m_types.size(); k++) {
            m_types[k]->constructor(self+m_byte_offsets[k]);
        }
    }

    void destroy(instance_ptr self) const {
        for (long k = (long)m_types.size() - 1; k >= 0; k--) {
            m_types[k]->destroy(self+m_byte_offsets[k]);
        }
    }

    void copy_constructor(instance_ptr self, instance_ptr other) const {
        for (long k = (long)m_types.size() - 1; k >= 0; k--) {
            m_types[k]->copy_constructor(self + m_byte_offsets[k], other+m_byte_offsets[k]);
        }
    }

    void assign(instance_ptr self, instance_ptr other) const {
        for (long k = (long)m_types.size() - 1; k >= 0; k--) {
            m_types[k]->assign(self + m_byte_offsets[k], other+m_byte_offsets[k]);
        }
    }
    
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
        m_name = "NamedTuple(...)";
    }

    static NamedTuple* Make(const std::vector<Type*>& types, const std::vector<std::string>& names) {
        return MakeSubtype<NamedTuple>(types,names);
    }
};

class Tuple : public CompositeType {
public:
    Tuple(const std::vector<Type*>& types, const std::vector<std::string>& names) : 
            CompositeType(TypeCategory::catTuple, types, names)
    {
        m_name = "Tuple(...)";
    }

    static Tuple* Make(const std::vector<Type*>& types) {
        return MakeSubtype<Tuple>(types, std::vector<std::string>());
    }
};

class TupleOf : public Type {
    class layout {
    public:
        std::atomic<int64_t> refcount;
        int64_t count;
        uint8_t data[];
    };

public:
    TupleOf(Type* type) : 
            Type(TypeCategory::catTupleOf),
            m_element_type(type)
    {
        m_name = "TupleOf(...)";
        m_size = sizeof(void*);
    }

    Type* getEltType() const {
        return m_element_type;
    }

    static TupleOf* Make(Type* elt) {
        static std::mutex guard;

        std::lock_guard<std::mutex> lock(guard);

        static std::map<Type*, TupleOf*> m;

        auto it = m.find(elt);
        if (it == m.end()) {
            it = m.insert(std::make_pair(elt, new TupleOf(elt))).first;
        }

        return it->second;
    };

    instance_ptr eltPtr(instance_ptr self, int64_t i) const {
        return (*(layout**)self)->data + i * m_element_type->bytecount();
    }

    int64_t count(instance_ptr self) const {
        return (*(layout**)self)->count;
    }

    template<class sub_constructor>
    void constructor(instance_ptr self, int64_t count, const sub_constructor& allocator) const {
        (*(layout**)self) = (layout*)malloc(sizeof(layout) + getEltType()->bytecount() * count);

        (*(layout**)self)->count = count;
        (*(layout**)self)->refcount = 1;
        
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

    void constructor(instance_ptr self) const {
        constructor(self, 0, [](instance_ptr i, int64_t k) {});
    }

    void destroy(instance_ptr self) const {
        (*(layout**)self)->refcount--;
        if ((*(layout**)self)->refcount == 0) {
            m_element_type->destroy((*(layout**)self)->count, [&](int64_t k) {return eltPtr(self,k);});
            free((*(layout**)self));
        }
    }

    void copy_constructor(instance_ptr self, instance_ptr other) const {
        (*(layout**)self) = (*(layout**)other);
        (*(layout**)self)->refcount++;
    }

    void assign(instance_ptr self, instance_ptr other) const {
        layout* old = (*(layout**)self);

        (*(layout**)self) = (*(layout**)other);
        (*(layout**)self)->refcount++;

        destroy((instance_ptr)&old);
    }

private:
    Type* m_element_type;
};



class ConstDict : public Type {
public:
    ConstDict(Type* key, Type* value) : 
            Type(TypeCategory::catConstDict),
            m_key(key),
            m_value(value)
    {
        m_name = "ConstDict(...)";
        m_size = sizeof(void*);
    }

    static ConstDict* Make(Type* key, Type* value) {
        static std::mutex guard;

        std::lock_guard<std::mutex> lock(guard);

        static std::map<std::pair<Type*, Type*>, ConstDict*> m;

        auto lookup_key = std::make_pair(key,value);

        auto it = m.find(lookup_key);
        if (it == m.end()) {
            it = m.insert(std::make_pair(lookup_key, new ConstDict(key, value))).first;
        }

        return it->second;
    };

    void constructor(instance_ptr self) const {
    
    }
    
    void destroy(instance_ptr self) const {

    }

    void copy_constructor(instance_ptr self, instance_ptr other) const {

    }

    void assign(instance_ptr self, instance_ptr other) const {

    }
    

private:
    Type* m_key;
    Type* m_value;
};

class None : public Type {
public:
    None() : Type(TypeCategory::catNone)
    {
        m_name = "NoneType";
        m_size = 0;
        m_is_default_constructible = true;
    }

    void constructor(instance_ptr self) const {}

    void destroy(instance_ptr self) const {}

    void copy_constructor(instance_ptr self, instance_ptr other) const {}

    void assign(instance_ptr self, instance_ptr other) const {}

    static None* Make() { static None res; return &res; }

};

template<class T>
class RegisterType : public Type {
public:
    RegisterType(TypeCategory kind) : Type(kind) 
    {
        m_size = sizeof(T);
        m_is_default_constructible = true;
    }

    void constructor(instance_ptr self) const {
        new ((T*)self) T();
    }

    void destroy(instance_ptr self) const {}

    void copy_constructor(instance_ptr self, instance_ptr other) const {
        *((T*)self) = *((T*)other);
    }

    void assign(instance_ptr self, instance_ptr other) const {
        *((T*)self) = *((T*)other);
    }
};

class Bool : public RegisterType<bool> {
public:
    Bool() : RegisterType(TypeCategory::catBool)
    {
        m_name = "Bool";
    }

    static Bool* Make() { static Bool res; return &res; }
};

class UInt8 : public RegisterType<uint8_t> {
public:
    UInt8() : RegisterType(TypeCategory::catUInt8)
    {
        m_name = "UInt8";
    }

    static UInt8* Make() { static UInt8 res; return &res; }
};

class UInt16 : public RegisterType<uint16_t> {
public:
    UInt16() : RegisterType(TypeCategory::catUInt16)
    {
        m_name = "UInt16";
    }

    static UInt16* Make() { static UInt16 res; return &res; }
};

class UInt32 : public RegisterType<uint32_t> {
public:
    UInt32() : RegisterType(TypeCategory::catUInt32)
    {
        m_name = "UInt32";
    }

    static UInt32* Make() { static UInt32 res; return &res; }
};

class UInt64 : public RegisterType<uint64_t> {
public:
    UInt64() : RegisterType(TypeCategory::catUInt64)
    {
        m_name = "UInt64";
    }

    static UInt64* Make() { static UInt64 res; return &res; }
};

class Int8 : public RegisterType<int8_t> {
public:
    Int8() : RegisterType(TypeCategory::catInt8)
    {
        m_name = "Int8";
        m_size = 1;
    }

    static Int8* Make() { static Int8 res; return &res; }
};

class Int16 : public RegisterType<int16_t> {
public:
    Int16() : RegisterType(TypeCategory::catInt16)
    {
        m_name = "Int16";
    }

    static Int16* Make() { static Int16 res; return &res; }
};

class Int32 : public RegisterType<int32_t> {
public:
    Int32() : RegisterType(TypeCategory::catInt32)
    {
        m_name = "Int32";
    }

    static Int32* Make() { static Int32 res; return &res; }
};

class Int64 : public RegisterType<int64_t> {
public:
    Int64() : RegisterType(TypeCategory::catInt64)
    {
        m_name = "Int64";
    }

    static Int64* Make() { static Int64 res; return &res; }
};

class Float32 : public RegisterType<float> {
public:
    Float32() : RegisterType(TypeCategory::catFloat32)
    {
        m_name = "Float32";
    }

    static Float32* Make() { static Float32 res; return &res; }
};

class Float64 : public RegisterType<double> {
public:
    Float64() : RegisterType(TypeCategory::catFloat64)
    {
        m_name = "Float64";
    }

    static Float64* Make() { static Float64 res; return &res; }
};

class String : public Type {
public:
    class layout {
    public:
        std::atomic<int64_t> refcount;
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

    static String* Make() { static String res; return &res; }

    void constructor(instance_ptr self, int64_t bytes_per_codepoint, int64_t count, const char* data) const {
        if (count == 0) {
            *(layout**)self = nullptr;
            return;
        }

        (*(layout**)self) = (layout*)malloc(sizeof(layout) + count * bytes_per_codepoint);

        (*(layout**)self)->bytes_per_codepoint = bytes_per_codepoint;
        (*(layout**)self)->pointcount = count;
        (*(layout**)self)->refcount = 1;
        
        ::memcpy((*(layout**)self)->data, data, count * bytes_per_codepoint);
    }

    instance_ptr eltPtr(instance_ptr self, int64_t i) const {
        if (*(layout**)self == nullptr) { 
            return self;
        }

        return (*(layout**)self)->eltPtr(i);
    }

    int64_t bytes_per_codepoint(instance_ptr self) const {
        if (*(layout**)self == nullptr) { 
            return 1; 
        }

        return (*(layout**)self)->bytes_per_codepoint;
    }

    int64_t count(instance_ptr self) const {
        if (*(layout**)self == nullptr) { 
            return 0; 
        }

        return (*(layout**)self)->pointcount;
    }

    void constructor(instance_ptr self) const {
        *(layout**)self = 0;
    }

    void destroy(instance_ptr self) const {
        if (!*(layout**)self) {
            return;
        }

        (*(layout**)self)->refcount--;

        if ((*(layout**)self)->refcount == 0) {
            free((*(layout**)self));
        }
    }

    void copy_constructor(instance_ptr self, instance_ptr other) const {
        (*(layout**)self) = (*(layout**)other);
        if (*(layout**)self) {
            (*(layout**)self)->refcount++;
        }
    }

    void assign(instance_ptr self, instance_ptr other) const {
        layout* old = (*(layout**)self);

        (*(layout**)self) = (*(layout**)other);
        
        if (*(layout**)self) {
            (*(layout**)self)->refcount++;
        }

        destroy((instance_ptr)&old);
    }
};

class Bytes : public Type {
public:
    class layout {
    public:
        std::atomic<int64_t> refcount;
        int64_t bytecount;
        uint8_t data[];
    };

    Bytes() : Type(TypeCategory::catBytes)
    {
        m_name = "Bytes";
        m_is_default_constructible = true;
        m_size = sizeof(layout*);
    }

    static Bytes* Make() { static Bytes res; return &res; }

    void constructor(instance_ptr self, int64_t count, const char* data) const {
        (*(layout**)self) = (layout*)malloc(sizeof(layout) + count);

        (*(layout**)self)->bytecount = count;
        (*(layout**)self)->refcount = 1;
        
        ::memcpy((*(layout**)self)->data, data, count);
    }

    instance_ptr eltPtr(instance_ptr self, int64_t i) const {
        //we don't want to have to return null here, but there is no actual memory to back this.
        if (*(layout**)self == nullptr) {
            return self;
        }

        return (*(layout**)self)->data + i;
    }

    int64_t count(instance_ptr self) const {
        if (*(layout**)self == nullptr) { 
            return 0; 
        }

        return (*(layout**)self)->bytecount;
    }

    void constructor(instance_ptr self) const {
        *(layout**)self = 0;
    }

    void destroy(instance_ptr self) const {
        if (!*(layout**)self) {
            return;
        }

        (*(layout**)self)->refcount--;

        if ((*(layout**)self)->refcount == 0) {
            free((*(layout**)self));
        }
    }

    void copy_constructor(instance_ptr self, instance_ptr other) const {
        (*(layout**)self) = (*(layout**)other);
        if (*(layout**)self) {
            (*(layout**)self)->refcount++;
        }
    }

    void assign(instance_ptr self, instance_ptr other) const {
        layout* old = (*(layout**)self);

        (*(layout**)self) = (*(layout**)other);
        
        if (*(layout**)self) {
            (*(layout**)self)->refcount++;
        }

        destroy((instance_ptr)&old);
    }
};

class Value : public Type {
public:
    Value(Type* type, std::string data) : 
            Type(TypeCategory::catValue),
            m_type(type),
            m_data(data)
        {
        m_size = 0;
        m_is_default_constructible = true;
        m_name = "Value(...)";
        }

    void constructor(instance_ptr self) const {}

    void destroy(instance_ptr self) const {}

    void copy_constructor(instance_ptr self, instance_ptr other) const {}

    void assign(instance_ptr self, instance_ptr other) const {}

    std::pair<Type*, instance_ptr> unwrap() const {
        return std::make_pair(m_type, (instance_ptr)&m_data[0]);
    }

    //when we have an ability to compare instances directly, we'll implement this...
    //static None* Make() { static None res; return &res; }

    static Value* ForNone() { static Value res(None::Make(), ""); return &res; }

private:
    Type* m_type;
    std::string m_data;
};

class Alternative : public Type {
public:
    void constructor(instance_ptr self) const {
    
    }
    
    void destroy(instance_ptr self) const {

    }

    void copy_constructor(instance_ptr self, instance_ptr other) const {

    }

    void assign(instance_ptr self, instance_ptr other) const {

    }
};

class Class : public Type {
public:
    void constructor(instance_ptr self) const {
    
    }
    
    void destroy(instance_ptr self) const {

    }

    void copy_constructor(instance_ptr self, instance_ptr other) const {

    }

    void assign(instance_ptr self, instance_ptr other) const {

    }
};

class PackedArray : public Type {
public:
    void constructor(instance_ptr self) const {
    
    }
    
    void destroy(instance_ptr self) const {

    }

    void copy_constructor(instance_ptr self, instance_ptr other) const {

    }

    void assign(instance_ptr self, instance_ptr other) const {

    }
};

class Pointer : public Type {
public:
    void constructor(instance_ptr self) const {
    
    }
    
    void destroy(instance_ptr self) const {

    }

    void copy_constructor(instance_ptr self, instance_ptr other) const {

    }

    void assign(instance_ptr self, instance_ptr other) const {

    }
};


