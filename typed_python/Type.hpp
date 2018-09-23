#pragma once

#include <string>
#include <vector>
#include <mutex>
#include <utility>

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
class Long;
class String;
class Bytes;
class Float32;
class Float64;
class OneOf;
class Value;
class TupleOf;
class NamedTuple;
class Kwargs;
class Tuple;
class ConstDict;
class Alternative;
class Class;
class PackedArray;
class Pointer;

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
        catLong,
        catString,
        catBytes,
        catFloat32,
        catFloat64,
        catValue,
        catOneOf,
        catTupleOf,
        catNamedTuple,
        catKwargs,
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
            case catLong:
                return f(*(Long*)this);
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
            case catKwargs:
                return f(*(Kwargs*)this);
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

protected:
    Type(TypeCategory in_typeCategory) : 
            m_typeCategory(in_typeCategory),
            m_size(0),
            m_name("Undefined")
        {}

    TypeCategory m_typeCategory;
    
    size_t m_size;

    std::string m_name;
};

class Value : public Type {
public:
    Value(Type* type, std::string data) : 
            Type(TypeCategory::catValue),
            m_type(type),
            m_data(data)
        {
        m_size = 0;
        m_name = "Value(...)";
        }

private:
    Type* m_type;
    std::string m_data;
};

class OneOf : public Type {
public:
    OneOf(const std::vector<Type*>& types) : 
            Type(TypeCategory::catOneOf),
            m_types(types)
        {
        m_size = computeBytecount();
        m_name = "OneOf(...)";
        }

    size_t computeBytecount() const { 
        size_t res = 0;
        for (auto t: m_types)
            res = std::max(res, t->bytecount());
        return res + 1;
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
        m_size = computeBytecount();
    }

    size_t computeBytecount() const { 
        size_t res = 0;
        for (auto t: m_types)
            res += t->bytecount();
        return res;
    }

protected:
    template<class subtype>
    static subtype* MakeSubtype(const std::vector<Type*>& types, const std::vector<std::string>& names) {
        static std::mutex guard;

        std::lock_guard<std::mutex> lock(guard);

        typedef std::pair<const std::vector<Type*>&, const std::vector<std::string>& > keytype;

        static std::map<keytype, subtype*> m;

        auto it = m.find(keytype(types,names));
        if (it == m.end()) {
            it = m.insert(std::make_pair(keytype(types,names), new subtype(types,names))).first;
        }

        return it->second;
    }

    std::vector<Type*> m_types;
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

class Kwargs : public CompositeType {
public:
    Kwargs(const std::vector<Type*>& types, const std::vector<std::string>& names) : 
            CompositeType(TypeCategory::catKwargs, types, names)
    {
        m_name = "Kwargs(...)";
    }

    static Kwargs* Make(const std::vector<Type*>& types, const std::vector<std::string>& names) {
        return MakeSubtype<Kwargs>(types,names);
    }
};

class Tuple : public CompositeType {
public:
    Tuple(const std::vector<Type*>& types, const std::vector<std::string>& names) : 
            CompositeType(TypeCategory::catTuple, types, names)
    {
        m_name = "Tuple(...)";
    }

    static Tuple* Make(const std::vector<Type*>& types, const std::vector<std::string>& names) {
        return MakeSubtype<Tuple>(types,names);
    }
};

class TupleOf : public Type {
public:
    TupleOf(Type* type) : 
            Type(TypeCategory::catTupleOf),
            m_element_type(type)
    {
        m_name = "TupleOf(...)";
        m_size = computeBytecount();
    }

    size_t computeBytecount() const { 
        return sizeof(void*); 
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

private:
    Type* m_key;
    Type* m_value;
};

class PrimitiveType : public Type {
public:
    PrimitiveType(TypeCategory kind) : Type(kind) 
        {}
};

class None : public PrimitiveType {
public:
    None() : PrimitiveType(TypeCategory::catNone)
    {
        m_name = "NoneType";
        m_size = 0;
    }

    static None* Make() { static None res; return &res; }

};

class Bool : public PrimitiveType {
public:
    Bool() : PrimitiveType(TypeCategory::catBool)
    {
        m_name = "Bool";
        m_size = 1;
    }

    static Bool* Make() { static Bool res; return &res; }
};

class UInt8 : public PrimitiveType {
public:
    UInt8() : PrimitiveType(TypeCategory::catUInt8)
    {
        m_name = "UInt8";
        m_size = 1;
    }

    static UInt8* Make() { static UInt8 res; return &res; }
};

class UInt16 : public PrimitiveType {
public:
    UInt16() : PrimitiveType(TypeCategory::catUInt16)
    {
        m_name = "UInt16";
        m_size = 2;
    }

    static UInt16* Make() { static UInt16 res; return &res; }
};

class UInt32 : public PrimitiveType {
public:
    UInt32() : PrimitiveType(TypeCategory::catUInt32)
    {
        m_name = "UInt32";
        m_size = 4;
    }

    static UInt32* Make() { static UInt32 res; return &res; }
};

class UInt64 : public PrimitiveType {
public:
    UInt64() : PrimitiveType(TypeCategory::catUInt64)
    {
        m_name = "UInt64";
        m_size = 8;
    }

    static UInt64* Make() { static UInt64 res; return &res; }
};

class Int8 : public PrimitiveType {
public:
    Int8() : PrimitiveType(TypeCategory::catInt8)
    {
        m_name = "Int8";
        m_size = 1;
    }

    static Int8* Make() { static Int8 res; return &res; }
};

class Int16 : public PrimitiveType {
public:
    Int16() : PrimitiveType(TypeCategory::catInt16)
    {
        m_name = "Int16";
        m_size = 2;
    }

    static Int16* Make() { static Int16 res; return &res; }
};

class Int32 : public PrimitiveType {
public:
    Int32() : PrimitiveType(TypeCategory::catInt32)
    {
        m_name = "Int32";
        m_size = 4;
    }

    static Int32* Make() { static Int32 res; return &res; }
};

class Int64 : public PrimitiveType {
public:
    Int64() : PrimitiveType(TypeCategory::catInt64)
    {
        m_name = "Int64";
        m_size = 8;
    }

    static Int64* Make() { static Int64 res; return &res; }
};

class Long : public Type {
public:
    Long() : Type(TypeCategory::catLong)
    {
        m_name = "Long";
        m_size = sizeof(void*);
    }

    static Long* Make() { static Long res; return &res; }
};

class Float32 : public PrimitiveType {
public:
    Float32() : PrimitiveType(TypeCategory::catFloat32)
    {
        m_name = "Float32";
        m_size = 4;
    }

    static Float32* Make() { static Float32 res; return &res; }
};

class Float64 : public PrimitiveType {
public:
    Float64() : PrimitiveType(TypeCategory::catFloat64)
    {
        m_name = "Float64";
        m_size = 8;
    }

    static Float64* Make() { static Float64 res; return &res; }
};

class String : public PrimitiveType {
public:
    String() : PrimitiveType(TypeCategory::catString)
    {
        m_name = "String";
        m_size = sizeof(void*);
    }

    static String* Make() { static String res; return &res; }
};

class Bytes : public PrimitiveType {
public:
    Bytes() : PrimitiveType(TypeCategory::catBytes)
    {
        m_name = "Bytes";
        m_size = sizeof(void*);
    }

    static Bytes* Make() { static Bytes res; return &res; }
};

class Alternative : public Type {};
class Class : public Type {};
class PackedArray : public Type {};
class Pointer : public Type {};

