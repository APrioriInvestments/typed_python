#pragma once

#include <Python.h>
#include <string>
#include <vector>
#include <algorithm>
#include <functional>
#include <mutex>
#include <set>
#include <utility>
#include <atomic>
#include <iostream>
#include <sstream>

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
class ConcreteAlternative;
class PythonSubclass;
class Class;
class Function;
class PackedArray;
class Pointer;

typedef uint8_t* instance_ptr;

class Hash32Accumulator {
public:
    Hash32Accumulator(int32_t init) : 
        m_state(init) 
    {
    }

    void add(int32_t i) {
        m_state = (m_state * 1000003) ^ i;
    }

    void addBytes(uint8_t* bytes, int64_t count) {
        while (count >= 4) {
            add(*(int32_t*)bytes);
            bytes += 4;
            count -= 4;
        }
        while (count) {
            add((uint8_t)*bytes);
            bytes++;
            count--;
        }
    }

    int32_t get() const {
        return m_state;
    }

    void addRegister(bool i) { add(i ? 1:0); }
    void addRegister(uint8_t i) { add(i); }
    void addRegister(uint16_t i) { add(i); }
    void addRegister(uint32_t i) { add(i); }
    void addRegister(uint64_t i) { addBytes((uint8_t*)&i, sizeof(i)); }

    void addRegister(int8_t i) { add(i); }
    void addRegister(int16_t i) { add(i); }
    void addRegister(int32_t i) { add(i); }
    void addRegister(int64_t i) { addBytes((uint8_t*)&i, sizeof(i)); }

    void addRegister(float i) { addBytes((uint8_t*)&i, sizeof(i)); }
    void addRegister(double i) { addBytes((uint8_t*)&i, sizeof(i)); }

private:
    int32_t m_state;
};

class SerializationBuffer {
public:
    SerializationBuffer() :
            m_buffer(nullptr),
            m_size(0),
            m_reserved(0)
    {
    }

    ~SerializationBuffer() {
        if (m_buffer) {
            free(m_buffer);
        }
    }

    SerializationBuffer(const SerializationBuffer&) = delete;
    SerializationBuffer& operator=(const SerializationBuffer&) = delete;

    void write_uint8(uint8_t i) {
        ensure(sizeof(i));
        m_buffer[m_size++] = i;
    }

    void write_uint32(uint32_t i) {
        ensure(sizeof(i));
        *(uint32_t*)(m_buffer+m_size) = i;
        m_size += sizeof(i);
    }

    void write_bytes(uint8_t* ptr, size_t bytecount) {
        ensure(bytecount);
        memcpy(m_buffer+m_size,ptr,bytecount);
        m_size += bytecount;
    }

    uint8_t* buffer() const {
        return m_buffer;
    }

    size_t size() const {
        return m_size;
    }

    void ensure(size_t t) {
        if (m_size + t > m_reserved) {
            m_reserved = m_size + t + 1024 * 128;
            m_buffer = (uint8_t*)::realloc(m_buffer, m_reserved);
        }
    }

private:
    uint8_t* m_buffer;
    size_t m_size;
    size_t m_reserved;
};

class DeserializationBuffer {
public:
    DeserializationBuffer(uint8_t* ptr, size_t sz) :
            m_buffer(ptr),
            m_size(sz),
            m_orig_size(sz)
    {
    }

    DeserializationBuffer(const SerializationBuffer&) = delete;
    DeserializationBuffer& operator=(const SerializationBuffer&) = delete;

    uint8_t read_uint8() {
        if (m_size < sizeof(uint8_t)) {
            throw std::runtime_error("out of data");
        }
        uint8_t* ptr = (uint8_t*)m_buffer;

        m_size -= sizeof(uint8_t);
        m_buffer += sizeof(uint8_t);
        
        return *ptr;
    }

    uint32_t read_uint32() {
        if (m_size < sizeof(uint32_t)) {
            throw std::runtime_error("out of data");
        }
        uint32_t* ptr = (uint32_t*)m_buffer;

        m_size -= sizeof(uint32_t);
        m_buffer += sizeof(uint32_t);
        
        return *ptr;
    }

    void read_bytes(uint8_t* ptr, size_t bytecount) {
        if (m_size < bytecount) {
            throw std::runtime_error("out of data");
        }
        memcpy(ptr,m_buffer,bytecount);

        m_size -= bytecount;
        m_buffer += bytecount;
    }

    size_t remaining() const {
        return m_size;
    }

    size_t pos() const {
        return m_orig_size - m_size;
    }

private:
    uint8_t* m_buffer;
    size_t m_size;
    size_t m_orig_size;
};

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
        catPythonSubclass,
        catClass,
        catFunction,
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

    const Type* pickConcreteSubclass(instance_ptr data) const {
        return this->check([&](auto& subtype) {
            return subtype.pickConcreteSubclassConcrete(data);
        });
    }

    const Type* pickConcreteSubclassConcrete(instance_ptr data) const {
        return this;
    }

    void repr(instance_ptr self, std::ostringstream& out) const {
        this->check([&](auto& subtype) {
            subtype.repr(self, out);
        });
    }

    char cmp(instance_ptr left, instance_ptr right) const {
        return this->check([&](auto& subtype) {
            return subtype.cmp(left, right);
        });
    }

    int32_t hash32(instance_ptr left) const {
        return this->check([&](auto& subtype) {
            return subtype.hash32(left);
        });
    }

    template<class buf_t>
    void serialize(instance_ptr left, buf_t& buffer) const {
        return this->check([&](auto& subtype) {
            return subtype.serialize(left, buffer);
        });
    }

    template<class buf_t>
    void deserialize(instance_ptr left, buf_t& buffer) const {
        return this->check([&](auto& subtype) {
            return subtype.deserialize(left, buffer);
        });
    }

    void swap(instance_ptr left, instance_ptr right) const {
        if (left == right) {
            return;
        }

        size_t remaining = m_size;
        while (remaining >= 8) {
            int64_t temp = *(int64_t*)left;
            *(int64_t*)left = *(int64_t*)right;
            *(int64_t*)right = temp;

            remaining -= 8;
            left += 8;
            right += 8;
        }

        while (remaining > 0) {
            int8_t temp = *(int8_t*)left;
            *(int8_t*)left = *(int8_t*)right;
            *(int8_t*)right = temp;

            remaining -= 1;
            left += 1;
            right += 1;
        }
    }

    static char byteCompare(uint8_t* l, uint8_t* r, size_t count) {
        while (count >= 8 && *(uint64_t*)l == *(uint64_t*)r) {
            l += 8;
            r += 8;
            count -= 8;
        }

        for (long k = 0; k < count; k++) {
            if (l[k] < r[k]) {
                return -1;
            }
            if (l[k] > r[k]) {
                return 1;
            }
        }
        return 0;
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
            case catConcreteAlternative:
                return f(*(ConcreteAlternative*)this);
            case catPythonSubclass:
                return f(*(PythonSubclass*)this);
            case catClass:
                return f(*(Class*)this);
            case catFunction:
                return f(*(Function*)this);
            case catPackedArray:
                return f(*(PackedArray*)this);
            case catPointer:
                return f(*(Pointer*)this);
            default:
                throw std::runtime_error("Invalid type found");
        }
    }

    const Type* getBaseType() const {
        return m_base;
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

    void setTypeRep(PyTypeObject* o) const {
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
            mTypeRep(nullptr),
            m_base(nullptr)
        {}

    TypeCategory m_typeCategory;
    
    size_t m_size;

    bool m_is_default_constructible;

    std::string m_name;

    mutable PyTypeObject* mTypeRep;

    const Type* m_base;
};

class OneOf : public Type {
public:
    OneOf(const std::vector<const Type*>& types) : 
                    Type(TypeCategory::catOneOf),
                    m_types(types)
    {   
        if (m_types.size() > 255) {
            throw std::runtime_error("Cant make a OneOf with more than 255 types.");
        }

        m_size = computeBytecount();
        
        m_name = "OneOf(";
        bool first = true;
        for (auto t: types) {
            if (first) { 
                first = false;
            } else {
                m_name += ", ";
            }

            m_name += t->name();
        }

        m_name += ")";
        
        m_is_default_constructible = false;

        for (auto typePtr: m_types) {
            if (typePtr->is_default_constructible()) {
                m_is_default_constructible = true;
                break;
            }
        }
    }

    template<class buf_t>
    void deserialize(instance_ptr self, buf_t& buffer) const {
        uint8_t which = buffer.read_uint8();
        if (which >= m_types.size()) {
            throw std::runtime_error("Corrupt OneOf");
        }
        *(uint8_t*)self = which;
        m_types[which]->deserialize(self+1, buffer);
    }

    template<class buf_t>
    void serialize(instance_ptr self, buf_t& buffer) const {
        buffer.write_uint8(*(uint8_t*)self);
        m_types[*((uint8_t*)self)]->serialize(self+1, buffer);
    }

    void repr(instance_ptr self, std::ostringstream& stream) const {
        m_types[*((uint8_t*)self)]->repr(self+1, stream);
    }

    int32_t hash32(instance_ptr left) const {
        Hash32Accumulator acc((int)getTypeCategory());

        acc.add(*(uint8_t*)left);
        acc.add(m_types[*((uint8_t*)left)]->hash32(left+1));

        return acc.get();
    }

    char cmp(instance_ptr left, instance_ptr right) const {
        if (((uint8_t*)left)[0] < ((uint8_t*)right)[0]) {
            return -1;
        }
        if (((uint8_t*)left)[0] > ((uint8_t*)right)[0]) {
            return 1;
        }

        return m_types[*((uint8_t*)left)]->cmp(left+1,right+1);
    }

    std::pair<const Type*, instance_ptr> unwrap(instance_ptr self) const {
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
                return;
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
    
    const std::vector<const Type*>& getTypes() const {
        return m_types;
    }

    static OneOf* Make(const std::vector<const Type*>& types) {
        std::vector<const Type*> flat_typelist;
        std::set<const Type*> seen;

        //make sure we only get each type once and don't have any other 'OneOf' in there...
        std::function<void (const std::vector<const Type*>)> visit = [&](const std::vector<const Type*>& subvec) {
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

        typedef const std::vector<const Type*> keytype;

        static std::map<keytype, OneOf*> m;

        auto it = m.find(flat_typelist);
        if (it == m.end()) {
            it = m.insert(std::make_pair(flat_typelist, new OneOf(flat_typelist))).first;
        }

        return it->second;
    }

private:
    std::vector<const Type*> m_types;
};

class CompositeType : public Type {
public:
    CompositeType(
                TypeCategory in_typeCategory,
                const std::vector<const Type*>& types,
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

    char cmp(instance_ptr left, instance_ptr right) const {
        for (long k = 0; k < m_types.size(); k++) {
            char res = m_types[k]->cmp(left + m_byte_offsets[k], right + m_byte_offsets[k]);
            if (res != 0) {
                return res;
            }
        }

        return 0;
    }

    template<class buf_t>
    void deserialize(instance_ptr self, buf_t& buffer) const {
        for (long k = 0; k < getTypes().size();k++) {
            getTypes()[k]->deserialize(eltPtr(self,k),buffer);
        }
    }

    template<class buf_t>
    void serialize(instance_ptr self, buf_t& buffer) const {
        for (long k = 0; k < getTypes().size();k++) {
            getTypes()[k]->serialize(eltPtr(self,k),buffer);
        }
    }

    void repr(instance_ptr self, std::ostringstream& stream) const {
        stream << "(";

        for (long k = 0; k < getTypes().size();k++) {
            if (k > 0) {
                stream << ", ";
            }

            if (k < m_names.size()) {
                stream << m_names[k] << "=";
            }

            getTypes()[k]->repr(eltPtr(self,k),stream);
        }
        if (getTypes().size() == 1) {
            stream << ",";
        }

        stream << ")";
    }

    int32_t hash32(instance_ptr left) const {
        Hash32Accumulator acc((int)getTypeCategory());

        for (long k = 0; k < getTypes().size();k++) {
            acc.add(getTypes()[k]->hash32(eltPtr(left,k)));
        }

        acc.add(getTypes().size());

        return acc.get();
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
    
    const std::vector<const Type*>& getTypes() const {
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
    static subtype* MakeSubtype(const std::vector<const Type*>& types, const std::vector<std::string>& names) {
        static std::mutex guard;

        std::lock_guard<std::mutex> lock(guard);

        typedef std::pair<const std::vector<const Type*>, const std::vector<std::string> > keytype;

        static std::map<keytype, subtype*> m;

        auto it = m.find(keytype(types,names));
        if (it == m.end()) {
            it = m.insert(std::make_pair(keytype(types,names), new subtype(types,names))).first;
        }

        return it->second;
    }

    std::vector<const Type*> m_types;
    std::vector<size_t> m_byte_offsets;
    std::vector<std::string> m_names;
};

class NamedTuple : public CompositeType {
public:
    NamedTuple(const std::vector<const Type*>& types, const std::vector<std::string>& names) : 
            CompositeType(TypeCategory::catNamedTuple, types, names)
    {
        assert(types.size() == names.size());

        m_name = "NamedTuple(";
        for (long k = 0; k < types.size();k++) {
            if (k) {
                m_name += ", ";
            }
            m_name += names[k] + "=" + types[k]->name();
        }
        m_name += ")";
    }

    static NamedTuple* Make(const std::vector<const Type*>& types, const std::vector<std::string>& names) {
        return MakeSubtype<NamedTuple>(types,names);
    }
};

class Tuple : public CompositeType {
public:
    Tuple(const std::vector<const Type*>& types, const std::vector<std::string>& names) : 
            CompositeType(TypeCategory::catTuple, types, names)
    {
        m_name = "Tuple(";
        for (long k = 0; k < types.size();k++) {
            if (k) {
                m_name += ", ";
            }
            m_name += types[k]->name();
        }
        m_name += ")";
    }

    static Tuple* Make(const std::vector<const Type*>& types) {
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
    TupleOf(const Type* type) : 
            Type(TypeCategory::catTupleOf),
            m_element_type(type)
    {
        m_name = "TupleOf(" + type->name() + ")";
        m_size = sizeof(void*);
        m_is_default_constructible = true;
    }

    template<class buf_t>
    void serialize(instance_ptr self, buf_t& buffer) const {
        int32_t ct = count(self);
        buffer.write_uint32(ct);
        for (long k = 0; k < ct;k++) {
            m_element_type->serialize(eltPtr(self,k),buffer);
        }
    }

    template<class buf_t>
    void deserialize(instance_ptr self, buf_t& buffer) const {
        int32_t ct = buffer.read_uint32();
        
        if (ct > buffer.remaining() && m_element_type->bytecount()) {
            throw std::runtime_error("Corrupt data (tuplecount)");
        }

        constructor(self, ct, [&](instance_ptr tgt, int k) {
            m_element_type->deserialize(tgt, buffer);
        });
    }

    void repr(instance_ptr self, std::ostringstream& stream) const {
        stream << "(";

        int32_t ct = count(self);

        for (long k = 0; k < ct;k++) {
            if (k > 0) {
                stream << ", ";
            }

            m_element_type->repr(eltPtr(self,k),stream);
        }

        stream << ")";
    }

    int32_t hash32(instance_ptr left) const {
        if (!(*(layout**)left)) {
            return 0x123;
        }

        if ((*(layout**)left)->hash_cache == -1) {
            Hash32Accumulator acc((int)getTypeCategory());
            
            int32_t ct = count(left);
            acc.add(ct);

            for (long k = 0; k < ct;k++) {
                acc.add(m_element_type->hash32(eltPtr(left, k)));
            }

            (*(layout**)left)->hash_cache = acc.get();
            if ((*(layout**)left)->hash_cache == -1) {
                (*(layout**)left)->hash_cache = -2;
            }
        }

        return (*(layout**)left)->hash_cache;
    }

    char cmp(instance_ptr left, instance_ptr right) const {
        if (!(*(layout**)left) && (*(layout**)right)) {
            return -1;
        }
        if (!(*(layout**)right) && (*(layout**)left)) {
            return 1;
        }
        if (!(*(layout**)right) && !(*(layout**)left)) {
            return 0;
        }
        layout& left_layout = **(layout**)left;
        layout& right_layout = **(layout**)right;

        size_t bytesPer = m_element_type->bytecount();

        for (long k = 0; k < left_layout.count && k < right_layout.count; k++) {
            char res = m_element_type->cmp(left_layout.data + bytesPer * k, 
                                           right_layout.data + bytesPer * k);

            if (res != 0) {
                return res;
            }
        }

        if (left_layout.count < right_layout.count) {
            return -1;
        }

        if (left_layout.count > right_layout.count) {
            return 1;
        }

        return 0;
    }

    const Type* getEltType() const {
        return m_element_type;
    }

    static TupleOf* Make(const Type* elt) {
        static std::mutex guard;

        std::lock_guard<std::mutex> lock(guard);

        static std::map<const Type*, TupleOf*> m;

        auto it = m.find(elt);
        if (it == m.end()) {
            it = m.insert(std::make_pair(elt, new TupleOf(elt))).first;
        }

        return it->second;
    };

    instance_ptr eltPtr(instance_ptr self, int64_t i) const {
        if (!(*(layout**)self)) {
            return self;
        }

        return (*(layout**)self)->data + i * m_element_type->bytecount();
    }

    int64_t count(instance_ptr self) const {
        if (!(*(layout**)self)) {
            return 0;
        }

        return (*(layout**)self)->count;
    }

    template<class sub_constructor>
    void constructor(instance_ptr self, int64_t count, const sub_constructor& allocator) const {
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

    void constructor(instance_ptr self) const {
        constructor(self, 0, [](instance_ptr i, int64_t k) {});
    }

    void destroy(instance_ptr self) const {
        if (!(*(layout**)self)) {
            return;
        }

        (*(layout**)self)->refcount--;
        if ((*(layout**)self)->refcount == 0) {
            m_element_type->destroy((*(layout**)self)->count, [&](int64_t k) {return eltPtr(self,k);});
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

private:
    const Type* m_element_type;
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
    ConstDict(const Type* key, const Type* value) : 
            Type(TypeCategory::catConstDict),
            m_key(key),
            m_value(value)
    {
        m_name = "ConstDict(" + key->name() + "->" + value->name() + ")";
        m_size = sizeof(void*);
        m_is_default_constructible = true;
        m_bytes_per_key = m_key->bytecount();
        m_bytes_per_key_value_pair = m_key->bytecount() + m_value->bytecount();
        m_bytes_per_key_subtree_pair = m_key->bytecount() + this->bytecount();
        m_key_value_pair_type = Tuple::Make({m_key, m_value});
    }

    static ConstDict* Make(const Type* key, const Type* value) {
        static std::mutex guard;

        std::lock_guard<std::mutex> lock(guard);

        static std::map<std::pair<const Type*, const Type*>, ConstDict*> m;

        auto lookup_key = std::make_pair(key,value);

        auto it = m.find(lookup_key);
        if (it == m.end()) {
            it = m.insert(std::make_pair(lookup_key, new ConstDict(key, value))).first;
        }

        return it->second;
    };

    template<class buf_t>
    void serialize(instance_ptr self, buf_t& buffer) const {
        int32_t ct = count(self);
        buffer.write_uint32(ct);
        for (long k = 0; k < ct;k++) {
            m_key->serialize(kvPairPtrKey(self,k),buffer);
            m_value->serialize(kvPairPtrValue(self,k),buffer);
        }
    }

    template<class buf_t>
    void deserialize(instance_ptr self, buf_t& buffer) const {
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

    void repr(instance_ptr self, std::ostringstream& stream) const {
        stream << "{";

        int32_t ct = count(self);

        for (long k = 0; k < ct;k++) {
            if (k > 0) {
                stream << ", ";
            }

            m_key->repr(kvPairPtrKey(self,k),stream);
            stream << ": ";
            m_value->repr(kvPairPtrValue(self,k),stream);
        }

        stream << "}";
    }

    int32_t hash32(instance_ptr left) const {
        if (size(left) == 0) {
            return 0x123456;
        }

        if ((*(layout**)left)->hash_cache == -1) {
            Hash32Accumulator acc((int)getTypeCategory());

            int32_t count = size(left);
            acc.add(count);
            for (long k = 0; k < count;k++) {
                acc.add(m_key->hash32(kvPairPtrKey(left,k)));
                acc.add(m_value->hash32(kvPairPtrValue(left,k)));
            }

            (*(layout**)left)->hash_cache = acc.get();
            if ((*(layout**)left)->hash_cache == -1) {
                (*(layout**)left)->hash_cache = -2;
            }
        }

        return (*(layout**)left)->hash_cache;
    }

    //to make this fast(er), we do dict size comparison first, then keys, then values    
    char cmp(instance_ptr left, instance_ptr right) const {
        if (size(left) < size(right)) {
            return -1;
        }
        if (size(left) > size(right)) {
            return 1;
        }

        int ct = count(left);
        for (long k = 0; k < ct; k++) {
            char res = m_key->cmp(kvPairPtrKey(left,k), kvPairPtrKey(right,k));
            if (res) { 
                return res;
            }
        }

        for (long k = 0; k < ct; k++) {
            char res = m_value->cmp(
                kvPairPtrValue(left,k), 
                kvPairPtrValue(right,k)
                );

            if (res) { 
                return res;
            }
        }

        return 0;
    }

    void addDicts(instance_ptr lhs, instance_ptr rhs, instance_ptr output) const {
        std::vector<instance_ptr> keep;
        
        int64_t lhsCount = count(lhs);
        int64_t rhsCount = count(rhs);
        
        for (long k = 0; k < lhsCount; k++) {
            instance_ptr lhsVal = kvPairPtrKey(lhs, k);

            if (!lookupValueByKey(rhs, lhsVal)) {
                keep.push_back(lhsVal);
            }
        }

        constructor(output, rhsCount + keep.size(), false);

        for (long k = 0; k < rhsCount; k++) {
            m_key->copy_constructor(kvPairPtrKey(output,k), kvPairPtrKey(rhs, k));
            m_value->copy_constructor(kvPairPtrValue(output,k), kvPairPtrValue(rhs, k));
        }
        for (long k = 0; k < keep.size(); k++) {
            m_key->copy_constructor(kvPairPtrKey(output,k + rhsCount), keep[k]);
            m_value->copy_constructor(kvPairPtrValue(output,k + rhsCount), keep[k] + m_bytes_per_key);
        }
        incKvPairCount(output, keep.size() + rhsCount);
    }

    TupleOf* tupleOfKeysType() const {
        return TupleOf::Make(m_key);
    }

    void subtractTupleOfKeysFromDict(instance_ptr lhs, instance_ptr rhs, instance_ptr output) const {
        TupleOf* tupleType = tupleOfKeysType();
        
        int64_t lhsCount = count(lhs);
        int64_t rhsCount = tupleType->count(rhs);

        std::set<int> remove;
            
        for (long k = 0; k < rhsCount; k++) {
            int64_t index = lookupIndexByKey(lhs, tupleType->eltPtr(rhs, k));
            if (index != -1) {
                remove.insert(index);
            }
        }

        constructor(output, lhsCount - remove.size(), false);

        long written = 0;
        for (long k = 0; k < lhsCount; k++) {
            if (remove.find(k) == remove.end()) {
                m_key->copy_constructor(kvPairPtrKey(output,written), kvPairPtrKey(lhs, k));
                m_value->copy_constructor(kvPairPtrValue(output,written), kvPairPtrValue(lhs, k));

                written++;
            }
        }

        incKvPairCount(output, written);
    }

    instance_ptr kvPairPtrKey(instance_ptr self, int64_t i) const {
        if (!(*(layout**)self)) {
            return self;
        }

        layout& record = **(layout**)self;

        return record.data + m_bytes_per_key_value_pair * i;
    }

    instance_ptr kvPairPtrValue(instance_ptr self, int64_t i) const {
        if (!(*(layout**)self)) {
            return self;
        }

        layout& record = **(layout**)self;

        return record.data + m_bytes_per_key_value_pair * i + m_bytes_per_key;
    }

    void incKvPairCount(instance_ptr self, int by = 1) const {
        if (by == 0) {
            return;
        }

        layout& record = **(layout**)self;
        record.count += by;
    }

    void sortKvPairs(instance_ptr self) const {
        if (!*(layout**)self) {
            return;
        }

        layout& record = **(layout**)self;

        assert(!record.subpointers);

        if (record.count <= 1) { 
            return; 
        }
        else if (record.count == 2) {
            if (m_key->cmp(kvPairPtrKey(self, 0), kvPairPtrKey(self,1)) > 0) {
                m_key->swap(kvPairPtrKey(self,0), kvPairPtrKey(self,1));
                m_value->swap(kvPairPtrValue(self,0), kvPairPtrValue(self,1));
            }
            return;
        } else {
            std::vector<int> indices;
            for (long k=0;k<record.count;k++) {
                indices.push_back(k);
            }

            std::sort(indices.begin(), indices.end(), [&](int l, int r) {
                char res = m_key->cmp(kvPairPtrKey(self,l),kvPairPtrKey(self,r));
                return res < 0;
                });

            //create a temporary buffer
            std::vector<uint8_t> d;
            d.resize(m_bytes_per_key_value_pair * record.count);

            //final_lookup contains the location of each value in the original sort
            for (long k = 0; k < indices.size(); k++) {
                m_key->swap(kvPairPtrKey(self, indices[k]), &d[m_bytes_per_key_value_pair*k]);
                m_value->swap(kvPairPtrValue(self, indices[k]), &d[m_bytes_per_key_value_pair*k+m_bytes_per_key]);
            }

            //now move them back
            for (long k = 0; k < indices.size(); k++) {
                m_key->swap(kvPairPtrKey(self, k), &d[m_bytes_per_key_value_pair*k]);
                m_value->swap(kvPairPtrValue(self, k), &d[m_bytes_per_key_value_pair*k+m_bytes_per_key]);
            }
        }
    }

    instance_ptr keyTreePtr(instance_ptr self, int64_t i) const {
        if (!(*(layout**)self)) {
            return self;
        }

        layout& record = **(layout**)self;

        return record.data + m_bytes_per_key_subtree_pair * i;
    }

    bool instanceIsSubtrees(instance_ptr self) const {
        if (!(*(layout**)self)) {
            return self;
        }

        layout& record = **(layout**)self;

        return record.subpointers != 0;
    }

    int64_t count(instance_ptr self) const {
        if (!(*(layout**)self)) {
            return 0;
        }

        layout& record = **(layout**)self;

        if (record.subpointers) {
            return record.subpointers;
        }

        return record.count;
    }

    int64_t size(instance_ptr self) const {
        if (!(*(layout**)self)) {
            return 0;
        }

        return (*(layout**)self)->count;
    }

    int64_t lookupIndexByKey(instance_ptr self, instance_ptr key) const {
        if (!(*(layout**)self)) {
            return -1;
        }

        layout& record = **(layout**)self;

        assert(record.subpointers == 0); //this is not implemented yet

        long low = 0;
        long high = record.count;

        while (low < high) {
            long mid = (low+high)/2;
            char res = m_key->cmp(kvPairPtrKey(self, mid), key);
            
            if (res == 0) {
                return mid;
            } else if (res < 0) {
                low = mid+1;
            } else {
                high = mid;
            }
        }

        return -1;
    }

    instance_ptr lookupValueByKey(instance_ptr self, instance_ptr key) const {
        int64_t offset = lookupIndexByKey(self, key);
        if (offset == -1) {
            return 0;
        }
        return kvPairPtrValue(self, offset);
    }

    void constructor(instance_ptr self, int64_t space, bool isPointerTree) const {
        if (space == 0) {
            (*(layout**)self) = nullptr;
            return;
        }

        int bytesPer = isPointerTree ? m_bytes_per_key_subtree_pair : m_bytes_per_key_value_pair;

        (*(layout**)self) = (layout*)malloc(sizeof(layout) + bytesPer * space);

        layout& record = **(layout**)self;

        record.count = 0;
        record.subpointers = 0;
        record.refcount = 1;
        record.hash_cache = -1;
    }

    void constructor(instance_ptr self) const {
        (*(layout**)self) = nullptr;
    }

    void destroy(instance_ptr self) const {
        if (!(*(layout**)self)) {
            return;
        }

        layout& record = **(layout**)self;

        record.refcount--;
        if (record.refcount == 0) {
            if (record.subpointers == 0) {
                m_key->destroy(record.count, [&](long ix) { 
                    return record.data + m_bytes_per_key_value_pair * ix; 
                });
                m_value->destroy(record.count, [&](long ix) { 
                    return record.data + m_bytes_per_key_value_pair * ix + m_bytes_per_key; 
                });
            } else {
                m_key->destroy(record.subpointers, [&](long ix) { 
                    return record.data + m_bytes_per_key_subtree_pair * ix; 
                });
                ((const Type*)this)->destroy(record.subpointers, [&](long ix) { 
                    return record.data + m_bytes_per_key_subtree_pair * ix + m_bytes_per_key; 
                });
            }

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
    
    
    const Type* keyValuePairType() const { return m_key_value_pair_type; }
    const Type* keyType() const { return m_key; }
    const Type* valueType() const { return m_value; }

private:
    const Type* m_key;
    const Type* m_value;
    const Type* m_key_value_pair_type;
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

    char cmp(instance_ptr left, instance_ptr right) const {
        return 0;
    }

    int32_t hash32(instance_ptr left) const {
        return (int)getTypeCategory();
    }

    void constructor(instance_ptr self) const {}

    void destroy(instance_ptr self) const {}

    void copy_constructor(instance_ptr self, instance_ptr other) const {}

    void assign(instance_ptr self, instance_ptr other) const {}

    static None* Make() { static None res; return &res; }

    template<class buf_t>
    void deserialize(instance_ptr self, buf_t& buffer) const {
    }

    template<class buf_t>
    void serialize(instance_ptr self, buf_t& buffer) const {
    }

    void repr(instance_ptr self, std::ostringstream& stream) const {
        stream << "None";
    }
};

template<class T>
class RegisterType : public Type {
public:
    RegisterType(TypeCategory kind) : Type(kind) 
    {
        m_size = sizeof(T);
        m_is_default_constructible = true;
    }

    char cmp(instance_ptr left, instance_ptr right) const {
        if ( (*(T*)left) < (*(T*)right) ) {
            return -1;
        }
        if ( (*(T*)left) > (*(T*)right) ) {
            return 1;
        }

        return 0;
    }

    int32_t hash32(instance_ptr left) const {
        Hash32Accumulator acc((int)getTypeCategory());

        acc.addRegister(*(T*)left);

        return acc.get();
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

    template<class buf_t>
    void deserialize(instance_ptr self, buf_t& buffer) const {
        buffer.read_bytes(self, m_size);
    }

    template<class buf_t>
    void serialize(instance_ptr self, buf_t& buffer) const {
        buffer.write_bytes(self, m_size);
    }
};

class Bool : public RegisterType<bool> {
public:
    Bool() : RegisterType(TypeCategory::catBool)
    {
        m_name = "Bool";
    }

    void repr(instance_ptr self, std::ostringstream& stream) const {
        stream << (*(bool*)self ? "True":"False");
    }

    static Bool* Make() { static Bool res; return &res; }
};

class UInt8 : public RegisterType<uint8_t> {
public:
    UInt8() : RegisterType(TypeCategory::catUInt8)
    {
        m_name = "UInt8";
    }

    void repr(instance_ptr self, std::ostringstream& stream) const {
        stream << (uint64_t)*(uint8_t*)self << "u8";
    }

    static UInt8* Make() { static UInt8 res; return &res; }
};

class UInt16 : public RegisterType<uint16_t> {
public:
    UInt16() : RegisterType(TypeCategory::catUInt16)
    {
        m_name = "UInt16";
    }

    void repr(instance_ptr self, std::ostringstream& stream) const {
        stream << (uint64_t)*(uint16_t*)self << "u16";
    }

    static UInt16* Make() { static UInt16 res; return &res; }
};

class UInt32 : public RegisterType<uint32_t> {
public:
    UInt32() : RegisterType(TypeCategory::catUInt32)
    {
        m_name = "UInt32";
    }

    void repr(instance_ptr self, std::ostringstream& stream) const {
        stream << (uint64_t)*(uint32_t*)self << "u32";
    }

    static UInt32* Make() { static UInt32 res; return &res; }
};

class UInt64 : public RegisterType<uint64_t> {
public:
    UInt64() : RegisterType(TypeCategory::catUInt64)
    {
        m_name = "UInt64";
    }

    void repr(instance_ptr self, std::ostringstream& stream) const {
        stream << *(uint64_t*)self << "u64";
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

    void repr(instance_ptr self, std::ostringstream& stream) const {
        stream << (int64_t)*(int8_t*)self << "i8";
    }

    static Int8* Make() { static Int8 res; return &res; }
};

class Int16 : public RegisterType<int16_t> {
public:
    Int16() : RegisterType(TypeCategory::catInt16)
    {
        m_name = "Int16";
    }

    void repr(instance_ptr self, std::ostringstream& stream) const {
        stream << (int64_t)*(int16_t*)self << "i16";
    }

    static Int16* Make() { static Int16 res; return &res; }
};

class Int32 : public RegisterType<int32_t> {
public:
    Int32() : RegisterType(TypeCategory::catInt32)
    {
        m_name = "Int32";
    }

    void repr(instance_ptr self, std::ostringstream& stream) const {
        stream << (int64_t)*(int32_t*)self << "i32";
    }

    static Int32* Make() { static Int32 res; return &res; }
};

class Int64 : public RegisterType<int64_t> {
public:
    Int64() : RegisterType(TypeCategory::catInt64)
    {
        m_name = "Int64";
    }

    void repr(instance_ptr self, std::ostringstream& stream) const {
        stream << *(int64_t*)self;
    }

    static Int64* Make() { static Int64 res; return &res; }
};

class Float32 : public RegisterType<float> {
public:
    Float32() : RegisterType(TypeCategory::catFloat32)
    {
        m_name = "Float32";
    }

    void repr(instance_ptr self, std::ostringstream& stream) const {
        stream << *(float*)self << "f32";
    }

    static Float32* Make() { static Float32 res; return &res; }
};

class Float64 : public RegisterType<double> {
public:
    Float64() : RegisterType(TypeCategory::catFloat64)
    {
        m_name = "Float64";
    }

    void repr(instance_ptr self, std::ostringstream& stream) const {
        stream << *(double*)self;
    }

    static Float64* Make() { static Float64 res; return &res; }
};

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

    static String* Make() { static String res; return &res; }

    template<class buf_t>
    void serialize(instance_ptr self, buf_t& buffer) const {
        buffer.write_uint32(count(self));
        buffer.write_uint8(bytes_per_codepoint(self));
        buffer.write_bytes(eltPtr(self,0), bytes_per_codepoint(self) * count(self));
    }

    template<class buf_t>
    void deserialize(instance_ptr self, buf_t& buffer) const {
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

    int32_t hash32(instance_ptr left) const {
        if (!(*(layout**)left)) {
            return 0x12345;
        }

        if ((*(layout**)left)->hash_cache == -1) {
            Hash32Accumulator acc((int)getTypeCategory());
            acc.addBytes(eltPtr(left, 0), bytes_per_codepoint(left) * count(left));
            (*(layout**)left)->hash_cache = acc.get();
            if ((*(layout**)left)->hash_cache == -1) {
                (*(layout**)left)->hash_cache = -2;
            }
        }

        return (*(layout**)left)->hash_cache;
    }

    char cmp(instance_ptr left, instance_ptr right) const {
        if ( !(*(layout**)left) && !(*(layout**)right) ) {
            return 0;
        }
        if ( !(*(layout**)left) && (*(layout**)right) ) {
            return -1;
        }
        if ( (*(layout**)left) && !(*(layout**)right) ) {
            return 1;
        }

        if (bytes_per_codepoint(left) < bytes_per_codepoint(right)) {
            return -1;
        }

        if (bytes_per_codepoint(left) > bytes_per_codepoint(right)) {
            return 1;
        }

        int bytesPer = bytes_per_codepoint(right);

        char res = byteCompare(
            eltPtr(left, 0), 
            eltPtr(right, 0), 
            bytesPer * std::min(count(left), count(right))
            );

        if (res) {
            return res;
        }

        if (count(left) < count(right)) { 
            return -1; 
        }

        if (count(left) > count(right)) { 
            return 1; 
        }

        return 0;
    }

    void constructor(instance_ptr self, int64_t bytes_per_codepoint, int64_t count, const char* data) const {
        if (count == 0) {
            *(layout**)self = nullptr;
            return;
        }

        (*(layout**)self) = (layout*)malloc(sizeof(layout) + count * bytes_per_codepoint);

        (*(layout**)self)->bytes_per_codepoint = bytes_per_codepoint;
        (*(layout**)self)->pointcount = count;
        (*(layout**)self)->hash_cache = -1;
        (*(layout**)self)->refcount = 1;
        
        if (data) {
            ::memcpy((*(layout**)self)->data, data, count * bytes_per_codepoint);
        }
    }

    void repr(instance_ptr self, std::ostringstream& stream) const {
        //as if it were bytes, which is totally wrong...
        stream << "\"";
        long bytes = count(self);
        uint8_t* base = eltPtr(self,0);

        static char hexDigits[] = "0123456789abcdef";
        
        for (long k = 0; k < bytes;k++) {
            if (base[k] == '"') {
                stream << "\\\"";
            } else if (base[k] == '\\') {
                stream << "\\\\";
            } else if (isprint(base[k])) {
                stream << base[k];
            } else {
                stream << "\\x" << hexDigits[base[k] / 16] << hexDigits[base[k] % 16];
            }
        }

        stream << "\"";
    }

    instance_ptr eltPtr(instance_ptr self, int64_t i) const {
        const static char* emptyPtr = "";

        if (*(layout**)self == nullptr) { 
            return (instance_ptr)emptyPtr;
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
        int32_t hash_cache;
        int32_t bytecount;
        uint8_t data[];
    };

    void repr(instance_ptr self, std::ostringstream& stream) const {
        stream << "b" << "'";
        long bytes = count(self);
        uint8_t* base = eltPtr(self,0);

        static char hexDigits[] = "0123456789abcdef";
        
        for (long k = 0; k < bytes;k++) {
            if (base[k] == '\'') {
                stream << "\\'";
            } else if (base[k] == '\r') {
                stream << "\\r";
            } else if (base[k] == '\n') {
                stream << "\\n";
            } else if (base[k] == '\t') {
                stream << "\\t";
            } else if (base[k] == '\\') {
                stream << "\\\\";
            } else if (isprint(base[k])) {
                stream << base[k];
            } else {
                stream << "\\x" << hexDigits[base[k] / 16] << hexDigits[base[k] % 16];
            }
        }
        
        stream << "'";
    }

    template<class buf_t>
    void serialize(instance_ptr self, buf_t& buffer) const {
        buffer.write_uint32(count(self));
        buffer.write_bytes(eltPtr(self, 0), count(self));
    }

    template<class buf_t>
    void deserialize(instance_ptr self, buf_t& buffer) const {
        int32_t ct = buffer.read_uint32();
        
        if (ct > buffer.remaining()) {
            throw std::runtime_error("Corrupt data (bytes)");
        }

        constructor(self, ct, nullptr);

        if (ct) {
            buffer.read_bytes(eltPtr(self,0), ct);
        }
    }
    
    int32_t hash32(instance_ptr left) const {
        Hash32Accumulator acc((int)getTypeCategory());

        if (!(*(layout**)left)) {
            return 0x1234;
        }

        if ((*(layout**)left)->hash_cache == -1) {
            Hash32Accumulator acc((int)getTypeCategory());
            
            acc.addBytes(eltPtr(left, 0), count(left));

            (*(layout**)left)->hash_cache = acc.get();
            if ((*(layout**)left)->hash_cache == -1) {
                (*(layout**)left)->hash_cache = -2;
            }
        }

        return (*(layout**)left)->hash_cache;
    }

    char cmp(instance_ptr left, instance_ptr right) const {
        if ( !(*(layout**)left) && !(*(layout**)right) ) {
            return 0;
        }
        if ( !(*(layout**)left) && (*(layout**)right) ) {
            return -1;
        }
        if ( (*(layout**)left) && !(*(layout**)right) ) {
            return 1;
        }

        char res = byteCompare(eltPtr(left, 0), eltPtr(right, 0), std::min(count(left), count(right)));

        if (res) {
            return res;
        }

        if (count(left) < count(right)) { 
            return -1; 
        }

        if (count(left) > count(right)) { 
            return 1; 
        }

        return 0;
    }

    Bytes() : Type(TypeCategory::catBytes)
    {
        m_name = "Bytes";
        m_is_default_constructible = true;
        m_size = sizeof(layout*);
    }

    static Bytes* Make() { static Bytes res; return &res; }

    void constructor(instance_ptr self, int64_t count, const char* data) const {
        if (count == 0) {
            *(layout**)self = nullptr;
            return;
        }
        (*(layout**)self) = (layout*)malloc(sizeof(layout) + count);

        (*(layout**)self)->bytecount = count;
        (*(layout**)self)->refcount = 1;
        (*(layout**)self)->hash_cache = -1;
        
        if (data) {
            ::memcpy((*(layout**)self)->data, data, count);
        }
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

class Instance {
private:
    class layout {
    public:
        std::atomic<int64_t> refcount;
        const Type* type;
        uint8_t data[];
    };

    Instance(layout* l) : 
        mLayout(l) 
    {
    }

public:
    static Instance deserialized(const Type* t, DeserializationBuffer& buf) {
        return createAndInitialize(t, [&](instance_ptr tgt) {
            t->deserialize(tgt, buf);
        });
    }

    static Instance create(const Type*t, instance_ptr data) {
        return createAndInitialize(t, [&](instance_ptr tgt) {
            t->copy_constructor(tgt, data);
        });
    }

    template<class initializer_type>
    static Instance createAndInitialize(const Type*t, const initializer_type& initFun) {
        layout* l = (layout*)malloc(sizeof(layout) + t->bytecount());
        
        try {
            initFun(l->data);
        } catch(...) {
            free(l);
            throw;
        }

        l->refcount = 1;
        l->type = t;

        return Instance(l);
    }

    Instance(const Instance& other) : mLayout(other.mLayout) {
        mLayout->refcount++;
    }

    Instance& operator=(const Instance& other) {
        other.mLayout->refcount++;

        mLayout->refcount--;
        if (mLayout->refcount == 0) {
            mLayout->type->destroy(mLayout->data);
            free(mLayout);
        }

        mLayout = other.mLayout;
        return *this;
    }

    bool operator<(const Instance& other) const {
        if (mLayout->type < other.mLayout->type) {
            return true;
        }
        if (mLayout->type > other.mLayout->type) {
            return false;
        }
        return mLayout->type->cmp(mLayout->data, other.mLayout->data) < 0;
    }

    std::string repr() const {
        std::ostringstream s;
        s << std::showpoint;
        mLayout->type->repr(mLayout->data, s);
        return s.str();
    }

    int32_t hash32() const {
        return mLayout->type->hash32(mLayout->data);
    }

    const Type* type() const {
        return mLayout->type;
    }

    instance_ptr data() const {
        return mLayout->data;
    }

private:
    layout* mLayout;
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

    char cmp(instance_ptr left, instance_ptr right) const {
        return 0;
    }

    int32_t hash32(instance_ptr left) const {
        return mInstance.hash32();
    }

    void repr(instance_ptr self, std::ostringstream& stream) const {
        mInstance.type()->repr(mInstance.data(), stream);
    }

    template<class buf_t>
    void serialize(instance_ptr self, buf_t& buffer) const {
    }

    template<class buf_t>
    void deserialize(instance_ptr self, buf_t& buffer) const {
    }

    void constructor(instance_ptr self) const {}

    void destroy(instance_ptr self) const {}

    void copy_constructor(instance_ptr self, instance_ptr other) const {}

    void assign(instance_ptr self, instance_ptr other) const {}

    const Instance& value() const {
        return mInstance;
    }

    static const Type* Make(Instance i) { 
        static std::mutex guard;

        std::lock_guard<std::mutex> lock(guard);

        static std::map<Instance, Value*> m;

        auto it = m.find(i);

        if (it == m.end()) {
            it = m.insert(std::make_pair(i, new Value(i))).first;
        }

        return it->second;
    }

    static const Type* MakeInt64(int64_t i) { 
        return Make(Instance::create(Int64::Make(), (instance_ptr)&i));
    }
    static const Type* MakeFloat64(double i) { 
        return Make(Instance::create(Float64::Make(), (instance_ptr)&i));
    }
    static const Type* MakeBool(bool i) { 
        return Make(Instance::create(Bool::Make(), (instance_ptr)&i));
    }

    static const Type* MakeBytes(char* data, size_t count) { 
        return Make(Instance::createAndInitialize(Bytes::Make(), [&](instance_ptr i) {
            Bytes::Make()->constructor(i, count, data);
        }));
    }

    static const Type* MakeString(size_t bytesPerCodepoint, size_t count, char* data) { 
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

    Alternative(std::string name, const std::vector<std::pair<std::string, NamedTuple*> >& subtypes) :
            Type(TypeCategory::catAlternative),
            m_default_construction_ix(0),
            m_default_construction_type(nullptr),
            m_subtypes(subtypes)
    {
        m_name = name;
            
        if (m_subtypes.size() > 255) {
            throw std::runtime_error("Can't have an alternative with more than 255 subelements");
        }

        m_size = sizeof(void*);

        m_is_default_constructible = false;
        m_all_alternatives_empty = true;

        for (auto& subtype_pair: m_subtypes) {
            if (subtype_pair.second->bytecount() > 0) {
                m_all_alternatives_empty = false;
            }

            if (m_arg_positions.find(subtype_pair.first) != m_arg_positions.end()) {
                throw std::runtime_error("Can't create an alternative with " + 
                        subtype_pair.first + " defined twice.");
            }

            m_arg_positions[subtype_pair.first] = m_arg_positions.size();

            if (subtype_pair.second->is_default_constructible()) {
                m_is_default_constructible = true;
                m_default_construction_ix = m_arg_positions[subtype_pair.first];
            }
        }

        m_size = (m_all_alternatives_empty ? 1 : sizeof(void*));
    }

    char cmp(instance_ptr left, instance_ptr right) const {
        if (m_all_alternatives_empty) {
            if (*(uint8_t*)left < *(uint8_t*)right) {
                return -1;
            }
            if (*(uint8_t*)left > *(uint8_t*)right) {
                return 1;
            }
            return 0;
        }

        layout& record_l = **(layout**)left;
        layout& record_r = **(layout**)right;

        if (record_l.which < record_r.which) {
            return -1;
        }
        if (record_l.which > record_r.which) {
            return 1;
        }

        return m_subtypes[record_l.which].second->cmp(record_l.data, record_r.data);
    }

    template<class buf_t>
    void serialize(instance_ptr self, buf_t& buffer) const {
        buffer.write_uint8(which(self));
        m_subtypes[which(self)].second->serialize(eltPtr(self), buffer);
    }

    template<class buf_t>
    void deserialize(instance_ptr self, buf_t& buffer) const {
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

    void repr(instance_ptr self, std::ostringstream& stream) const {
        stream << m_subtypes[which(self)].first;
        m_subtypes[which(self)].second->repr(eltPtr(self), stream);
    }

    int32_t hash32(instance_ptr left) const {
        Hash32Accumulator acc((int)TypeCategory::catAlternative);

        acc.add(which(left));
        acc.add(m_subtypes[which(left)].second->hash32(eltPtr(left)));

        return acc.get();
    }

    instance_ptr eltPtr(instance_ptr self) const {
        if (m_all_alternatives_empty) {
            return self;
        }

        layout& record = **(layout**)self;
        
        return record.data;
    }

    int64_t which(instance_ptr self) const {
        if (m_all_alternatives_empty) {
            return *(uint8_t*)self;
        }

        layout& record = **(layout**)self;
        
        return record.which;
    }

    void constructor(instance_ptr self) const;

    void destroy(instance_ptr self) const {
        if (m_all_alternatives_empty) {
            return;
        }

        layout& record = **(layout**)self;
        
        record.refcount--;
        if (record.refcount == 0) {
            m_subtypes[record.which].second->destroy(record.data);
            free(*(layout**)self);
        }
    }

    void copy_constructor(instance_ptr self, instance_ptr other) const {
        if (m_all_alternatives_empty) {
            *(uint8_t*)self = *(uint8_t*)other;
            return;
        }

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

    static Alternative* Make(std::string name,
                         const std::vector<std::pair<std::string, NamedTuple*> >& types
                         ) {
        static std::mutex guard;

        std::lock_guard<std::mutex> lock(guard);

        typedef std::pair<std::string, std::vector<std::pair<std::string, NamedTuple*> > > keytype;

        static std::map<keytype, Alternative*> m;

        auto it = m.find(keytype(name, types));

        if (it == m.end()) {
            it = m.insert(std::make_pair(keytype(name, types), new Alternative(name, types))).first;
        }

        return it->second;
    }

    const std::vector<std::pair<std::string, NamedTuple*> >& subtypes() const {
        return m_subtypes;
    }

    bool all_alternatives_empty() const {
        return m_all_alternatives_empty;
    }

    const Type* pickConcreteSubclassConcrete(instance_ptr data) const;

private:
    bool m_all_alternatives_empty;

    int m_default_construction_ix;

    mutable const Type* m_default_construction_type;

    std::vector<std::pair<std::string, NamedTuple*> > m_subtypes;

    std::map<std::string, int> m_arg_positions;
};

class ConcreteAlternative : public Type {
public:
    typedef Alternative::layout layout;

    ConcreteAlternative(const Alternative* m_alternative, int64_t which) :
            Type(TypeCategory::catConcreteAlternative),
            m_alternative(m_alternative),
            m_which(which)
    {   
        m_base = m_alternative;
        m_name = m_alternative->name() + "." + m_alternative->subtypes()[which].first;
        m_size = m_alternative->bytecount();
        m_is_default_constructible = m_alternative->subtypes()[which].second->is_default_constructible();
    }

    int32_t hash32(instance_ptr left) const {
        return m_alternative->hash32(left);
    }

    void repr(instance_ptr self, std::ostringstream& stream) const {
        m_alternative->repr(self,stream);
    }

    template<class buf_t>
    void deserialize(instance_ptr self, buf_t& buffer) const {
        m_alternative->deserialize(self,buffer);
    }

    template<class buf_t>
    void serialize(instance_ptr self, buf_t& buffer) const {
        m_alternative->serialize(self,buffer);
    }

    char cmp(instance_ptr left, instance_ptr right) const {
        return m_alternative->cmp(left,right);
    }

    void constructor(instance_ptr self) const {
        if (m_alternative->all_alternatives_empty()) {
            *(uint8_t*)self = m_which;
        } else {
            constructor(self, [&](instance_ptr i) {
                m_alternative->subtypes()[m_which].second->constructor(i);
            });
        }
    }

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

    void destroy(instance_ptr self) const {
        m_alternative->destroy(self);
    }

    void copy_constructor(instance_ptr self, instance_ptr other) const {
        m_alternative->copy_constructor(self, other);
    }

    void assign(instance_ptr self, instance_ptr other) const {
        m_alternative->assign(self, other);
    }

    static ConcreteAlternative* Make(const Alternative* alt, int64_t which) {
        static std::mutex guard;

        std::lock_guard<std::mutex> lock(guard);

        typedef std::pair<const Alternative*, int64_t> keytype;

        static std::map<keytype, ConcreteAlternative*> m;

        auto it = m.find(keytype(alt ,which));

        if (it == m.end()) {
            it = m.insert(
                std::make_pair(keytype(alt,which), new ConcreteAlternative(alt,which))
                ).first;
        }

        return it->second;
    }

    const Type* elementType() const {
        return m_alternative->subtypes()[m_which].second;
    }

    const Alternative* getAlternative() const {
        return m_alternative;
    }

    int64_t which() const {
        return m_which;
    }

private:
    const Alternative* m_alternative;

    int64_t m_which;
};

inline const Type* Alternative::pickConcreteSubclassConcrete(instance_ptr data) const {
    uint8_t i = which(data);

    return ConcreteAlternative::Make(this, i);
}

inline void Alternative::constructor(instance_ptr self) const {
    if (!m_default_construction_type) {
        m_default_construction_type = ConcreteAlternative::Make(this, m_default_construction_ix);
    }

    m_default_construction_type->constructor(self);
}

class PythonSubclass : public Type {
public:
    PythonSubclass(const Type* base, PyTypeObject* typePtr) :
            Type(TypeCategory::catPythonSubclass)
    {   
        m_base = base;
        m_name = typePtr->tp_name;
        m_size = m_base->bytecount();
        mTypeRep = typePtr;
        m_is_default_constructible = m_base->is_default_constructible();
    }

    int32_t hash32(instance_ptr left) const {
        return m_base->hash32(left);
    }

    template<class buf_t>
    void serialize(instance_ptr self, buf_t& buffer) const {
        m_base->serialize(self,buffer);
    }

    template<class buf_t>
    void deserialize(instance_ptr self, buf_t& buffer) const {
        m_base->deserialize(self,buffer);
    }

    void repr(instance_ptr self, std::ostringstream& stream) const {
        m_base->repr(self,stream);
    }

    char cmp(instance_ptr left, instance_ptr right) const {
        return m_base->cmp(left,right);
    }

    void constructor(instance_ptr self) const {
        m_base->constructor(self);
    }

    void destroy(instance_ptr self) const {
        m_base->destroy(self);
    }

    void copy_constructor(instance_ptr self, instance_ptr other) const {
        m_base->copy_constructor(self, other);
    }

    void assign(instance_ptr self, instance_ptr other) const {
        m_base->assign(self, other);
    }

    static PythonSubclass* Make(const Type* base, PyTypeObject* pyType) {
        static std::mutex guard;

        std::lock_guard<std::mutex> lock(guard);

        typedef std::pair<const Type*, PyTypeObject*> keytype;

        static std::map<keytype, PythonSubclass*> m;

        auto it = m.find(keytype(base, pyType));

        if (it == m.end()) {
            it = m.insert(
                std::make_pair(keytype(base,pyType), new PythonSubclass(base, pyType))
                ).first;
        }

        return it->second;
    }

    const Type* baseType() const {
        return m_base;
    }

    PyTypeObject* pyType() const {
        return mTypeRep;
    }
};

class Function : public Type {
public:
    class FunctionArg {
    public:
        FunctionArg(std::string name, const Type* typeFilterOrNull, PyObject* defaultValue, bool isStarArg, bool isKwarg) : 
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

        const Type* getTypeFilter() const {
            return m_typeFilter;
        }

        bool getIsStarArg() const {
            return m_isStarArg;
        }

        bool getIsKwarg() const {
            return m_isKwarg;
        }

    private:
        std::string m_name;
        const Type* m_typeFilter;
        PyObject* m_defaultValue;
        bool m_isStarArg;
        bool m_isKwarg;
    };

    class FunctionOverload {
    public:
        FunctionOverload(
            PyFunctionObject* functionObj, 
            const Type* returnType, 
            const std::vector<FunctionArg>& args
            ) : 
                mFunctionObj(functionObj),
                mReturnType(returnType),
                mArgs(args)
        {
        }

        PyFunctionObject* getFunctionObj() const {
            return mFunctionObj;
        }

        const Type* getReturnType() const {
            return mReturnType;
        }

        const std::vector<FunctionArg>& getArgs() const {
            return mArgs;
        }


    private:
        PyFunctionObject* mFunctionObj;
        const Type* mReturnType;
        std::vector<FunctionArg> mArgs;
    };

    Function(std::string inName, 
            const std::vector<FunctionOverload>& overloads
            ) :
        Type(catFunction),
        mOverloads(overloads)
    {
        m_name = inName;
        m_is_default_constructible = true;
        m_size = 0;
    }

    static Function* merge(const Function* f1, const Function* f2) {
        std::vector<FunctionOverload> overloads(f1->mOverloads);
        for (auto o: f2->mOverloads) {
            overloads.push_back(o);
        }
        return new Function(f1->m_name, overloads);
    }

    char cmp(instance_ptr left, instance_ptr right) const {
        return 0;
    }

    template<class buf_t>
    void deserialize(instance_ptr self, buf_t& buffer) const {
    }

    template<class buf_t>
    void serialize(instance_ptr self, buf_t& buffer) const {
    }

    void repr(instance_ptr self, std::ostringstream& stream) const {
        stream << "<function " << m_name << ">";
    }

    int32_t hash32(instance_ptr left) const {
        Hash32Accumulator acc((int)getTypeCategory());

        acc.addRegister((uint64_t)mPyFunc);

        return acc.get();
    }

    void constructor(instance_ptr self) const {
    }

    void destroy(instance_ptr self) const {
    }

    void copy_constructor(instance_ptr self, instance_ptr other) const {
    }

    void assign(instance_ptr self, instance_ptr other) const {
    }

private:
    PyFunctionObject* mPyFunc;
    std::vector<FunctionOverload> mOverloads;
};

class Class : public Type {
public:
    Class(std::string inName, 
          const std::vector<std::pair<std::string, const Type*> >& members,
          const std::map<std::string, const Function*>& memberFunctions,
          const std::map<std::string, const Function*>& staticFunctions,
          const std::map<std::string, PyObject*>& classMembers
          ) : 
            Type(catClass),
            m_members(members),
            m_memberFunctions(memberFunctions),
            m_staticFunctions(staticFunctions),
            m_classMembers(classMembers)
    {
        m_name = inName;

        m_is_default_constructible = true;

        m_size = 0;
        for (auto t: m_members) {
            m_byte_offsets.push_back(m_size);
            m_size += t.second->bytecount();

            if (!t.second->is_default_constructible()) {
                m_is_default_constructible = false;
            }
        }
    }

    static Class* Make(
            std::string inName, 
            const std::vector<std::pair<std::string, const Type*> >& members,
            const std::map<std::string, const Function*>& memberFunctions,
            const std::map<std::string, const Function*>& staticFunctions,
            const std::map<std::string, PyObject*>& classMembers
            )
    {
        return new Class(inName, members, memberFunctions, staticFunctions, classMembers);
    }

    instance_ptr eltPtr(instance_ptr self, int64_t ix) const {
        return self + m_byte_offsets[ix];
    }

    char cmp(instance_ptr left, instance_ptr right) const {
        for (long k = 0; k < m_members.size(); k++) {
            char res = m_members[k].second->cmp(left + m_byte_offsets[k], right + m_byte_offsets[k]);
            if (res != 0) {
                return res;
            }
        }

        return 0;
    }

    template<class buf_t>
    void deserialize(instance_ptr self, buf_t& buffer) const {
        for (long k = 0; k < m_members.size();k++) {
            m_members[k].second->deserialize(eltPtr(self,k),buffer);
        }
    }

    template<class buf_t>
    void serialize(instance_ptr self, buf_t& buffer) const {
        for (long k = 0; k < m_members.size();k++) {
            m_members[k].second->serialize(eltPtr(self,k),buffer);
        }
    }

    void repr(instance_ptr self, std::ostringstream& stream) const {
        stream << "(";

        for (long k = 0; k < m_members.size();k++) {
            if (k > 0) {
                stream << ", ";
            }

            stream << m_members[k].first << "=";

            m_members[k].second->repr(eltPtr(self,k),stream);
        }
        if (m_members.size() == 1) {
            stream << ",";
        }

        stream << ")";
    }

    int32_t hash32(instance_ptr left) const {
        Hash32Accumulator acc((int)getTypeCategory());

        for (long k = 0; k < m_members.size();k++) {
            acc.add(m_members[k].second->hash32(eltPtr(left,k)));
        }

        acc.add(m_members.size());

        return acc.get();
    }

    template<class sub_constructor>
    void constructor(instance_ptr self, const sub_constructor& initializer) const {
        for (int64_t k = 0; k < m_members.size(); k++) {
            try {
                initializer(eltPtr(self, k), k);
            } catch(...) {
                for (long k2 = k-1; k2 >= 0; k2--) {
                    m_members[k2].second->destroy(eltPtr(self,k2));
                }
                throw;
            }
        }
    }

    void constructor(instance_ptr self) const {
        if (!m_is_default_constructible) {
            throw std::runtime_error(m_name + " is not default-constructible");
        }

        for (size_t k = 0; k < m_members.size(); k++) {
            m_members[k].second->constructor(self+m_byte_offsets[k]);
        }
    }

    void destroy(instance_ptr self) const {
        for (long k = (long)m_members.size() - 1; k >= 0; k--) {
            m_members[k].second->destroy(self+m_byte_offsets[k]);
        }
    }

    void copy_constructor(instance_ptr self, instance_ptr other) const {
        for (long k = (long)m_members.size() - 1; k >= 0; k--) {
            m_members[k].second->copy_constructor(self + m_byte_offsets[k], other+m_byte_offsets[k]);
        }
    }

    void assign(instance_ptr self, instance_ptr other) const {
        for (long k = (long)m_members.size() - 1; k >= 0; k--) {
            m_members[k].second->assign(self + m_byte_offsets[k], other+m_byte_offsets[k]);
        }
    }
    const std::vector<std::pair<std::string, const Type*> >& getMembers() const {
        return m_members;
    }

    const std::map<std::string, const Function*>& getMemberFunctions() const {
        return m_memberFunctions;
    }

    const std::map<std::string, const Function*>& getStaticFunctions() const {
        return m_staticFunctions;
    }

    const std::map<std::string, PyObject*>& getClassMembers() const {
        return m_classMembers;
    }
    
    const std::vector<size_t>& getOffsets() const {
        return m_byte_offsets;
    }

    int memberNamed(const char* c) const {
        for (long k = 0; k < m_members.size(); k++) {
            if (m_members[k].first == c) {
                return k;
            }
        }

        return -1;
    }

private:
    std::vector<size_t> m_byte_offsets;

    std::vector<std::pair<std::string, const Type*> > m_members;

    std::map<std::string, const Function*> m_memberFunctions;
    std::map<std::string, const Function*> m_staticFunctions;
    std::map<std::string, PyObject*> m_classMembers;
};

class PackedArray : public Type {
public:
    void constructor(instance_ptr self) const {
    
    }
    
    template<class buf_t>
    void serialize(instance_ptr self, buf_t& buffer) const {
    }

    template<class buf_t>
    void deserialize(instance_ptr self, buf_t& buffer) const {
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
    
    template<class buf_t>
    void serialize(instance_ptr self, buf_t& buffer) const {
    }
    
    template<class buf_t>
    void deserialize(instance_ptr self, buf_t& buffer) const {
    }

    void destroy(instance_ptr self) const {

    }

    void copy_constructor(instance_ptr self, instance_ptr other) const {

    }

    void assign(instance_ptr self, instance_ptr other) const {

    }
};


