#pragma once

#include <Python.h>
#include <string>
#include <vector>
#include <algorithm>
#include <mutex>
#include <set>
#include <utility>
#include <atomic>
#include <iostream>

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
class Class;
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
            mTypeRep(nullptr),
            m_base(nullptr)
        {}

    TypeCategory m_typeCategory;
    
    size_t m_size;

    bool m_is_default_constructible;

    std::string m_name;

    PyTypeObject* mTypeRep;

    const Type* m_base;
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
        
        m_name = "OneOf(";
        bool first = true;
        for (auto t: types) {
            if (first) { 
                first = false;
            } else {
                m_name += ",";
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

    char cmp(instance_ptr left, instance_ptr right) const {
        for (long k = 0; k < m_types.size(); k++) {
            char res = m_types[k]->cmp(left + m_byte_offsets[k], right + m_byte_offsets[k]);
            if (res != 0) {
                return res;
            }
        }

        return 0;
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

        m_name = "NamedTuple(";
        for (long k = 0; k < types.size();k++) {
            if (k) {
                m_name += ",";
            }
            m_name += names[k] + "=" + types[k]->name();
        }
        m_name += ")";
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
        m_name = "Tuple(";
        for (long k = 0; k < types.size();k++) {
            if (k) {
                m_name += ",";
            }
            m_name += types[k]->name();
        }
        m_name += ")";
    }

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
        m_name = "TupleOf(" + type->name() + ")";
        m_size = sizeof(void*);
        m_is_default_constructible = true;
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
        m_name = "ConstDict(" + key->name() + "->" + value->name() + ")";
        m_size = sizeof(void*);
        m_is_default_constructible = true;
        m_bytes_per_key = m_key->bytecount();
        m_bytes_per_key_value_pair = m_key->bytecount() + m_value->bytecount();
        m_bytes_per_key_subtree_pair = m_key->bytecount() + this->bytecount();
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

    void incKvPairCount(instance_ptr self) const {
        layout& record = **(layout**)self;
        record.count++;
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

    instance_ptr lookupValueByKey(instance_ptr self, instance_ptr key) const {
        if (!(*(layout**)self)) {
            return 0;
        }

        layout& record = **(layout**)self;

        assert(record.subpointers == 0); //this is not implemented yet

        long low = 0;
        long high = record.count;

        while (low < high) {
            long mid = (low+high)/2;
            char res = m_key->cmp(kvPairPtrKey(self, mid), key);
            
            if (res == 0) {
                return kvPairPtrValue(self, mid);
            } else if (res < 0) {
                low = mid+1;
            } else {
                high = mid;
            }
        }

        return 0;
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
                ((Type*)this)->destroy(record.subpointers, [&](long ix) { 
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
    
    
    Type* keyType() const { return m_key; }
    Type* valueType() const { return m_value; }

private:
    Type* m_key;
    Type* m_value;
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
        
        ::memcpy((*(layout**)self)->data, data, count * bytes_per_codepoint);
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
        (*(layout**)self) = (layout*)malloc(sizeof(layout) + count);

        (*(layout**)self)->bytecount = count;
        (*(layout**)self)->refcount = 1;
        (*(layout**)self)->hash_cache = -1;
        
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

    char cmp(instance_ptr left, instance_ptr right) const {
        return 0;
    }

    int32_t hash32(instance_ptr left) const {
        return m_type->hash32((uint8_t*)&m_data[0]);
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
                return -1;
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

    Type* elementType() const {
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

inline void Alternative::constructor(instance_ptr self) const {
    if (!m_default_construction_type) {
        m_default_construction_type = ConcreteAlternative::Make(this, m_default_construction_ix);
    }

    m_default_construction_type->constructor(self);
}


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


