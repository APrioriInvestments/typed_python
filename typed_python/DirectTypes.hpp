#pragma once

#include <vector>
#include "Type.hpp"
#include "PyInstance.hpp"


//templates for TupleOf, ListOf, Dict, ConstDict, OneOf
template<class element_type>
class ListOf {
public:
    static ListOfType* getType() {
        static ListOfType* t = ListOfType::Make(TypeDetails<element_type>::getType());
        return t;
    }

    static Type* getEltType() {
        static Type* t = getType()->getEltType();
        return t;
    }

    static ListOf<element_type> fromPython(PyObject* p) {
        ListOfType::layout* l = nullptr;
        PyInstance::copyConstructFromPythonInstance(getType(), (instance_ptr)&l, p, true);
        return (ListOf<element_type>)l;
    }

    element_type& operator[](int64_t offset) {
       return *(element_type*)((uint8_t*)mLayout->data + TypeDetails<element_type>::bytecount * offset);
    }
 
    const element_type& operator[](int64_t offset) const {
       return *(element_type*)((uint8_t*)mLayout->data + TypeDetails<element_type>::bytecount * offset);
    }

    size_t size() const {
        return !mLayout ? 0 : sizeof(*mLayout) + TypeDetails<element_type>::bytecount * mLayout->count;
    }

    size_t count() const {
        return !mLayout ? 0 : mLayout->count;
    }

    void append(const element_type& e) {
        getType()->append((instance_ptr)&mLayout, (instance_ptr)&e);
    }

    template<class buf_t>
    void serialize(buf_t& buffer) {
        getType()->serialize<buf_t>((instance_ptr)&mLayout, buffer);
    }

    template<class buf_t>
    void deserialize(buf_t& buffer) {
        getType()->deserialize<buf_t>((instance_ptr)&mLayout, buffer);
    }

    ListOf(): mLayout(0) {
        getType()->constructor((instance_ptr)&mLayout);
    }

    ListOf(const std::vector<element_type>& v): mLayout(0) {
        getType()->constructor((instance_ptr)&mLayout, v.size(),
            [&](instance_ptr target, int64_t k) {
                getEltType()->constructor(target);
                getEltType()->assign(target, (instance_ptr)(&v[k]));
            }
        );
    }

    ListOf(const ListOf& other)
    {
        getType()->copy_constructor(
               (instance_ptr)&mLayout,
               (instance_ptr)&other.mLayout
               );
    }

    ListOf(ListOfType::layout* l) {
        getType()->copy_constructor(
               (instance_ptr)&mLayout,
               (instance_ptr)&l
               );
    }

    ~ListOf() {
        getType()->destroy((instance_ptr)&mLayout);
    }

    ListOf& operator=(const ListOf& other)
    {
        getType()->assign(
               (instance_ptr)&mLayout,
               (instance_ptr)&other.mLayout
               );
        return *this;
    }

    ListOfType::layout* getLayout() const {
        return mLayout;
    }

private:
    ListOfType::layout* mLayout;
};

template<class element_type>
class TypeDetails<ListOf<element_type>> {
public:
    static Type* getType() {
        static Type* t = ListOfType::Make(TypeDetails<element_type>::getType());
        if (t->bytecount() != bytecount) {
            throw std::runtime_error("somehow we have the wrong bytecount!");
        }
        return t;
    }

    static const uint64_t bytecount = sizeof(void*);
};

template<class element_type>
class TupleOf {
public:
    static TupleOfType* getType() {
        static TupleOfType* t = TupleOfType::Make(TypeDetails<element_type>::getType());

        return t;
    }

    static Type* getEltType() {
        static Type* t = getType()->getEltType();
        return t;
    }

    static TupleOf<element_type> fromPython(PyObject* p) {
        TupleOfType::layout* l = nullptr;
        PyInstance::copyConstructFromPythonInstance(getType(), (instance_ptr)&l, p, true);
        return (TupleOf<element_type>)l;
    }

    element_type& operator[](int64_t offset) {
       return *(element_type*)((uint8_t*)mLayout->data + TypeDetails<element_type>::bytecount * offset);
    }

    const element_type& operator[](int64_t offset) const {
       return *(element_type*)((uint8_t*)mLayout->data + TypeDetails<element_type>::bytecount * offset);
    }

    size_t size() const {
        return !mLayout ? 0 : sizeof(*mLayout) + TypeDetails<element_type>::bytecount * mLayout->count;
    }

    size_t count() const {
        return !mLayout ? 0 : mLayout->count;
    }

    template<class buf_t>
    void serialize(buf_t& buffer) {
        getType()->serialize<buf_t>((instance_ptr)&mLayout, buffer);
    }

    template<class buf_t>
    void deserialize(buf_t& buffer) {
        getType()->deserialize<buf_t>((instance_ptr)&mLayout, buffer);
    }

    TupleOf(): mLayout(0) {
        getType()->constructor((instance_ptr)&mLayout);
    }

    TupleOf(const TupleOf& other)
    {
        getType()->copy_constructor(
               (instance_ptr)&mLayout,
               (instance_ptr)&other.mLayout
               );
    }

    TupleOf(TupleOfType::layout* l) {
        getType()->copy_constructor(
               (instance_ptr)&mLayout,
               (instance_ptr)&l
               );
    }

    TupleOf(const std::vector<element_type>& v): mLayout(0) {
        getType()->constructor((instance_ptr)&mLayout, v.size(),
            [&](instance_ptr target, int64_t k) {
                getEltType()->constructor(target);
                getEltType()->assign(target, (instance_ptr)(&v[k]));
            }
        );
    }

    ~TupleOf() {
        getType()->destroy((instance_ptr)&mLayout);
    }

    TupleOf& operator=(const TupleOf& other)
    {
        getType()->assign(
               (instance_ptr)&mLayout,
               (instance_ptr)&other.mLayout
               );
        return *this;
    }

    TupleOfType::layout* getLayout() const {
        return mLayout;
    }

private:
    TupleOfType::layout* mLayout;
};

template<class element_type>
class TypeDetails<TupleOf<element_type>> {
public:
    static Type* getType() {
        static Type* t = TupleOf<element_type>::getType();
        if (t->bytecount() != bytecount) {
            throw std::runtime_error("somehow we have the wrong bytecount!");
        }
        return t;
    }

    static const uint64_t bytecount = sizeof(void*);
};

class String {
public:
    static StringType* getType() {
        static StringType* t = StringType::Make();
        return t;
    }

    String():mLayout(0) {}

    explicit String(const char *pc):mLayout(0) {
        getType()->constructor((instance_ptr)&mLayout, 1, strlen(pc), pc);
    }

    explicit String(std::string& s):mLayout(0) {
        getType()->constructor((instance_ptr)&mLayout, 1, s.length(), s.data());
    }

    explicit String(StringType::layout* l):mLayout(0) {
        getType()->constructor((instance_ptr)&mLayout, l->bytes_per_codepoint, l->pointcount, (const char*)l->data);
    }

    ~String() {
        StringType::destroyStatic((instance_ptr)&mLayout);
    }

    String(const String& other)
    {
        getType()->copy_constructor(
           (instance_ptr)&mLayout,
           (instance_ptr)&other.mLayout
           );
    }

    String& operator=(const String& other)
    {
        getType()->assign(
           (instance_ptr)&mLayout,
           (instance_ptr)&other.mLayout
           );
        return *this;
    }

    template<class buf_t>
    void serialize(buf_t& buffer) {
        getType()->serialize<buf_t>((instance_ptr)&mLayout, buffer);
    }

    template<class buf_t>
    void deserialize(buf_t& buffer) {
        getType()->deserialize<buf_t>((instance_ptr)&mLayout, buffer);
    }

    static inline char cmp(const String& left, const String& right) {
        return StringType::cmpStatic(left.mLayout, right.mLayout);
    }
    bool operator==(const String& right) const { return cmp(*this, right) == 0; }
    bool operator!=(const String& right) const { return cmp(*this, right) != 0; }
    bool operator<(const String& right) const { return cmp(*this, right) < 0; }
    bool operator>(const String& right) const { return cmp(*this, right) > 0; }
    bool operator<=(const String& right) const { return cmp(*this, right) <= 0; }
    bool operator>=(const String& right) const { return cmp(*this, right) >= 0; }

    static String concatenate(const String& left, const String& right) {
        return String(StringType::concatenate(left.getLayout(), right.getLayout()));
    }
    String operator+(const String& right) {
        return String(StringType::concatenate(mLayout, right.getLayout()));
    }
    String& operator+=(const String& right) {
        return *this = String(StringType::concatenate(mLayout, right.getLayout()));
    }
    String upper() const { return String(StringType::upper(mLayout)); }
    String lower() const { return String(StringType::lower(mLayout)); }
    int64_t find(const String& sub, int64_t start, int64_t end) const { return StringType::find(mLayout, sub.getLayout(), start, end); }
    int64_t find(const String& sub, int64_t start) const { return StringType::find(mLayout, sub.getLayout(), start, mLayout ? mLayout->pointcount : 0); }
    int64_t find(const String& sub) const { return StringType::find(mLayout, sub.getLayout(), 0, mLayout ? mLayout->pointcount : 0); }
    ListOf<String> split(const String& sep, int64_t max = -1) const {
       ListOf<String> ret;
       StringType::split(ret.getLayout(), mLayout, sep.getLayout(), max);
       return ret;
    }
    ListOf<String> split(int64_t max = -1) {
       ListOf<String> ret;
       StringType::split_3(ret.getLayout(), mLayout, max);
       return ret;
    }
    bool isalpha() const { return StringType::isalpha(mLayout); }
    bool isalnum() const { return StringType::isalnum(mLayout); }
    bool isdecimal() const { return StringType::isdecimal(mLayout); }
    bool isdigit() const { return StringType::isdigit(mLayout); }
    bool islower() const { return StringType::islower(mLayout); }
    bool isnumeric() const { return StringType::isnumeric(mLayout); }
    bool isprintable() const { return StringType::isprintable(mLayout); }
    bool isspace() const { return StringType::isspace(mLayout); }
    bool istitle() const { return StringType::istitle(mLayout); }
    bool isupper() const { return StringType::isupper(mLayout); }

    String substr(int64_t start, int64_t stop) const { return String(StringType::getsubstr(mLayout, start, stop)); }

    StringType::layout* getLayout() const {
        return mLayout;
    }

private:
    StringType::layout* mLayout;
};

template<>
class TypeDetails<String> {
public:
    static Type* getType() {
        static Type* t = StringType::Make();
        if (t->bytecount() != bytecount) {
            throw std::runtime_error("somehow we have the wrong bytecount!");
        }
        return t;
    }

    static const uint64_t bytecount = sizeof(void*);
};

template<class key_type, class value_type>
class Dict {
public:
    static DictType* getType() {
        static DictType* t = DictType::Make(
            TypeDetails<key_type>::getType(),
            TypeDetails<value_type>::getType()
        );

        return t;
    }

    static Dict<key_type, value_type> fromPython(PyObject* p) {
        DictType::layout* l = nullptr;
        PyInstance::copyConstructFromPythonInstance(getType(), (instance_ptr)&l, p, true);
        return (Dict<key_type, value_type>)l;
    }

    size_t size() const {
        return !mLayout ? 0 : sizeof(*mLayout) ; //+ TypeDetails<element_type>::bytecount * mLayout->count;
    }

    template<class buf_t>
    void serialize(buf_t& buffer) {
        getType()->serialize<buf_t>((instance_ptr)&mLayout, buffer);
    }

    template<class buf_t>
    void deserialize(buf_t& buffer) {
        getType()->deserialize<buf_t>((instance_ptr)&mLayout, buffer);
    }

    Dict(): mLayout(0) {
        getType()->constructor((instance_ptr)&mLayout);
    }

    ~Dict() {
        getType()->destroy((instance_ptr)&mLayout);
    }

    Dict(const Dict& other)
    {
        getType()->copy_constructor(
               (instance_ptr)&mLayout,
               (instance_ptr)&other.mLayout
               );
    }

    Dict(DictType::layout* l) {
        getType()->copy_constructor(
               (instance_ptr)&mLayout,
               (instance_ptr)&l
               );
    }

    Dict& operator=(const Dict& other)
    {
        getType()->assign(
               (instance_ptr)&mLayout,
               (instance_ptr)&other.mLayout
               );
        return *this;
    }

    void insertKeyValue(const key_type& k, const value_type& v) {
        value_type *pv = (value_type*)getType()->lookupValueByKey((instance_ptr)&mLayout, (instance_ptr)&k);
        if (pv) {
            TypeDetails<value_type>::getType()->assign((instance_ptr)pv,(instance_ptr)&v);
        } else {
            pv = (value_type*)getType()->insertKey((instance_ptr)&mLayout, (instance_ptr)&k);
            TypeDetails<value_type>::getType()->copy_constructor((instance_ptr)pv, (instance_ptr)&v);
        }
    }
    const value_type* lookupValueByKey(const key_type&  k) const {
        return (value_type*)(getType()->lookupValueByKey((instance_ptr)&mLayout, (instance_ptr)&k));
    }

    bool deleteKey(const key_type& k) {
        return getType()->deleteKey((instance_ptr)&mLayout, (instance_ptr)&k);
    }

    DictType::layout* getLayout() const {
        return mLayout;
    }

private:
    DictType::layout* mLayout;
};

template<class key_type, class value_type>
class TypeDetails<Dict<key_type, value_type>> {
public:
    static Type* getType() {
        static Type* t = Dict<key_type, value_type>::getType();
        if (t->bytecount() != bytecount) {
            throw std::runtime_error("somehow we have the wrong bytecount!");
        }
        return t;
    }

    static const uint64_t bytecount = sizeof(void*);
};

template<class key_type, class value_type>
class ConstDict {
public:
    static ConstDictType* getType() {
        static ConstDictType* t = ConstDictType::Make(
            TypeDetails<key_type>::getType(),
            TypeDetails<value_type>::getType()
        );

        return t;
    }

    static ConstDict<key_type, value_type> fromPython(PyObject* p) {
        ConstDictType::layout* l = nullptr;
        PyInstance::copyConstructFromPythonInstance(getType(), (instance_ptr)&l, p, true);
        return (ConstDict<key_type, value_type>)l;
    }

    size_t size() const {
        return !mLayout ? 0 : sizeof(*mLayout) ; //+ TypeDetails<element_type>::bytecount * mLayout->count;
    }

    template<class buf_t>
    void serialize(buf_t& buffer) {
        getType()->serialize<buf_t>((instance_ptr)&mLayout, buffer);
    }

    template<class buf_t>
    void deserialize(buf_t& buffer) {
        getType()->deserialize<buf_t>((instance_ptr)&mLayout, buffer);
    }

    ConstDict(): mLayout(0) {
        getType()->constructor((instance_ptr)&mLayout);
    }

    ~ConstDict() {
        getType()->destroy((instance_ptr)&mLayout);
    }

    ConstDict(const ConstDict& other)
    {
        getType()->copy_constructor(
               (instance_ptr)&mLayout,
               (instance_ptr)&other.mLayout
               );
    }

    ConstDict(ConstDictType::layout* l) {
        getType()->copy_constructor(
               (instance_ptr)&mLayout,
               (instance_ptr)&l
               );
    }

    ConstDict& operator=(const ConstDict& other)
    {
        getType()->assign(
               (instance_ptr)&mLayout,
               (instance_ptr)&other.mLayout
               );
        return *this;
    }

    const value_type* lookupValueByKey(const key_type&  k) const {
        return (value_type*)(getType()->lookupValueByKey((instance_ptr)&mLayout, (instance_ptr)&k));
    }

    ConstDictType::layout* getLayout() const {
        return mLayout;
    }

private:
    ConstDictType::layout* mLayout;
};

template<class key_type, class value_type>
class TypeDetails<ConstDict<key_type, value_type>> {
public:
    static Type* getType() {
        static Type* t = ConstDict<key_type, value_type>::getType();
        if (t->bytecount() != bytecount) {
            throw std::runtime_error("somehow we have the wrong bytecount!");
        }
        return t;
    }

    static const uint64_t bytecount = sizeof(void*);
};


// TypeIndex calculates the index k of a type T in a parameter pack Ts...
// compile-time error if not found
template<class T, int k, class... Ts>
struct TypeIndex;

template<class T, int k>
struct TypeIndex<T, k> {
    static_assert(sizeof(T)==0, "TypeIndex: type T not found in parameter pack");
};

template<class T, int k, class... Ts>
struct TypeIndex<T, k, T, Ts...> {
    static const int value = k;
};

template<class T, int k, class T1, class... Ts>
struct TypeIndex<T, k, T1, Ts...> {
    static const int value = TypeIndex<T, k+1, Ts...>::value;
};


template<class... Ts>
class OneOf;

template<class T1, class... Ts>
class OneOf<T1, Ts...> {
public:
    static const size_t m_datasize = std::max(TypeDetails<T1>::bytecount, OneOf<Ts...>::m_datasize);
    struct layout {
        uint8_t which;
        uint8_t data[OneOf<T1, Ts...>::m_datasize];
    };

    const layout* getLayout() const { return &mLayout; }

    static OneOfType* getType() {
        static OneOfType* t = OneOfType::Make({TypeDetails<T1>::getType(), TypeDetails<Ts>::getType()...});

        return t;
    }

    static OneOf<T1, Ts...> fromPython(PyObject* p) {
        OneOf<T1, Ts...>::layout* l = nullptr;
        PyInstance::copyConstructFromPythonInstance(getType(), (instance_ptr)&l, p, true);
        return (OneOf<T1, Ts...>)l;
    }

    std::pair<Type*, instance_ptr> unwrap() {
        return getType()->unwrap((instance_ptr)&mLayout);
    }

    template <class T>
    bool getValue(T& o) {
        static const int k = TypeIndex<T, 0, T1, Ts...>::value;
        if (k != mLayout.which)
            return false;
        getType()->getTypes()[k]->copy_constructor((instance_ptr)&o, (instance_ptr)&mLayout.data);
        return true;
    }

    size_t size() const {
        return sizeof(layout);
    }

    template<class buf_t>
    void serialize(buf_t& buffer) {
        getType()->serialize<buf_t>((instance_ptr)&mLayout, buffer);
    }

    template<class buf_t>
    void deserialize(buf_t& buffer) {
        getType()->deserialize<buf_t>((instance_ptr)&mLayout, buffer);
    }

    template <class T>
    OneOf(const T& o) {
        getType()->constructor((instance_ptr)&mLayout);
        static const int k = TypeIndex<T, 0, T1, Ts...>::value;
        mLayout.which = k;
        getType()->getTypes()[k]->copy_constructor((instance_ptr)&mLayout.data, (instance_ptr)&o);
    }

    OneOf() {
        getType()->constructor((instance_ptr)&mLayout);
    }

    ~OneOf() {
        getType()->destroy((instance_ptr)&mLayout);
    }

    OneOf(const OneOf<T1, Ts...>& other)
    {
        getType()->copy_constructor(
               (instance_ptr)&mLayout,
               (instance_ptr)&other.mLayout
               );
    }

    explicit OneOf(OneOf<T1, Ts...>::layout* l) {
        getType()->copy_constructor(
               (instance_ptr)&mLayout,
               (instance_ptr)&l
               );
    }

    OneOf<T1, Ts...>& operator=(const OneOf<T1, Ts...>& other)
    {
        getType()->assign(
               (instance_ptr)&mLayout,
               (instance_ptr)&other.mLayout
               );
        return *this;
    }
private:
    layout mLayout;
};

template <>
class OneOf<> {
public:
    static const size_t m_datasize = 0;
};

template<class... Ts>
class TypeDetails<OneOf<Ts...>> {
public:
    static Type* getType() {
        static Type* t = OneOf<Ts...>::getType();
        if (t->bytecount() != bytecount) {
            throw std::runtime_error("somehow we have the wrong bytecount!");
        }
        return t;
    }

    static const uint64_t bytecount = 1 + OneOf<Ts...>::m_datasize;
};



