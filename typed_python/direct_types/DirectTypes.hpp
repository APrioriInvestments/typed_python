#pragma once

#include <vector>
#include "../Type.hpp"
#include "../PyInstance.hpp"


//templates for TupleOf, ListOf, Dict, ConstDict, OneOf
template<class element_type>
class ListOf {
public:
    class iterator_type {
    public:
        iterator_type(ListOfType::layout* in_tuple, int64_t in_offset) :
            m_tuple(in_tuple),
            m_offset(in_offset)
        {}

        bool operator!=(const iterator_type& other) const {
            return m_tuple != other.m_tuple || m_offset != other.m_offset;
        }

        bool operator==(const iterator_type& other) const {
            return m_tuple == other.m_tuple && m_offset == other.m_offset;
        }

        iterator_type& operator++() {
            m_offset++;
            return *this;
        }

        iterator_type operator++(int) {
            return iterator_type(m_tuple, m_offset++);
        }

        const element_type& operator*() const {
            return *(element_type*)getType()->eltPtr((instance_ptr)&m_tuple, m_offset);
        }

    private:
        ListOfType::layout* m_tuple;
        int64_t m_offset;
    };

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

    PyObject* toPython() {
        return PyInstance::extractPythonObject((instance_ptr)&mLayout, getType());
    }

    element_type& operator[](int64_t offset) {
       return *(element_type*)((uint8_t*)mLayout->data + TypeDetails<element_type>::bytecount * offset);
    }
 
    const element_type& operator[](int64_t offset) const {
       return *(element_type*)((uint8_t*)mLayout->data + TypeDetails<element_type>::bytecount * offset);
    }

    size_t size() const {
        return mLayout ? mLayout->count : 0;
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

    ListOf(const ListOf& other) {
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

    ListOf& operator=(const ListOf& other) {
        getType()->assign(
               (instance_ptr)&mLayout,
               (instance_ptr)&other.mLayout
               );
        return *this;
    }

    ListOfType::layout* getLayout() const {
        return mLayout;
    }

    iterator_type begin() const {
        return iterator_type(mLayout, 0);
    }

    iterator_type end() const {
        return iterator_type(mLayout, size());
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
            throw std::runtime_error("ListOf: somehow we have the wrong bytecount!");
        }
        return t;
    }

    static const uint64_t bytecount = sizeof(void*);
};

template<class element_type>
class TupleOf {
public:
    class iterator_type {
    public:
        iterator_type(TupleOfType::layout* in_tuple, int64_t in_offset) :
            m_tuple(in_tuple),
            m_offset(in_offset)
        {}

        bool operator!=(const iterator_type& other) const {
            return m_tuple != other.m_tuple || m_offset != other.m_offset;
        }

        bool operator==(const iterator_type& other) const {
            return m_tuple == other.m_tuple && m_offset == other.m_offset;
        }

        iterator_type& operator++() {
            m_offset++;
            return *this;
        }

        iterator_type operator++(int) {
            return iterator_type(m_tuple, m_offset++);
        }

        const element_type& operator*() const {
            return *(element_type*)getType()->eltPtr((instance_ptr)&m_tuple, m_offset);
        }

    private:
        TupleOfType::layout* m_tuple;
        int64_t m_offset;
    };

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

    //repeatedly call 'F' with a pointer to an uninitialized element_type and an index.
    //stop when it returns 'false'
    template<class F>
    static TupleOf<element_type> createUnbounded(const F& allocator) {
        TupleOfType::layout* layoutPtr = nullptr;

        getType()->constructorUnbounded((instance_ptr)&layoutPtr, [&](instance_ptr tgt, int64_t index) {
            return allocator((element_type*)tgt, index);
        });

        return TupleOf<element_type>(layoutPtr);
    }

    PyObject* toPython() {
        return PyInstance::extractPythonObject((instance_ptr)&mLayout, getType());
    }

    PyObject* toPython(Type* elementTypeOverride) {
        return PyInstance::extractPythonObject((instance_ptr)&mLayout, ::TupleOfType::Make(elementTypeOverride));
    }

    element_type& operator[](int64_t offset) {
       return *(element_type*)((uint8_t*)mLayout->data + TypeDetails<element_type>::bytecount * offset);
    }

    const element_type& operator[](int64_t offset) const {
       return *(element_type*)((uint8_t*)mLayout->data + TypeDetails<element_type>::bytecount * offset);
    }

    size_t size() const {
        return mLayout ? mLayout->count : 0;
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

    TupleOf(const TupleOf& other) {
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

    template<class other_t>
    TupleOf<other_t> cast() {
        return TupleOf<other_t>(mLayout);
    }

    TupleOf& operator=(const TupleOf& other) {
        getType()->assign(
               (instance_ptr)&mLayout,
               (instance_ptr)&other.mLayout
               );
        return *this;
    }

    TupleOfType::layout* getLayout() const {
        return mLayout;
    }

    iterator_type begin() const {
        return iterator_type(mLayout, 0);
    }

    iterator_type end() const {
        return iterator_type(mLayout, size());
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
            throw std::runtime_error("TupleOf: somehow we have the wrong bytecount!");
        }
        return t;
    }

    static const uint64_t bytecount = sizeof(void*);
};

class None {
public:
    static NoneType* getType() {
        static NoneType* t = NoneType::Make();
        return t;
    }
    None() {}
    ~None() {}
    None(None& other) {}
    None& operator=(const None& other) { return *this; }
};


template<>
class TypeDetails<None> {
public:
    static Type* getType() {
        static Type* t = None::getType();
        if (t->bytecount() != bytecount) {
            throw std::runtime_error("None: somehow we have the wrong bytecount!");
        }
        return t;
    }

    static const uint64_t bytecount = 0;
};


class Bytes {
public:
    static BytesType* getType() {
        static BytesType* t = BytesType::Make();
        return t;
    }

    Bytes():mLayout(0) {}

    explicit Bytes(const char *pc, size_t bytecount):mLayout(0) {
        getType()->constructor((instance_ptr)&mLayout, bytecount, pc);
    }

    explicit Bytes(const char *pc):mLayout(0) {
        getType()->constructor((instance_ptr)&mLayout, strlen(pc), pc);
    }

    explicit Bytes(const std::string& s):mLayout(0) {
        getType()->constructor((instance_ptr)&mLayout, s.length(), s.data());
    }

    explicit Bytes(BytesType::layout* l):mLayout(0) {
        getType()->constructor((instance_ptr)&mLayout, l->bytecount, (const char*)l->data);
    }

    ~Bytes() {
        getType()->destroy((instance_ptr)&mLayout);
    }

    Bytes(const Bytes& other) {
        getType()->copy_constructor(
           (instance_ptr)&mLayout,
           (instance_ptr)&other.mLayout
           );
    }

    Bytes& operator=(const Bytes& other) {
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

    size_t size() const {
        return getType()->count((instance_ptr)&mLayout);
    }

    int32_t hashValue() const {
        return getType()->hash32((instance_ptr)&mLayout);
    }

    static inline char cmp(const Bytes& left, const Bytes& right) {
        return BytesType::cmpStatic(left.mLayout, right.mLayout);
    }
    bool operator==(const Bytes& right) const { return cmp(*this, right) == 0; }
    bool operator!=(const Bytes& right) const { return cmp(*this, right) != 0; }
    bool operator<(const Bytes& right) const { return cmp(*this, right) < 0; }
    bool operator>(const Bytes& right) const { return cmp(*this, right) > 0; }
    bool operator<=(const Bytes& right) const { return cmp(*this, right) <= 0; }
    bool operator>=(const Bytes& right) const { return cmp(*this, right) >= 0; }

    const uint8_t& operator[](size_t s) const {
        return *(uint8_t*)getType()->eltPtr((instance_ptr)&mLayout, s);
    }

    static Bytes concatenate(const Bytes& left, const Bytes& right) {
        return Bytes(BytesType::concatenate(left.getLayout(), right.getLayout()));
    }

    Bytes operator+(const Bytes& right) {
        return Bytes(BytesType::concatenate(mLayout, right.getLayout()));
    }

    Bytes& operator+=(const Bytes& right) {
        return *this = Bytes(BytesType::concatenate(mLayout, right.getLayout()));
    }

    BytesType::layout* getLayout() const { return mLayout; }
private:
    BytesType::layout* mLayout;
};

template<>
class TypeDetails<Bytes> {
public:
    static Type* getType() {
        static Type* t = BytesType::Make();
        if (t->bytecount() != bytecount) {
            throw std::runtime_error("Bytes: somehow we have the wrong bytecount!");
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

    explicit String(const std::string& s):mLayout(0) {
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

    String& operator=(const char* other) {
        return *this = String(other);
    }

    String& operator=(const std::string& other) {
        return *this = String(other);
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

    size_t size() const {
        return getType()->count((instance_ptr)&mLayout);
    }

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
            throw std::runtime_error("String: somehow we have the wrong bytecount!");
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

    PyObject* toPython() {
        return PyInstance::extractPythonObject((instance_ptr)&mLayout, getType());
    }

    size_t size() const {
        return !mLayout ? 0 : getType()->size((instance_ptr)&mLayout);
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

    Dict(const Dict& other) {
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

    Dict& operator=(const Dict& other) {
        getType()->assign(
               (instance_ptr)&mLayout,
               (instance_ptr)&other.mLayout
               );
        return *this;
    }

    value_type& operator[](const key_type& k) {
        value_type *pv = (value_type*)getType()->lookupValueByKey((instance_ptr)&mLayout, (instance_ptr)&k);

        if (!pv) {
            pv = (value_type*)getType()->insertKey((instance_ptr)&mLayout, (instance_ptr)&k);
            TypeDetails<value_type>::getType()->constructor((instance_ptr)pv);
        }

        return *pv;
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
            throw std::runtime_error("Dict: somehow we have the wrong bytecount!");
        }
        return t;
    }

    static const uint64_t bytecount = sizeof(void*);
};

template<class key_type, class value_type>
class ConstDict {
public:
    class iterator_type {
    public:
        iterator_type(ConstDictType::layout* in_dict, int64_t in_offset) :
                m_dict(in_dict),
                m_offset(in_offset)
        {
        }

        bool operator!=(const iterator_type& other) const {
            return m_dict != other.m_dict || m_offset != other.m_offset;
        }

        bool operator==(const iterator_type& other) const {
            return m_dict == other.m_dict && m_offset == other.m_offset;
        }

        iterator_type& operator++() {
            m_offset++;
            return *this;
        }

        iterator_type operator++(int) {
            return iterator_type(m_dict, m_offset++);
        }

        std::pair<key_type&, value_type&> operator*() const {
            return std::pair<key_type&, value_type&>(
                *(key_type*)getType()->kvPairPtrKey((instance_ptr)&m_dict, m_offset),
                *(value_type*)getType()->kvPairPtrValue((instance_ptr)&m_dict, m_offset)
            );
        }

    private:
        ConstDictType::layout* m_dict;
        int64_t m_offset;
    };

    static ConstDictType* getType() {
        static ConstDictType* t = ConstDictType::Make(
            TypeDetails<key_type>::getType(),
            TypeDetails<value_type>::getType()
        );

        return t;
    }

    static ConstDict fromPython(PyObject* p) {
        ConstDictType::layout* l = nullptr;
        PyInstance::copyConstructFromPythonInstance(getType(), (instance_ptr)&l, p, true);
        return (ConstDict<key_type, value_type>)l;
    }

    // returns the number of items in the ConstDict
    size_t size() const {
        return !mLayout ? 0 : getType()->count((instance_ptr)&mLayout);
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
    ConstDict(std::vector<std::pair<key_type, value_type>> initlist):mLayout(0) {
        size_t count = initlist.size();
        getType()->constructor((instance_ptr)&mLayout, count, false);

        for (size_t k = 0; k < count; k++) {
            bool initkey = false;
            try {
                key_type* pk = (key_type*)getType()->kvPairPtrKey((instance_ptr)&mLayout, k);
                new ((key_type *)pk) key_type(initlist[k].first);
                initkey = true;
                value_type* pv = (value_type*)getType()->kvPairPtrValue((instance_ptr)&mLayout, k);
                new ((value_type *)pv) value_type(initlist[k].second);
            } catch(...) {
                try {
                    if (initkey) {
                        instance_ptr pk = getType()->kvPairPtrKey((instance_ptr)&mLayout, k);
                        TypeDetails<key_type>::getType()->destroy(pk);
                    }
                    for (size_t j = k - 1; j >= 0; j--) {
                        instance_ptr pk = getType()->kvPairPtrKey((instance_ptr)&mLayout, j);
                        TypeDetails<key_type>::getType()->destroy(pk);
                        instance_ptr pv = getType()->kvPairPtrValue((instance_ptr)&mLayout, j);
                        TypeDetails<value_type>::getType()->destroy(pv);
                    }
                } catch(...) {
                }
                throw;
            }
        }
        getType()->incKvPairCount((instance_ptr)&mLayout, count);
        getType()->sortKvPairs((instance_ptr)&mLayout);
    }

    ~ConstDict() {
        getType()->destroy((instance_ptr)&mLayout);
    }

    ConstDict(const ConstDict& other) {
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

    ConstDict& operator=(const ConstDict& other) {
        getType()->assign(
               (instance_ptr)&mLayout,
               (instance_ptr)&other.mLayout
               );
        return *this;
    }

    const value_type* lookupValueByKey(const key_type&  k) const {
        return (value_type*)(getType()->lookupValueByKey((instance_ptr)&mLayout, (instance_ptr)&k));
    }

    static ConstDict add(const ConstDict& lhs, const ConstDict& rhs) {
        ConstDict result;
        getType()->addDicts((instance_ptr)&lhs.mLayout, (instance_ptr)&rhs.mLayout, (instance_ptr)&result.mLayout);
        return result;
    }

    ConstDict operator+(const ConstDict& rhs) {
        return add(*this, rhs);
    }

    static ConstDict subtract(const ConstDict& lhs, const TupleOf<key_type>& keys) {
        ConstDict result;
        TupleOfType::layout *l = keys.getLayout();
        getType()->subtractTupleOfKeysFromDict((instance_ptr)&lhs.mLayout, (instance_ptr)&l, (instance_ptr)&result.mLayout);
        return result;
    }

    ConstDict operator-(const TupleOf<key_type>& keys) {
        return subtract(*this, keys);
    }

    ConstDictType::layout* getLayout() const {
        return mLayout;
    }

    iterator_type begin() const {
        return iterator_type(mLayout, 0);
    }

    iterator_type end() const {
        return iterator_type(mLayout, size());
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
            throw std: :runtime_error("ConstDict: somehow we have the wrong bytecount!");
        }
        return t;
    }

    static const uint64_t bytecount = sizeof(void*);
};


// TypeIndex calculates the first index k of a type T in a parameter pack Ts...
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

    OneOf(const OneOf<T1, Ts...>& other) {
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

    OneOf<T1, Ts...>& operator=(const OneOf<T1, Ts...>& other) {
        getType()->assign(
               (instance_ptr)&mLayout,
               (instance_ptr)&other.mLayout
               );
        return *this;
    }

    const layout* getLayout() const { return &mLayout; }
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
            throw std::runtime_error("OneOf: somehow we have the wrong bytecount!");
        }
        return t;
    }

    static const uint64_t bytecount = 1 + OneOf<Ts...>::m_datasize;
};
