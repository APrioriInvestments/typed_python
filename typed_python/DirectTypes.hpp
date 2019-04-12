#pragma once

#include "Type.hpp"
#include "PyInstance.hpp"

class String {
    public:
        static StringType* getType() {
            static StringType* t = StringType::Make();
            return t;
        }

        String() {
            mLayout = nullptr;
        }

        explicit String(const char *pc) {
            getType()->constructor((instance_ptr)&mLayout, 1, strlen(pc), pc);
        }

        explicit String(std::string& s) {
            getType()->constructor((instance_ptr)&mLayout, 1, s.length(), s.data());
        }

        explicit String(StringType::layout* l) {
            getType()->assign(
               (instance_ptr)&mLayout,
               (instance_ptr)&l
               );
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

//templates for TupleOf, ListOf, Dict, ConstDict, OneOf
template<class element_type>
class ListOf {
public:
    static ListOfType* getType() {
        static ListOfType* t = ListOfType::Make(TypeDetails<element_type>::getType());

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

    ListOf() {
        mLayout = nullptr;
    }

    ~ListOf() {
        getType()->destroy((instance_ptr)&mLayout);
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

    ListOf& operator=(const ListOf& other)
    {
        getType()->assign(
               (instance_ptr)&mLayout,
               (instance_ptr)&other.mLayout
               );
        return *this;
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

    template<class buf_t>
    void serialize(buf_t& buffer) {
        getType()->serialize<buf_t>((instance_ptr)&mLayout, buffer);
    }

    template<class buf_t>
    void deserialize(buf_t& buffer) {
        getType()->deserialize<buf_t>((instance_ptr)&mLayout, buffer);
    }

    TupleOf() {
        mLayout = nullptr;
    }

    ~TupleOf() {
        getType()->destroy((instance_ptr)&mLayout);
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

    TupleOf& operator=(const TupleOf& other)
    {
        getType()->assign(
               (instance_ptr)&mLayout,
               (instance_ptr)&other.mLayout
               );
        return *this;
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

    Dict() {
        mLayout = nullptr;
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
    const value_type lookupValueByKey(key_type k) {
        return *(value_type*)(getType()->lookupValueByKey((instance_ptr)&mLayout, (instance_ptr)&k));
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

    ConstDict() {
        mLayout = nullptr;
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
    const value_type lookupValueByKey(key_type k) {
        return *(value_type*)(getType()->lookupValueByKey((instance_ptr)&mLayout, (instance_ptr)&k));
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

template<class... Ts>
class OneOf;

template<class T1, class... Ts>
class OneOf<T1, Ts...> {
public:
    static const size_t m_datasize = std::max(TypeDetails<T1>::bytecount, OneOf<Ts...>::m_datasize);
    class layout {
        uint8_t which;
        uint8_t data[OneOf<T1, Ts...>::m_datasize];
    };
private:
    layout mLayout;
public:
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

    OneOf() {
        getType()->constructor((instance_ptr)&mLayout);
    }

    ~OneOf() {
        getType()->destroy((instance_ptr)&mLayout);
    }

    OneOf(const OneOf& other)
    {
        getType()->copy_constructor(
               (instance_ptr)&mLayout,
               (instance_ptr)&other.mLayout
               );
    }

    OneOf(OneOf<T1, Ts...>::layout* l) {
        getType()->copy_constructor(
               (instance_ptr)&mLayout,
               (instance_ptr)&l
               );
    }

    OneOf& operator=(const OneOf& other)
    {
        getType()->assign(
               (instance_ptr)&mLayout,
               (instance_ptr)&other.mLayout
               );
        return *this;
    }
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
