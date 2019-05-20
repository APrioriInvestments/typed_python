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

#include "../Type.hpp"
#include "../PyInstance.hpp"

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
        OneOf<T1, Ts...>::layout l;
        PyInstance::copyConstructFromPythonInstance(getType(), (instance_ptr)&l, p, true);
        return OneOf<T1, Ts...>(l);
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

    OneOf<T1, Ts...>& operator=(const OneOf<T1, Ts...>& other) {
        getType()->assign(
               (instance_ptr)&mLayout,
               (instance_ptr)&other.mLayout
               );
        return *this;
    }

    const layout* getLayout() const { return &mLayout; }
private:
    explicit OneOf(layout l): mLayout(l) {
    }

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
