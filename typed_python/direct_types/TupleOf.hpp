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

#include <vector>
#include "../Type.hpp"
#include "../PyInstance.hpp"

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
