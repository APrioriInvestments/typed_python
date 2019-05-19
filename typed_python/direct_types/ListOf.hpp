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
class ListOf {
public:
    class iterator_type {
    public:
        iterator_type(ListOfType::layout* in_list, int64_t in_offset) :
            m_list(in_list),
            m_offset(in_offset)
        {}

        bool operator!=(const iterator_type& other) const {
            return m_list != other.m_list || m_offset != other.m_offset;
        }

        bool operator==(const iterator_type& other) const {
            return m_list == other.m_list && m_offset == other.m_offset;
        }

        iterator_type& operator++() {
            m_offset++;
            return *this;
        }

        iterator_type operator++(int) {
            return iterator_type(m_list, m_offset++);
        }

        const element_type& operator*() const {
            return *(element_type*)getType()->eltPtr((instance_ptr)&m_list, m_offset);
        }

    private:
        ListOfType::layout* m_list;
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
        return ListOf<element_type>(l);
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
    ListOf(ListOfType::layout* l): mLayout(l) {
        // deliberately stealing a reference
    }

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

