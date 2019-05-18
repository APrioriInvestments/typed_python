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

