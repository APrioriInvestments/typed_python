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
#include "TupleOf.hpp"
#include "../Type.hpp"
#include "../PyInstance.hpp"

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
        return ConstDict<key_type, value_type>(l);
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
    explicit ConstDict(ConstDictType::layout* l): mLayout(l) {
        // deliberately stealing a reference
    }

    ConstDictType::layout* mLayout;
};

template<class key_type, class value_type>
class TypeDetails<ConstDict<key_type, value_type>> {
public:
    static Type* getType() {
        static Type* t = ConstDict<key_type, value_type>::getType();
        if (t->bytecount() != bytecount) {
            throw std::runtime_error("ConstDict: somehow we have the wrong bytecount!");
        }
        return t;
    }

    static const uint64_t bytecount = sizeof(void*);
};

