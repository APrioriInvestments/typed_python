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

#include "../typed_python/AllTypes.hpp"
#include "../typed_python/Instance.hpp"
#include "../typed_python/DictType.hpp"

/*******

A convenience wrapper around an Instance holding a Dict object.

*******/

template<class key_type, class value_type>
class DictInstance {
public:
    DictInstance(Type* keyType, Type* valueType) :
            mKeyType(keyType),
            mValueType(valueType)
    {
        mInstance = Instance::create(DictType::Make(keyType, valueType));
    }

    Type* getKeyType() const {
        return mKeyType;
    }

    Type* getValueType() const {
        return mValueType;
    }

    value_type* lookupKey(const key_type& key) {
        return (value_type*)((DictType*)mInstance.type())->lookupValueByKey(mInstance.data(), (instance_ptr)&key);
    }

    value_type* insertKey(const key_type& key) {
        return (value_type*)((DictType*)mInstance.type())->insertKey(mInstance.data(), (instance_ptr)&key);
    }

    bool deleteKey(const key_type& key) {
        return ((DictType*)mInstance.type())->deleteKey(mInstance.data(), (instance_ptr)&key);
    }

    value_type* lookupOrInsert(const key_type& key) {
        auto resPtr = lookupKey(key);

        if (resPtr) {
            return resPtr;
        }

        return insertKey(key);
    }

    size_t size() const {
        return ((DictType*)mInstance.type())->size(mInstance.data());
    }

private:
    Instance mInstance;
    Type* mKeyType;
    Type* mValueType;
};

