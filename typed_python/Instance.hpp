/******************************************************************************
   Copyright 2017-2019 typed_python Authors

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

#include <atomic>
#include <string>
#include "Type.hpp"

typedef uint8_t* instance_ptr;


class InstanceRef {
public:
    InstanceRef(instance_ptr p, Type* t) :
        mData(p),
        mType(t)
    {
    }

    InstanceRef() :
        mData(nullptr),
        mType(nullptr)
    {
        static Type* noneType = NoneType::Make();

        mType = noneType;
        mData = (instance_ptr)this;
    }

    Type* type() const {
        return mType;
    }

    template<class T>
    T& cast() {
        return *(T*)data();
    }

    instance_ptr data() const {
        return mData;
    }

    bool operator<(const InstanceRef& other) const {
        if (mType < other.mType) { return true; }
        if (mType > other.mType) { return false; }
        if (mData < other.mData) { return true; }
        return false;
    }

    bool operator==(const InstanceRef& other) const {
        return mType == other.mType && mData == other.mData;
    }

    // if we wrap a Type* as a python object, return it. Otherwise nullptr.
    Type* extractType(bool includePrimitives=false) const;

    // if we wrap a PyObject return it. Otherwise nullptr.
    PyObject* extractPyobj() const;

private:
    instance_ptr mData;
    Type* mType;
};


class Instance {
private:
    class layout {
    public:
        std::atomic<int64_t> refcount;
        Type* type;
        uint8_t data[];
    };

    Instance(layout* l);

public:
    static Instance create(bool val);

    static Instance create(int64_t val);

    static Instance create(uint64_t val);

    static Instance create(double val);

    static Instance create(Type*t, instance_ptr data);

    static Instance create(PyObject* o);

    static Instance create(Type*t);

    Instance();

    Instance(const Instance& other);

    Instance(instance_ptr p, Type* t);

    Instance(const InstanceRef& ref);

    InstanceRef ref() const {
        return InstanceRef(data(), type());
    }

    template<class initializer_type>
    static Instance createAndInitialize(Type* t, const initializer_type& initFun) {
        return Instance(t, initFun);
    }

    Instance clone() const {
        return Instance::createAndInitialize(type(), [&](instance_ptr newSelf) {
            type()->copy_constructor(newSelf, data());
        });
    }

    template<class initializer_type>
    Instance(Type* t, const initializer_type& initFun) : mLayout(nullptr) {
        if (t->isNone()) {
            initFun((instance_ptr)this);
            return;
        }

        t->assertForwardsResolvedSufficientlyToInstantiate();

        layout* l = (layout*)tp_malloc(sizeof(layout) + t->bytecount());

        try {
            initFun(l->data);
        } catch(...) {
            tp_free(l);
            throw;
        }

        l->refcount = 1;
        l->type = t;

        mLayout = l;
    }

    ~Instance();

    Instance& operator=(const Instance& other);

    bool operator<(const Instance& other) const;

    std::string repr() const;

    typed_python_hash_type hash() const;

    Type* type() const {
        static Type* noneType = NoneType::Make();

        return mLayout ? mLayout->type : noneType;
    }

    // if we wrap a Type* as a python object, return it. Otherwise nullptr.
    Type* extractType(bool includePrimitives=false) const;

    // if we wrap a PyObject return it. Otherwise nullptr.
    PyObject* extractPyobj() const;

    template<class T>
    T& cast() {
        return *(T*)data();
    }

    instance_ptr data() const {
        if (mLayout) {
            return mLayout->data;
        }

        return (instance_ptr)this;
    }

    int64_t refcount() const {
        if (!mLayout) {
            return 1000000000;
        }

        return mLayout->refcount;
    }

private:
    // if the nullptr, then this is the None object.
    layout* mLayout;
};

namespace std {
    template<>
    struct hash<InstanceRef> {
        typedef InstanceRef argument_type;
        typedef std::size_t result_type;

        result_type operator()(argument_type const& s) const noexcept {
            return std::hash<void*>()((void*)s.data()) ^ std::hash<void*>()((void*)s.type());
        }
    };
}

