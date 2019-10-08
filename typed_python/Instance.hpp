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


class Instance {
private:
    class layout {
    public:
        std::atomic<int64_t> refcount;
        Type* type;
        uint8_t data[];
    };

    Instance(layout* l);

    static layout* allocateNoneLayout();

    static layout* noneLayout();

public:
    static Instance create(bool val);

    static Instance create(long val);

    static Instance create(unsigned long val);

    static Instance create(double val);

    static Instance create(Type*t, instance_ptr data);

    static Instance create(Type*t);

    Instance();

    Instance(const Instance& other);

    Instance(instance_ptr p, Type* t);

    template<class initializer_type>
    static Instance createAndInitialize(Type* t, const initializer_type& initFun) {
        return Instance(t, initFun);
    }

    template<class initializer_type>
    Instance(Type* t, const initializer_type& initFun) : mLayout(nullptr) {
        t->assertForwardsResolved();

        layout* l = (layout*)malloc(sizeof(layout) + t->bytecount());

        try {
            initFun(l->data);
        } catch(...) {
            free(l);
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
        return mLayout->type;
    }

    instance_ptr data() const {
        return mLayout->data;
    }

    int64_t refcount() const {
        return mLayout->refcount;
    }

private:
    layout* mLayout;
};
