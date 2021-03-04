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

#include "AllTypes.hpp"
#include "Instance.hpp"

Instance Instance::create(bool val) {
    return create(Bool::Make(), (instance_ptr)&val);
}

Instance Instance::create(int64_t val) {
    return create(Int64::Make(), (instance_ptr)&val);
}

Instance Instance::create(uint64_t val) {
    return create(UInt64::Make(), (instance_ptr)&val);
}

Instance Instance::create(double val) {
    return create(Float64::Make(), (instance_ptr)&val);
}

Instance Instance::create(Type*t, instance_ptr data) {
    t->assertForwardsResolvedSufficientlyToInstantiate();

    return createAndInitialize(t, [&](instance_ptr tgt) {
        t->copy_constructor(tgt, data);
    });
}
Instance Instance::create(Type*t) {
    t->assertForwardsResolvedSufficientlyToInstantiate();

    return createAndInitialize(t, [&](instance_ptr tgt) {
        t->constructor(tgt);
    });
}

Instance::Instance() {
    // by default, None
    mLayout = nullptr;
}

Instance::Instance(const Instance& other) : mLayout(other.mLayout) {
    if (mLayout) {
        mLayout->refcount++;
    }
}

Instance::Instance(instance_ptr p, Type* t) : mLayout(nullptr) {
    if (t->isNone()) {
        return;
    }

    t->assertForwardsResolvedSufficientlyToInstantiate();

    layout* l = (layout*)tp_malloc(sizeof(layout) + t->bytecount());

    try {
        t->copy_constructor(l->data, p);
    } catch(...) {
        tp_free(l);
        throw;
    }

    l->refcount = 1;
    l->type = t;

    mLayout = l;
}

Instance::~Instance() {
    if (mLayout && mLayout->refcount.fetch_sub(1) == 1) {
        mLayout->type->destroy(mLayout->data);
        tp_free(mLayout);
    }
}

Instance& Instance::operator=(const Instance& other) {
    if (other.mLayout) {
        other.mLayout->refcount++;
    }

    if (mLayout && mLayout->refcount.fetch_sub(1) == 1) {
        mLayout->type->destroy(mLayout->data);
        tp_free(mLayout);
    }

    mLayout = other.mLayout;
    return *this;
}

bool Instance::operator<(const Instance& other) const {
    if (!mLayout && !other.mLayout) {
        return false;
    }
    if (!mLayout) {
        return true;
    }
    if (!other.mLayout) {
        return false;
    }
    if (mLayout->type < other.mLayout->type) {
        return true;
    }
    if (mLayout->type > other.mLayout->type) {
        return false;
    }
    return mLayout->type->cmp(mLayout->data, other.mLayout->data, Py_LT, true);
}

std::string Instance::repr() const {
    if (!mLayout) {
        return "None";
    }

    std::ostringstream s;
    ReprAccumulator accumulator(s);

    accumulator << std::showpoint;
    mLayout->type->repr(mLayout->data, accumulator, false);
    return s.str();
}

typed_python_hash_type Instance::hash() const {
    if (!mLayout) {
        return 0;
    }

    return mLayout->type->hash(mLayout->data);
}
