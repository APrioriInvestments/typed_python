#pragma once

#include <atomic>
#include <string>
// TODO #include "Type.hpp"

class DeserializationBuffer;

typedef uint8_t* instance_ptr;


class Instance {
private:
    class layout {
    public:
        std::atomic<int64_t> refcount;
        Type* type;
        uint8_t data[];
    };

    Instance(layout* l) :
        mLayout(l)
    {
    }

    static layout* allocateNoneLayout() {
        layout* result = (layout*)malloc(sizeof(layout));
        result->refcount = 0;
        result->type = None::Make();

        return result;
    }

    static layout* noneLayout() {
        static layout* noneLayout = allocateNoneLayout();
        noneLayout->refcount++;

        return noneLayout;
    }

public:
    static Instance deserialized(Type* t, DeserializationBuffer& buf) {
        t->assertForwardsResolved();

        return createAndInitialize(t, [&](instance_ptr tgt) {
            t->deserialize(tgt, buf);
        });
    }

    static Instance create(bool val) {
        return create(Bool::Make(), (instance_ptr)&val);
    }

    static Instance create(long val) {
        return create(Int64::Make(), (instance_ptr)&val);
    }

    static Instance create(double val) {
        return create(Float64::Make(), (instance_ptr)&val);
    }

    static Instance create(Type*t, instance_ptr data) {
        t->assertForwardsResolved();

        return createAndInitialize(t, [&](instance_ptr tgt) {
            t->copy_constructor(tgt, data);
        });
    }

    template<class initializer_type>
    static Instance createAndInitialize(Type* t, const initializer_type& initFun) {
        return Instance(t, initFun);
    }

    Instance() {
        // by default, None
        mLayout = noneLayout();
        mLayout->refcount++;
    }

    Instance(const Instance& other) : mLayout(other.mLayout) {
        mLayout->refcount++;
    }

    Instance(instance_ptr p, Type* t) : mLayout(nullptr) {
        t->assertForwardsResolved();

        layout* l = (layout*)malloc(sizeof(layout) + t->bytecount());

        try {
            t->copy_constructor(l->data, p);
        } catch(...) {
            free(l);
            throw;
        }

        l->refcount = 1;
        l->type = t;

        mLayout = l;
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

    ~Instance() {
        mLayout->refcount--;
        if (mLayout->refcount == 0) {
            mLayout->type->destroy(mLayout->data);
            free(mLayout);
        }
    }

    Instance& operator=(const Instance& other) {
        other.mLayout->refcount++;

        mLayout->refcount--;
        if (mLayout->refcount == 0) {
            mLayout->type->destroy(mLayout->data);
            free(mLayout);
        }

        mLayout = other.mLayout;
        return *this;
    }

    bool operator<(const Instance& other) const {
        if (mLayout->type < other.mLayout->type) {
            return true;
        }
        if (mLayout->type > other.mLayout->type) {
            return false;
        }
        return mLayout->type->cmp(mLayout->data, other.mLayout->data) < 0;
    }

    std::string repr() const {
        std::ostringstream s;
        ReprAccumulator accumulator(s);

        accumulator << std::showpoint;
        mLayout->type->repr(mLayout->data, accumulator);
        return s.str();
    }

    int32_t hash32() const {
        return mLayout->type->hash32(mLayout->data);
    }

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
