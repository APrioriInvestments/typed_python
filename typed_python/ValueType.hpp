#pragma once

#include "Type.hpp"

class Value : public Type {
public:
    bool isBinaryCompatibleWithConcrete(Type* other) {
        return this == other;
    }

    template<class visitor_type>
    void _visitContainedTypes(const visitor_type& visitor) {
    }

    template<class visitor_type>
    void _visitReferencedTypes(const visitor_type& visitor) {
    }

    void _forwardTypesMayHaveChanged() {
    }

    char cmp(instance_ptr left, instance_ptr right) {
        return 0;
    }

    int32_t hash32(instance_ptr left) {
        return mInstance.hash32();
    }

    void repr(instance_ptr self, ReprAccumulator& stream) {
        mInstance.type()->repr(mInstance.data(), stream);
    }

    template<class buf_t>
    void serialize(instance_ptr self, buf_t& buffer) {
    }

    template<class buf_t>
    void deserialize(instance_ptr self, buf_t& buffer) {
    }

    void constructor(instance_ptr self) {}

    void destroy(instance_ptr self) {}

    void copy_constructor(instance_ptr self, instance_ptr other) {}

    void assign(instance_ptr self, instance_ptr other) {}

    const Instance& value() const {
        return mInstance;
    }

    static Type* Make(Instance i) {
        static std::mutex guard;

        std::lock_guard<std::mutex> lock(guard);

        static std::map<Instance, Value*> m;

        auto it = m.find(i);

        if (it == m.end()) {
            it = m.insert(std::make_pair(i, new Value(i))).first;
        }

        return it->second;
    }

    static Type* MakeInt64(int64_t i);
    static Type* MakeFloat64(double i);
    static Type* MakeBool(bool i);
    static Type* MakeBytes(char* data, size_t count);
    static Type* MakeString(size_t bytesPerCodepoint, size_t count, char* data);

private:
    Value(Instance instance) :
            Type(TypeCategory::catValue),
            mInstance(instance)
    {
        m_size = 0;
        m_is_default_constructible = true;
        m_name = mInstance.repr();
    }

    Instance mInstance;
};

