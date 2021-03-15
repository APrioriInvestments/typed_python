/******************************************************************************
   Copyright 2017-2020 typed_python Authors

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

#include "Type.hpp"

PyDoc_STRVAR(Value_doc,
    "Value(x) -> type representing the single immutable value x"
    );

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
        Type* t = mInstance.type();
        visitor(t);
        if (t != mInstance.type()) {
            throw std::runtime_error("visitor shouldn't have changed the type of a Value");
        }
    }

    bool cmp(instance_ptr left, instance_ptr right, int pyComparisonOp, bool suppressExceptions) {
        return cmpResultToBoolForPyOrdering(pyComparisonOp, 0);
    }

    ShaHash _computeIdentityHash(MutuallyRecursiveTypeGroup* groupHead=nullptr) {
        return ShaHash(m_typeCategory) + MutuallyRecursiveTypeGroup::tpInstanceShaHash(mInstance, groupHead);
    }

    template<class visitor_type>
    void _visitCompilerVisiblePythonObjects(const visitor_type& visitor) {

    }

    template<class visitor_type>
    void _visitCompilerVisibleInstances(const visitor_type& visitor) {
        visitor(mInstance);
    }

    typed_python_hash_type hash(instance_ptr left) {
        return mInstance.hash();
    }

    void repr(instance_ptr self, ReprAccumulator& stream, bool isStr) {
        mInstance.type()->repr(mInstance.data(), stream, isStr);
    }

    void deepcopyConcrete(
        instance_ptr dest,
        instance_ptr src,
        DeepcopyContext& context
    ) {
        // do nothing
    }

    size_t deepBytecountConcrete(instance_ptr instance, std::unordered_set<void*>& alreadyVisited, std::set<Slab*>* outSlabs) {
        return 0;
    }

    template<class buf_t>
    void serialize(instance_ptr self, buf_t& buffer, size_t fieldNumber) {
        buffer.writeEmpty(fieldNumber);
    }

    template<class buf_t>
    void deserialize(instance_ptr self, buf_t& buffer, size_t wireType) {
        assertWireTypesEqual(wireType, WireType::EMPTY);
    }

    bool isPODConcrete() {
        return true;
    }

    void constructor(instance_ptr self) {}

    void destroy(instance_ptr self) {}

    void copy_constructor(instance_ptr self, instance_ptr other) {}

    void assign(instance_ptr self, instance_ptr other) {}

    const Instance& value() const {
        return mInstance;
    }

    static Type* Make(Instance i) {
        PyEnsureGilAcquired getTheGil;

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
        m_doc = Value_doc;

        endOfConstructorInitialization(); // finish initializing the type object.
    }

    Instance mInstance;
};
