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

PyDoc_STRVAR(SubclassOf_doc,
    "SubclassOf(T1)\n"
    "\n"
    "Where T1 is a subclass of Type.\n"
    );

class SubclassOfType : public Type {
public:
    SubclassOfType(Type* subclassOf) noexcept :
                    Type(TypeCategory::catSubclassOf),
                    m_subclassOf(subclassOf)
    {
        m_doc = SubclassOf_doc;

        endOfConstructorInitialization(); // finish initializing the type object.
    }

    ShaHash _computeIdentityHash(MutuallyRecursiveTypeGroup* groupHead = nullptr) {
        return ShaHash(1, m_typeCategory) + m_subclassOf->identityHash();
    }

    bool isBinaryCompatibleWithConcrete(Type* other) {
        return false;
    }

    template<class visitor_type>
    void _visitContainedTypes(const visitor_type& visitor) {
        visitor(m_subclassOf);
    }

    template<class visitor_type>
    void _visitReferencedTypes(const visitor_type& visitor) {
        _visitContainedTypes(visitor);
    }

    bool _updateAfterForwardTypesChanged();

    bool isPODConcrete() {
        return true;
    }

    std::string computeName() const;

    void deepcopyConcrete(
        instance_ptr dest,
        instance_ptr src,
        DeepcopyContext& context
    ) {
    }

    size_t deepBytecountConcrete(instance_ptr instance, std::unordered_set<void*>& alreadyVisited, std::set<Slab*>* outSlabs) {
        return 0;
    }

    template<class buf_t>
    void deserialize(instance_ptr self, buf_t& buffer, size_t wireType) {
        Type* actual = buffer.getContext().deserializeNativeType(buffer, wireType);

        ((Type**)self)[0] = actual;
    }

    template<class buf_t>
    void serialize(instance_ptr self, buf_t& buffer, size_t fieldNumber) {
        buffer.getContext().serializeNativeType(*(Type**)self, buffer, fieldNumber);
    }

    void repr(instance_ptr self, ReprAccumulator& stream, bool isStr);

    typed_python_hash_type hash(instance_ptr left);

    bool cmp(instance_ptr left, instance_ptr right, int pyComparisonOp, bool suppressExceptions);

    void constructor(instance_ptr self);

    void destroy(instance_ptr self);

    void copy_constructor(instance_ptr self, instance_ptr other);

    void assign(instance_ptr self, instance_ptr other);

    Type* getSubclassOf() const {
        return m_subclassOf;
    }

    void _updateTypeMemosAfterForwardResolution() {
        SubclassOfType::Make(m_subclassOf, this);
    }

    static SubclassOfType* Make(Type* subclassOf, SubclassOfType* knownType = nullptr);

private:
    Type* m_subclassOf;
};
