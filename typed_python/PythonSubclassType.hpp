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

#include "Type.hpp"

class PythonSubclass : public Type {
public:
    PythonSubclass(Type* base, PyTypeObject* typePtr) :
            Type(TypeCategory::catPythonSubclass)
    {
        m_base = base;
        mTypeRep = (PyTypeObject*)incref((PyObject*)typePtr);
        m_name = typePtr->tp_name;

        m_reprFun = PyObject_GetAttrString((PyObject*)typePtr, "__repr__");

        if (!m_reprFun) {
            PyErr_Clear();
        }

        // make sure we don't get the __repr__ from our own implementation
        // in the base class. We want to get the subclass' version.
        if (!PyFunction_Check(m_reprFun)) {
            m_reprFun = nullptr;
        }

        m_hashFun = PyObject_GetAttrString((PyObject*)typePtr, "__hash__");

        if (!m_hashFun) {
            PyErr_Clear();
        }

        if (!PyFunction_Check(m_hashFun)) {
            m_hashFun = nullptr;
        }

        endOfConstructorInitialization(); // finish initializing the type object.
    }

    ShaHash _computeIdentityHash(MutuallyRecursiveTypeGroup* groupHead = nullptr) {
        return (
            ShaHash(1, m_typeCategory)
            + m_base->identityHash(groupHead)
            + MutuallyRecursiveTypeGroup::pyObjectShaHash((PyObject*)mTypeRep, groupHead)
        );
    }

    template<class visitor_type>
    void _visitCompilerVisiblePythonObjects(const visitor_type& visitor) {
        visitor((PyObject*)mTypeRep);
    }

    bool isBinaryCompatibleWithConcrete(Type* other);

    template<class visitor_type>
    void _visitContainedTypes(const visitor_type& visitor) {
        visitor(m_base);
    }

    template<class visitor_type>
    void _visitReferencedTypes(const visitor_type& visitor) {
        visitor(m_base);
    }

    bool _updateAfterForwardTypesChanged() {
        size_t size = m_base->bytecount();
        bool is_default_constructible = m_base->is_default_constructible();

        bool anyChanged = (
            size != m_size ||
            m_is_default_constructible != is_default_constructible
        );

        m_size = size;
        m_is_default_constructible = is_default_constructible;

        return anyChanged;
    }

    typed_python_hash_type hash(instance_ptr left);

    template<class buf_t>
    void serialize(instance_ptr self, buf_t& buffer, size_t fieldNumber) {
        m_base->serialize(self, buffer, fieldNumber);
    }

    void deepcopyConcrete(
        instance_ptr dest,
        instance_ptr src,
        DeepcopyContext& context
    ) {
        m_base->deepcopy(dest, src, context);
    }

    size_t deepBytecountConcrete(instance_ptr instance, std::unordered_set<void*>& alreadyVisited, std::set<Slab*>* outSlabs) {
        return m_base->deepBytecount(instance, alreadyVisited, outSlabs);
    }

    template<class buf_t>
    void deserialize(instance_ptr self, buf_t& buffer, size_t wireType) {
        m_base->deserialize(self, buffer, wireType);
    }

    void repr(instance_ptr self, ReprAccumulator& stream, bool isStr);

    bool cmp(instance_ptr left, instance_ptr right, int pyComparisonOp, bool suppressExceptions) {
        return m_base->cmp(left, right, pyComparisonOp, suppressExceptions);
    }

    bool isPODConcrete() {
        return m_base->isPOD();
    }

    void constructor(instance_ptr self) {
        m_base->constructor(self);
    }

    void destroy(instance_ptr self) {
        m_base->destroy(self);
    }

    void copy_constructor(instance_ptr self, instance_ptr other) {
        m_base->copy_constructor(self, other);
    }

    void assign(instance_ptr self, instance_ptr other) {
        m_base->assign(self, other);
    }

    static PythonSubclass* Make(Type* base, PyTypeObject* pyType);

    Type* baseType() const {
        return m_base;
    }

    PyTypeObject* pyType() const {
        return mTypeRep;
    }

private:
    PyObject* m_reprFun;

    PyObject* m_hashFun;
};
