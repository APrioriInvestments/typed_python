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

#include "util.hpp"
#include "Type.hpp"

class PyObjectHandleTypeBase : public Type {
public:
    PyObjectHandleTypeBase(Type::TypeCategory cat) : Type(cat)
    {
    }

    class layout_type {
    public:
        std::atomic<int64_t> refcount;
        PyObject* pyObj;
    };

    typedef layout_type* layout_ptr;

    // return a new layout with a refcount of 1, but steal the reference
    // to the python object.
    static layout_type* stealToCreateLayout(PyObject* p) {
        return createLayout(p, false);
    }

    // return a new layout with a refcount of 1, increffing the argument
    // before placing it in the layout.
    static layout_type* createLayout(PyObject* p, bool alsoIncref = true) {
        layout_type* res = (layout_type*)tp_malloc(sizeof(layout_type));

        if (alsoIncref) {
            incref(p);
        }

        res->pyObj = p;
        res->refcount = 1;

        return res;
    }

    void initializeHandleAt(instance_ptr ptr) {
        ((layout_type**)ptr)[0] = (layout_type*)tp_malloc(sizeof(layout_type));
        ((layout_type**)ptr)[0]->pyObj = NULL;
        ((layout_type**)ptr)[0]->refcount = 1;
    }

    void initializeFromPyObject(instance_ptr ptr, PyObject* o) {
        initializeHandleAt(ptr);
        getPyObj(ptr) = incref(o);
    }

    static PyObject*& getPyObj(instance_ptr ptr) {
        return ((layout_type**)ptr)[0]->pyObj;
    }

    static layout_type*& getHandlePtr(instance_ptr ptr) {
        return *(layout_type**)ptr;
    }

    static void destroyLayoutIfRefcountIsZero(layout_type* p) {
        if (p->refcount == 0) {
            PyEnsureGilAcquired getTheGil;
            decref(p->pyObj);
            tp_free(p);
        }
    }

    void destroy(instance_ptr self) {
        getHandlePtr(self)->refcount--;

        if (getHandlePtr(self)->refcount == 0) {
            PyEnsureGilAcquired getTheGil;
            decref(getPyObj(self));
            tp_free(*(layout_type**)self);
        }
    }

    void copy_constructor(instance_ptr self, instance_ptr other) {
        getHandlePtr(self) = getHandlePtr(other);
        getHandlePtr(self)->refcount++;
    }

    void assign(instance_ptr self, instance_ptr other) {
        if (getHandlePtr(self) == getHandlePtr(other)) {
            return;
        }
        getHandlePtr(other)->refcount++;
        destroy(self);
        getHandlePtr(self) = getHandlePtr(other);
    }
};

//wraps an actual python instance. Note that we assume we're holding the GIL whenever
//we interact with actual python objects. Compiled code needs to treat these objects
//with extreme care. We hold the pointer in our own refcounted datastructure so that
//compiled code can move python objects around without hitting the GIL (we only
//refcount when we completely release a handle to a python object).
class PythonObjectOfType : public PyObjectHandleTypeBase {
public:
    PythonObjectOfType(PyTypeObject* typePtr, PyObject* givenType) :
            PyObjectHandleTypeBase(TypeCategory::catPythonObjectOfType)
    {
        mPyTypePtr = (PyTypeObject*)incref((PyObject*)typePtr);

        if (givenType) {
            mGivenType = incref(givenType);
        } else {
            mGivenType = nullptr;
        }

        m_name = std::string("PythonObjectOfType(") + mPyTypePtr->tp_name + ")";

        m_is_simple = false;

        endOfConstructorInitialization(); // finish initializing the type object.
    }

    template<class visitor_type>
    void _visitCompilerVisiblePythonObjects(const visitor_type& visitor) {
        visitor((PyObject*)mPyTypePtr);
    }

    ShaHash _computeIdentityHash(MutuallyRecursiveTypeGroup* groupHead = nullptr) {
        ShaHash res(1, m_typeCategory);

        res += MutuallyRecursiveTypeGroup::pyObjectShaHash((PyObject*)mPyTypePtr, groupHead);

        return res;
    }

    bool isBinaryCompatibleWithConcrete(Type* other) {
        return other == this;
    }

    template<class visitor_type>
    void _visitContainedTypes(const visitor_type& visitor) {
    }

    template<class visitor_type>
    void _visitReferencedTypes(const visitor_type& visitor) {
    }

    bool _updateAfterForwardTypesChanged() {
        m_size = sizeof(layout_type*);

        int isinst = PyObject_IsInstance(Py_None, (PyObject*)mPyTypePtr);
        if (isinst == -1) {
            isinst = 0;
            PyErr_Clear();
        }

        m_is_default_constructible = isinst != 0;

        //none of these values can ever change, so we can just return
        //because we don't need to be updated again.
        return false;
    }

    int64_t refcount(instance_ptr self) const {
        return getHandlePtr(self)->refcount;
    }

    template<class buf_t>
    void serialize(instance_ptr self, buf_t& buffer, size_t fieldNumber) {
        PyObject* p = getPyObj(self);
        buffer.getContext().serializePythonObject(p, buffer, fieldNumber);
    }

    static size_t deepBytecountForPyObj(PyObject* o, std::unordered_set<void*>& alreadyVisited, std::set<Slab*>* outSlabs);

    static PyObject* deepcopyPyObject(
        PyObject* o,
        DeepcopyContext& context
    );

    void deepcopyConcrete(
        instance_ptr dest,
        instance_ptr src,
        DeepcopyContext& context
    );

    size_t deepBytecountConcrete(instance_ptr instance, std::unordered_set<void*>& alreadyVisited, std::set<Slab*>* outSlabs);

    template<class buf_t>
    void deserialize(instance_ptr self, buf_t& buffer, size_t wireType) {
         initializeHandleAt(self);
         getPyObj(self) = buffer.getContext().deserializePythonObject(buffer, wireType);
    }

    typed_python_hash_type hash(instance_ptr left) {
        return PyObject_Hash(getPyObj(left));
    }

    void repr(instance_ptr self, ReprAccumulator& stream, bool isStr);

    bool cmp(instance_ptr left, instance_ptr right, int pyComparisonOp, bool suppressExceptions);

    void constructor(instance_ptr self) {
        initializeHandleAt(self);

        PyEnsureGilAcquired getTheGil;
        getPyObj(self) = incref(Py_None);
    }

    // construct a new Type. If 'givenType' is not NULL, then it's the type the user
    // gave us (and we inferred a real type from it based on the lookup table in internals.py)
    static PythonObjectOfType* Make(PyTypeObject* pyType, PyObject* givenType=NULL);

    static PythonObjectOfType* AnyPyObject();

    static PythonObjectOfType* AnyPyType();

    PyTypeObject* pyType() const {
        return mPyTypePtr;
    }

private:
    PyTypeObject* mPyTypePtr;

    // this is the object we were given, which in some cases might not really
    // be a type, but which users expect to refer to a type (for instance, a threading.RLock,
    // or anything else in internals._nonTypesAcceptedAsTypes);
    PyObject* mGivenType;
};
