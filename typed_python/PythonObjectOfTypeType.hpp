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

#include "util.hpp"
#include "Type.hpp"

//wraps an actual python instance. Note that we assume we're holding the GIL whenever
//we interact with actual python objects. Compiled code needs to treat these objects
//with extreme care...
class PythonObjectOfType : public Type {
public:
    PythonObjectOfType(PyTypeObject* typePtr) :
            Type(TypeCategory::catPythonObjectOfType)
    {
        mPyTypePtr = (PyTypeObject*)incref((PyObject*)typePtr);
        m_name = typePtr->tp_name;
        m_is_simple = false;

        endOfConstructorInitialization(); // finish initializing the type object.
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
        m_size = sizeof(PyObject*);

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

    typed_python_hash_type hash64(instance_ptr left) {
        PyObject* p = *(PyObject**)left;

        return PyObject_Hash(p);
    }

    template<class buf_t>
    void serialize(instance_ptr self, buf_t& buffer, size_t fieldNumber) {
        PyObject* p = *(PyObject**)self;
        buffer.getContext().serializePythonObject(p, buffer, fieldNumber);
    }

    template<class buf_t>
    void deserialize(instance_ptr self, buf_t& buffer, size_t wireType) {
         *(PyObject**)self = buffer.getContext().deserializePythonObject(buffer, wireType);
    }

    void repr(instance_ptr self, ReprAccumulator& stream);

    bool cmp(instance_ptr left, instance_ptr right, int pyComparisonOp);

    void constructor(instance_ptr self) {
        *(PyObject**)self = incref(Py_None);
    }

    void destroy(instance_ptr self) {
        decref(*(PyObject**)self);
    }

    void copy_constructor(instance_ptr self, instance_ptr other) {
        *(PyObject**)self = incref(*(PyObject**)other);
    }

    void assign(instance_ptr self, instance_ptr other) {
        incref(*(PyObject**)other);
        decref(*(PyObject**)self);
        *(PyObject**)self = *(PyObject**)other;
    }

    static PythonObjectOfType* Make(PyTypeObject* pyType);

    static PythonObjectOfType* AnyPyObject();

    PyTypeObject* pyType() const {
        return mPyTypePtr;
    }

private:
    PyTypeObject* mPyTypePtr;
};
