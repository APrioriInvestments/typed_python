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

bool PythonSubclass::isBinaryCompatibleWithConcrete(Type* other) {
    Type* nonPyBase = m_base;
    while (nonPyBase->getTypeCategory() == TypeCategory::catPythonSubclass) {
        nonPyBase = nonPyBase->getBaseType();
    }

    Type* otherNonPyBase = other;
    while (otherNonPyBase->getTypeCategory() == TypeCategory::catPythonSubclass) {
        otherNonPyBase = otherNonPyBase->getBaseType();
    }

    return nonPyBase->isBinaryCompatibleWith(otherNonPyBase);
}

// static
PythonSubclass* PythonSubclass::Make(Type* base, PyTypeObject* pyType) {
    PyEnsureGilAcquired getTheGil;

    static std::map<PyTypeObject*, PythonSubclass*> m;

    auto it = m.find(pyType);

    if (it == m.end()) {
        it = m.insert(
            std::make_pair(pyType, new PythonSubclass(base, pyType))
            ).first;
    }

    if (it->second->getBaseType() != base) {
        throw std::runtime_error(
            "Expected to find the same base type. Got "
                + it->second->getBaseType()->name() + " != " + base->name()
            );
    }

    return it->second;
}


typed_python_hash_type PythonSubclass::hash(instance_ptr left) {
    if (m_hashFun) {
        PyEnsureGilAcquired acquireTheGil;

        PyObjectStealer selfAsObject(PyInstance::initialize(this, [&](instance_ptr selfData) {
            this->copy_constructor(selfData, left);
        }));

        PyObjectStealer result(
            PyObject_CallFunctionObjArgs(
                m_hashFun,
                //the typecast is necessary since this is a varargs function, and so the
                //C-style calling convention doesn't know to transform the PyObjectStealer
                //into a PyObject*
                (PyObject*)selfAsObject,
                NULL
            )
        );

        if (!result) {
            throw PythonExceptionSet();
        }

        if (!PyLong_Check(result)) {
            throw std::runtime_error("Expected " + this->name() + ".__hash__ to return an integer");
        }

        return PyLong_AsLong(result);
    }

    return m_base->hash(left);
}

void PythonSubclass::repr(instance_ptr self, ReprAccumulator& stream, bool isStr) {
    if (!isStr && m_reprFun) {
        PyEnsureGilAcquired acquireTheGil;

        PyObjectStealer selfAsObject(PyInstance::initialize(this, [&](instance_ptr selfData) {
            this->copy_constructor(selfData, self);
        }));

        PyObjectStealer result(
            PyObject_CallFunctionObjArgs(
                m_reprFun,
                //the typecast is necessary since this is a varargs function, and so the
                //C-style calling convention doesn't know to transform the PyObjectStealer
                //into a PyObject*
                (PyObject*)selfAsObject,
                NULL
            )
        );

        if (!result) {
            throw PythonExceptionSet();
        }

        if (!PyUnicode_Check(result)) {
            throw std::runtime_error("Expected " + this->name() + ".__repr__ to return a string");
        }

        stream << PyUnicode_AsUTF8(result);
        return;
    }

    m_base->repr(self, stream, isStr);
}
