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

#include "PythonSerializationContext.hpp"
#include "AllTypes.hpp"
#include "PyInstance.hpp"
#include "MutuallyRecursiveTypeGroup.hpp"

void PythonSerializationContext::setFlags() {
    PyEnsureGilAcquired acquireTheGil;

    PyObjectStealer compressionEnabled(PyObject_GetAttrString(mContextObj, "compressionEnabled"));

    if (!compressionEnabled) {
        throw PythonExceptionSet();
    }

    mCompressionEnabled = ((PyObject*)compressionEnabled) == Py_True;


    PyObjectStealer internalizeTypeGroups(PyObject_GetAttrString(mContextObj, "internalizeTypeGroups"));

    if (!internalizeTypeGroups) {
        throw PythonExceptionSet();
    }

    mInternalizeTypeGroups = ((PyObject*)internalizeTypeGroups) == Py_True;

    PyObjectStealer serializeHashSequence(PyObject_GetAttrString(mContextObj, "serializeHashSequence"));

    if (!serializeHashSequence) {
        throw PythonExceptionSet();
    }

    mSerializeHashSequence = ((PyObject*)serializeHashSequence) == Py_True;
}

std::string PythonSerializationContext::getNameForPyObj(PyObject* o) const {
    PyEnsureGilAcquired acquireTheGil;

    PyObjectStealer nameForObject(PyObject_CallMethod(mContextObj, "nameForObject", "(O)", o));

    if (!nameForObject) {
        throw PythonExceptionSet();
    }

    if (nameForObject != Py_None) {
        if (!PyUnicode_Check(nameForObject)) {
            decref(nameForObject);
            throw std::runtime_error("nameForObject returned something other than None or a string.");
        }

        return std::string(PyUnicode_AsUTF8(nameForObject));
    }

    return "";
}
