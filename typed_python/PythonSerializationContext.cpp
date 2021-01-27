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

std::shared_ptr<ByteBuffer> PythonSerializationContext::compress(uint8_t* begin, uint8_t* end) const {
    return compressOrDecompress(begin, end, true);
}

std::shared_ptr<ByteBuffer> PythonSerializationContext::decompress(uint8_t* begin, uint8_t* end) const {
    return compressOrDecompress(begin, end, false);
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

std::shared_ptr<ByteBuffer> PythonSerializationContext::compressOrDecompress(uint8_t* begin, uint8_t* end, bool compress) const {
    assertHoldingTheGil();

    if (!mContextObj) {
        return std::shared_ptr<ByteBuffer>(new RangeByteBuffer(begin,end));
    }

    PyObjectStealer pyBytes(
        PyBytes_FromStringAndSize((const char*)begin, end-begin)
        );

    PyObjectStealer outBytes(
        PyObject_CallMethod(
            mContextObj,
            compress ? "compress" : "decompress",
            "(O)",
            (PyObject*)pyBytes //without this cast, the actual "Stealer" object gets passed
                               //because this is a C varargs function and it doesn't know
                               //that the intended type is PyObject*.
            )
        );

    if (!outBytes) {
        throw PythonExceptionSet();
    }

    if (!PyBytes_Check(outBytes)) {
        PyErr_Format(PyExc_TypeError,
            compress ?
                    "'compress' method didn't return bytes object."
                :   "'decompress' method didn't return bytes object."
            );
        throw PythonExceptionSet();
    }

    return std::shared_ptr<ByteBuffer>(new PyBytesByteBuffer(outBytes));
}
