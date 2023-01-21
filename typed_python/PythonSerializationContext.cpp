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
    Class* serContext = (Class*)mContextObj.type();

    auto getBool = [&](const char* name) {
        int i = serContext->getMemberIndex(name);

        if (i < 0) {
            throw std::runtime_error(
                "Somehow this SerializationContext has no member " + std::string(name)
            );
        }

        if (!serContext->checkInitializationFlag(mContextObj.data(), i)) {
            throw std::runtime_error(
                "Somehow this SerializationContext member " + std::string(name) + " is empty"
            );
        }

        if (!serContext->getMemberType(i)->isBool()) {
            throw std::runtime_error(
                "Somehow this SerializationContext member " + std::string(name) + " is not a bool"
            );
        }

        return *(bool*)serContext->eltPtr(mContextObj.data(), i);
    };

    mCompressionEnabled = getBool("compressionEnabled");
    mSerializeHashSequence = getBool("serializeHashSequence");
    mSerializePodListsInline = getBool("serializePodListsInline");
    mCompressUsingThreads = getBool("compressUsingThreads");
    mSuppressLineInfo = !getBool("encodeLineInformationForCode");
}

std::string PythonSerializationContext::getNameForPyObj(PyObject* o) const {
    PyEnsureGilAcquired acquireTheGil;

    PyObjectStealer contextAsPyObj(PyInstance::extractPythonObject(mContextObj, false));

    PyObjectStealer nameForObject(PyObject_CallMethod(contextAsPyObj, "nameForObject", "(O)", o));

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
