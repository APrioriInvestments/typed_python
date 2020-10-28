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

#include "MutuallyRecursiveTypeGroup.hpp"
#include "TypeOrPyobj.hpp"

// return type(), or check if pyobj is a Type and if so unwrap it.
Type* TypeOrPyobj::typeOrPyobjAsType() const {
    if (mType) {
        return mType;
    }

    if (!mPyObj || !PyType_Check(mPyObj)) {
        return nullptr;
    }

    return PyInstance::extractTypeFrom((PyTypeObject*)mPyObj);
}

  // return pyobj(), or convert the Type to its pyobj and return that.
PyObject* TypeOrPyobj::typeOrPyobjAsObject() const {
    if (mType) {
        return (PyObject*)PyInstance::typeObj(mType);
    }

    return mPyObj;
}


ShaHash TypeOrPyobj::identityHash() {
    if (mType) {
        return mType->identityHash();
    }

    return MutuallyRecursiveTypeGroup::pyObjectShaHash(mPyObj, nullptr);
}

std::string TypeOrPyobj::name() {
    if (mType) {
        return "<Type " + mType->name() + " of cat " + Type::categoryToString(mType->getTypeCategory()) + ">";
    }

    if (mPyObj) {
        std::string lexical = MutuallyRecursiveTypeGroup::pyObjectSortName(mPyObj);
        if (lexical != "<UNNAMED>") {
            return "<PyObj named " + lexical + ">";
        }

        PyObjectStealer repr(PyObject_Repr(mPyObj));

        if (repr) {
            return "<PyObj of type " + std::string(mPyObj->ob_type->tp_name) +
                " with repr " + std::string(PyUnicode_AsUTF8(repr)).substr(0, 150) + ">";
            decref(repr);
        }

        return "<PyObj of type " + std::string(mPyObj->ob_type->tp_name) + ">";
    }

    throw std::runtime_error("Invalid TypeOrPyobj encountered.");
}
