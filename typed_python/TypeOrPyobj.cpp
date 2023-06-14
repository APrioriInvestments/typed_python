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

TypeOrPyobj::TypeOrPyobj(Type* t) :
    mType(t),
    mPyObj(nullptr)
{
    if (!mType) {
        throw std::runtime_error("Can't construct a TypeOrPyobj with a null Type");
    }
}

TypeOrPyobj::TypeOrPyobj(PyObject* o) :
    mType(nullptr),
    mPyObj(o)
{
    PyEnsureGilAcquired getTheGil;

    if (PyType_Check(mPyObj)) {
        mType = PyInstance::extractTypeFrom((PyTypeObject*)mPyObj, true);

        if (mType) {
            mPyObj = nullptr;
            return;
        }
    }

    if (!mPyObj) {
        throw std::runtime_error("Can't construct a TypeOrPyobj with a null PyObject");
    }

    if (sInternedPyObjects.find(mPyObj) == sInternedPyObjects.end()) {
        sInternedPyObjects.insert(mPyObj);
        incref(mPyObj);
    }
}

TypeOrPyobj::TypeOrPyobj(PyTypeObject* o) :
    mType(nullptr),
    mPyObj((PyObject*)o)
{
    PyEnsureGilAcquired getTheGil;

    mType = PyInstance::extractTypeFrom(o, true);
    if (mType) {
        mPyObj = nullptr;
        return;
    }

    if (!mPyObj) {
        throw std::runtime_error("Can't construct a TypeOrPyobj with a null PyObject");
    }

    if (sInternedPyObjects.find(mPyObj) == sInternedPyObjects.end()) {
        sInternedPyObjects.insert(mPyObj);
        incref(mPyObj);
    }
}

TypeOrPyobj TypeOrPyobj::withoutIntern(PyObject* o) {
    TypeOrPyobj obj;

    obj.mPyObj = o;

    if (PyType_Check(o)) {
        obj.mType = PyInstance::extractTypeFrom((PyTypeObject*)o, true);
        if (obj.mType) {
            obj.mPyObj = nullptr;
            return obj;
        }
    }

    return obj;
}

std::string TypeOrPyobj::name() const {
    if (mType) {
        std::ostringstream s;
        s << "<" << (mType->isForwardDefined() ? "Forward":"")
            << "Type " << mType->name() + " of cat " + Type::categoryToString(mType->getTypeCategory()) + "@" << (void*)mType << ">";
        return s.str();
    }

    if (mPyObj) {
        std::ostringstream s;

        std::string lexical = pyObjectSortName(mPyObj);

        if (lexical != "<UNNAMED>") {
            s << "<PyObj named " << lexical << "@" << (void*)mPyObj << ">";
            return s.str();
        }

        PyObjectStealer repr(PyObject_Repr(mPyObj));

        s << "<PyObj of type " << mPyObj->ob_type->tp_name;

        if (repr) {
            s << " with repr " << std::string(PyUnicode_AsUTF8(repr)).substr(0, 200);
        }

        s << "@" << (void*)mPyObj << ">";

        return s.str();
    }

    throw std::runtime_error("Invalid TypeOrPyobj encountered.");
}

std::string TypeOrPyobj::pyObjectSortName(PyObject* o) {
    PyEnsureGilAcquired getTheGil;

    if (PyObject_HasAttrString(o, "__module__") && PyObject_HasAttrString(o, "__name__")) {
        std::string modulename, clsname;

        PyObjectStealer moduleName(PyObject_GetAttrString(o, "__module__"));
        if (!moduleName) {
            PyErr_Clear();
        } else {
            if (PyUnicode_Check(moduleName)) {
                modulename = std::string(PyUnicode_AsUTF8(moduleName));
            }
        }

        PyObjectStealer clsName(PyObject_GetAttrString(o, "__name__"));
        if (!clsName) {
            PyErr_Clear();
        } else {
            if (PyUnicode_Check(clsName)) {
                modulename = std::string(PyUnicode_AsUTF8(clsName));
            }
        }

        if (clsname.size()) {
            if (modulename.size()) {
                return modulename + "|" + clsname;
            }

            return clsname;
        }
    }

    return "<UNNAMED>";
}

// static
std::unordered_set<PyObject*> TypeOrPyobj::sInternedPyObjects;
