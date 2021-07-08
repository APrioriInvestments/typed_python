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

#include <Python.h>
#include "util.hpp"

class Type;

class TypeOrPyobj {
public:
    TypeOrPyobj() :
        mType(nullptr),
        mPyObj(nullptr)
    {}

    TypeOrPyobj(Type* t) :
        mType(t),
        mPyObj(nullptr)
    {
        if (!mType) {
            throw std::runtime_error("Can't construct a TypeOrPyobj with a null Type");
        }
    }

    TypeOrPyobj(PyObject* o) :
        mType(nullptr),
        mPyObj(o)
    {
        if (!mPyObj) {
            throw std::runtime_error("Can't construct a TypeOrPyobj with a null PyObject");
        }
        incref(mPyObj);
    }

    ~TypeOrPyobj() {
        if (mPyObj) {
            decref(mPyObj);
        }
    }

    TypeOrPyobj(const TypeOrPyobj& other) {
        mType = other.mType;
        mPyObj = other.mPyObj;
        if (mPyObj) {
            incref(mPyObj);
        }
    }

    static TypeOrPyobj steal(PyObject* o) {
        TypeOrPyobj res;
        res.mPyObj = o;
        return res;
    }

    TypeOrPyobj& operator=(const TypeOrPyobj& other) {
        if (other.mPyObj) {
            incref(other.mPyObj);
        }
        if (mPyObj) {
            decref(mPyObj);
        }

        mType = other.mType;
        mPyObj = other.mPyObj;

        return *this;
    }

    ShaHash identityHash();

    // warning - this calls 'repr', which may alter the object
    std::string name();

    bool operator==(const PyObject* o) const {
        return mPyObj == o;
    }

    bool operator==(const Type* t) const {
        return mType == t;
    }

    bool operator==(const TypeOrPyobj& other) const {
        return mType == other.mType && mPyObj == other.mPyObj;
    }

    bool operator<(const TypeOrPyobj& other) const {
        if (mType < other.mType) {
            return true;
        }
        if (mType > other.mType) {
            return false;
        }

        if (mPyObj < other.mPyObj) {
            return true;
        }
        if (mPyObj > other.mPyObj) {
            return false;
        }

        return false;
    }

    bool isType() const {
        return mType != nullptr;
    }

    Type* type() const {
        return mType;
    }

    PyObject* pyobj() const {
        return mPyObj;
    }

    // return type(), or check if pyobj is a Type and if so unwrap it.
    Type* typeOrPyobjAsType() const;

    // return pyobj(), or convert the Type to its pyobj and return that.
    PyObject* typeOrPyobjAsObject() const;

    // make sure that if we're a type object, we have the type in the Type slot
    TypeOrPyobj canonical() const {
        if (mType) {
            return *this;
        }

        if (typeOrPyobjAsType()) {
            return TypeOrPyobj(typeOrPyobjAsType());
        }

        return *this;
    }

private:
    Type* mType;
    PyObject* mPyObj;
};
