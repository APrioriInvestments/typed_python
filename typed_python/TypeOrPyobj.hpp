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

// holds a reference to a Type* or a PyObject*. This interns any PyObjects it ever sees,
// assuming that they're part of the 'type graph', so that we don't need to inc/decref them
// all the time, so be careful. If you pass a PyObject* that wraps a Type*, you'll get the Type*
// back, not the PyObject, so be careful.
class TypeOrPyobj {
public:
    TypeOrPyobj() :
        mType(nullptr),
        mPyObj(nullptr)
    {}

    TypeOrPyobj(Type* t);

    TypeOrPyobj(PyObject* o);

    TypeOrPyobj(PyTypeObject* o);

    ~TypeOrPyobj() {
    }

    // produce a TypeOrPyobj but don't internalize it. This means that
    // the reference could go bad at some point, so don't store this unless
    // the incref is permanent.
    static TypeOrPyobj withoutIntern(PyObject* o);

    TypeOrPyobj(const TypeOrPyobj& other) {
        mType = other.mType;
        mPyObj = other.mPyObj;
    }

    TypeOrPyobj& operator=(const TypeOrPyobj& other) {
        mType = other.mType;
        mPyObj = other.mPyObj;

        return *this;
    }

    // warning - this calls 'repr', which may alter the object
    std::string name() const;

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

    static std::string pyObjectSortName(PyObject* o);

private:
    void unwrapForward();

    Type* mType;
    PyObject* mPyObj;

    static std::unordered_set<PyObject*> sInternedPyObjects;
};

namespace std {
    template <>
    struct hash<TypeOrPyobj> {
        size_t operator()(const TypeOrPyobj& k) const {
            if (k.isType()) {
                return hash<void*>()((void*)k.type());
            }
            return hash<void*>()((void*)k.pyobj());
        }
    };
}
