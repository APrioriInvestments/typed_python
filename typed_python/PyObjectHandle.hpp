/******************************************************************************
   Copyright 2017-2021 typed_python Authors

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

#include "PyInstance.hpp"
#include "FunctionType.hpp"
#include "_types.hpp"
#include <set>
#include <unordered_map>



class PyObjectHandle {
public:
    PyObjectHandle() : mPyObj(nullptr)
    {
    }

    PyObjectHandle(PyObject* o) : mPyObj(o)
    {
        incref(mPyObj);
    }

    PyObjectHandle(Type* o) : mPyObj((PyObject*)PyInstance::typeObj(o))
    {
        incref(mPyObj);
    }

    ~PyObjectHandle() {
        decref(mPyObj);
    }

    PyObjectHandle(const PyObjectHandle& other) {
        mPyObj = incref(other.mPyObj);
    }

    static PyObjectHandle steal(PyObject* o) {
        PyObjectHandle res;
        res.mPyObj = o;
        return res;
    }

    PyObjectHandle& operator=(const PyObjectHandle& other) {
        incref(other.mPyObj);
        decref(mPyObj);

        mPyObj = other.mPyObj;

        return *this;
    }

    bool operator==(const PyObject* o) const {
        return mPyObj == o;
    }

    bool operator==(Type* t) const {
        return mPyObj == (PyObject*)PyInstance::typeObj(t);
    }

    bool operator==(const PyObjectHandle& other) const {
        return mPyObj == other.mPyObj;
    }

    bool operator!=(const PyObject* o) const {
        return mPyObj != o;
    }

    bool operator!=(Type* t) const {
        return mPyObj != (PyObject*)PyInstance::typeObj(t);
    }

    bool operator!=(const PyObjectHandle& other) const {
        return mPyObj != other.mPyObj;
    }

    operator bool() const {
        return mPyObj;
    }

    bool operator<(const PyObjectHandle& other) const {
        return mPyObj < other.mPyObj;
    }

    PyObject* pyobj() const {
        return mPyObj;
    }

    Type* forceTypeObj() const {
        if (!mPyObj) {
            return nullptr;
        }

        if (PyType_Check(mPyObj)) {
            return PyInstance::unwrapTypeArgToTypePtr(mPyObj);
        }

        return nullptr;
    }

    Type* typeObj() const {
        if (!mPyObj) {
            return nullptr;
        }

        if (PyType_Check(mPyObj)) {
            return PyInstance::extractTypeFrom((PyTypeObject*)mPyObj);
        }

        return nullptr;
    }

private:
    PyObject* mPyObj;
};

namespace std {
    template<>
    struct hash<PyObjectHandle> {
        typedef PyObjectHandle argument_type;
        typedef std::size_t result_type;

        result_type operator()(argument_type const& s) const noexcept {
            return std::hash<PyObject*>()(s.pyobj());
        }
    };
}
