/******************************************************************************
   Copyright 2017-2023 typed_python Authors

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

class PyFunctionGlobal {
public:
    PyObject_HEAD

    Function* mFuncType;

    int64_t mOverloadIx;

    std::string* mName;

    FunctionGlobal& getGlobal();

    static PyObject* tp_getattro(PyObject* selfObj, PyObject* attr);

    static PyObject* tp_repr(PyObject *selfObj);

    static void dealloc(PyFunctionGlobal *self);

    static PyObject *new_(PyTypeObject *type, PyObject *args, PyObject *kwargs);

    static PyObject* newPyFunctionGlobal(Function* func, long overloadIx, std::string globalName);

    static int init(PyFunctionGlobal *self, PyObject *args, PyObject *kwargs);

    static PyObject* isUnresolved(PyFunctionGlobal* self, PyObject* args, PyObject* kwargs);

    static PyObject* getValue(PyFunctionGlobal* self, PyObject* args, PyObject* kwargs);
};

extern PyTypeObject PyType_FunctionGlobal;
