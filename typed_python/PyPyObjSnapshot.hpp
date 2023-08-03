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

class PyPyObjSnapshot {
public:
    PyObject_HEAD

    PyObjSnapshot* mPyobj;

    // a graph object we're keeping alive
    PyObject* mGraph;

    // caches of our 'elements/keys/byKey' members, which we produce on demand
    PyObject* mElements;
    PyObject* mKeys;
    PyObject* mByKey;
    PyObject* mNamedElements;

    static PyObject* tp_repr(PyObject *selfObj);

    static PyObject* create(PyObject* self, PyObject* args, PyObject* kwargs);

    static PyObject* tp_getattro(PyObject* selfObj, PyObject* attr);

    static void dealloc(PyPyObjSnapshot *self);

    static PyObject *new_(PyTypeObject *type, PyObject *args, PyObject *kwargs);

    static PyObject* newPyObjSnapshot(PyObjSnapshot* o, PyObject* pyGraphPtr=nullptr);

    static int init(PyPyObjSnapshot *self, PyObject *args, PyObject *kwargs);
};

extern PyTypeObject PyType_PyObjSnapshot;
