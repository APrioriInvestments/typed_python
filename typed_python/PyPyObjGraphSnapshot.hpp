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
#include "PyObjGraphSnapshot.hpp"

class PyPyObjGraphSnapshot {
public:
    PyObject_HEAD

    PyObjGraphSnapshot* mGraphSnapshot;
    bool mOwnsSnapshot;

    static PyObject* tp_repr(PyObject *selfObj);

    static void dealloc(PyPyObjGraphSnapshot *self);

    static PyObject* hashToObject(PyObject* graph, PyObject *args, PyObject *kwargs);

    static PyObject* internalize(PyObject* graph, PyObject *args, PyObject *kwargs);

    static PyObject* resolveForwards(PyObject* graph, PyObject *args, PyObject *kwargs);

    static PyObject* extractTypes(PyObject* graph, PyObject *args, PyObject *kwargs);

    static PyObject* getObjects(PyObject* graph, PyObject *args, PyObject *kwargs);

    static PyObject *new_(PyTypeObject *type, PyObject *args, PyObject *kwargs);

    static PyObject* newPyObjGraphSnapshot(PyObjGraphSnapshot* o, bool ownsIt);

    static int init(PyPyObjGraphSnapshot *self, PyObject *args, PyObject *kwargs);
};

extern PyTypeObject PyType_PyObjGraphSnapshot;
