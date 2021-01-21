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
#include "Slab.hpp"

class PySlab {
public:
    PyObject_HEAD

    Slab* mSlab;

    static void dealloc(PySlab *self);

    static PyObject *new_(PyTypeObject *type, PyObject *args, PyObject *kwargs);

    static PyObject* newPySlab(Slab* s);

    static int init(PySlab *self, PyObject *args, PyObject *kwargs);

    static PyObject* slabPtr(PySlab* self, PyObject* args, PyObject* kwargs);

    static PyObject* refcount(PySlab* self, PyObject* args, PyObject* kwargs);

    static PyObject* bytecount(PySlab* self, PyObject* args, PyObject* kwargs);

    static PyObject* allocCount(PySlab* self, PyObject* args, PyObject* kwargs);

    static PyObject* liveAllocCount(PySlab* self, PyObject* args, PyObject* kwargs);

    static PyObject* extractObject(PySlab* self, PyObject* args, PyObject* kwargs);

    static PyObject* allocIsAlive(PySlab* self, PyObject* args, PyObject* kwargs);

    static PyObject* allocRefcount(PySlab* self, PyObject* args, PyObject* kwargs);

    static PyObject* allocType(PySlab* self, PyObject* args, PyObject* kwargs);

    static PyObject* getTag(PySlab* self, PyObject* args, PyObject* kwargs);

    static PyObject* setTag(PySlab* self, PyObject* args, PyObject* kwargs);
};

extern PyTypeObject PyType_Slab;
