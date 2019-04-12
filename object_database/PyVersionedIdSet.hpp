/******************************************************************************
   Copyright 2017-2019 Nativepython Authors

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
#include "VersionedIdSet.hpp"
#include <memory>

extern PyTypeObject PyType_VersionedIdSet;

class PyVersionedIdSet {
public:
    PyObject_HEAD;
    std::shared_ptr<VersionedIdSet> idSet;

    static void dealloc(PyVersionedIdSet *self);

    static PyObject *new_(PyTypeObject *type, PyObject *args, PyObject *kwds);

    static PyObject* isActive(PyVersionedIdSet* self, PyObject* args, PyObject* kwargs);

    static PyObject* lookupOne(PyVersionedIdSet* self, PyObject* args, PyObject* kwargs);

    static PyObject* lookupFirst(PyVersionedIdSet* self, PyObject* args, PyObject* kwargs);

    static PyObject* lookupNext(PyVersionedIdSet* self, PyObject* args, PyObject* kwargs);

    static PyObject* add(PyVersionedIdSet* self, PyObject* args, PyObject* kwargs);

    static PyObject* addTransaction(PyVersionedIdSet* self, PyObject* args, PyObject* kwargs);

    static PyObject* remove(PyVersionedIdSet* self, PyObject* args, PyObject* kwargs);

    static PyObject* transactionCount(PyVersionedIdSet* self, PyObject* args, PyObject* kwargs);

    static PyObject* totalEntryCount(PyVersionedIdSet* self, PyObject* args, PyObject* kwargs);

    static PyObject* wantsGuaranteedLowestIdMoveForward(PyVersionedIdSet* self, PyObject* args, PyObject* kwargs);

    static PyObject* moveGuaranteedLowestIdForward(PyVersionedIdSet* self, PyObject* args, PyObject* kwargs);

    static int init(PyVersionedIdSet *self, PyObject *args, PyObject *kwds);
};

