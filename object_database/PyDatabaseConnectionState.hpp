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
#include <map>
#include <memory>
#include <vector>
#include <string>
#include <iostream>

#include "DatabaseConnectionState.hpp"

class PyDatabaseConnectionState {
public:
    PyObject_HEAD;
    std::shared_ptr<DatabaseConnectionState> state;

    static void dealloc(PyDatabaseConnectionState *self);

    static PyObject *new_(PyTypeObject *type, PyObject *args, PyObject *kwargs);

    static int init(PyDatabaseConnectionState *self, PyObject *args, PyObject *kwargs);

    static PyObject* objectCount(PyDatabaseConnectionState* self, PyObject* args, PyObject* kwargs);

    static PyObject* outstandingViewCount(PyDatabaseConnectionState* self, PyObject* args, PyObject* kwargs);

    static PyObject* setSerializationContext(PyDatabaseConnectionState* self, PyObject* args, PyObject* kwargs);

    static PyObject* setFieldId(PyDatabaseConnectionState* self, PyObject* args, PyObject* kwargs);

    static PyObject* getMinTid(PyDatabaseConnectionState* self, PyObject* args, PyObject* kwargs);

    static PyObject* setIdentityRoot(PyDatabaseConnectionState* self, PyObject* args, PyObject* kwargs);

    static PyObject* allocateIdentity(PyDatabaseConnectionState* self, PyObject* args, PyObject* kwargs);

    static PyObject* incomingTransaction(PyDatabaseConnectionState* self, PyObject* args, PyObject* kwargs);

    static PyObject* markTypeSubscribed(PyDatabaseConnectionState* self, PyObject* args, PyObject* kwargs);

    static PyObject* markObjectSubscribed(PyDatabaseConnectionState* self, PyObject* args, PyObject* kwargs);

    static PyObject* typeSubscriptionLowestTransaction(PyDatabaseConnectionState* self, PyObject* args, PyObject* kwargs);

    static PyObject* objectSubscriptionLowestTransaction(PyDatabaseConnectionState* self, PyObject* args, PyObject* kwargs);

    static PyObject* markObjectLazy(PyDatabaseConnectionState* self, PyObject* args, PyObject* kwargs);

    static PyObject* markObjectNotLazy(PyDatabaseConnectionState* self, PyObject* args, PyObject* kwargs);

    static PyObject* setTriggerLazyLoad(PyDatabaseConnectionState* self, PyObject* args, PyObject* kwargs);
};

extern PyTypeObject PyType_DatabaseConnectionState;
