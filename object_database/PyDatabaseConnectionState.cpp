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

#include "PyDatabaseConnectionState.hpp"
#include "ObjectFieldId.hpp"
#include "IndexId.hpp"
#include "../typed_python/direct_types/all.hpp"
#include "../typed_python/SerializationContext.hpp"
#include "../typed_python/PythonSerializationContext.hpp"

PyMethodDef PyDatabaseConnectionState_methods[] = {
    {"objectCount", (PyCFunction)PyDatabaseConnectionState::objectCount, METH_VARARGS | METH_KEYWORDS, NULL},
    {"outstandingViewCount", (PyCFunction)PyDatabaseConnectionState::outstandingViewCount, METH_VARARGS | METH_KEYWORDS, NULL},
    {"setIdentityRoot", (PyCFunction)PyDatabaseConnectionState::setIdentityRoot, METH_VARARGS | METH_KEYWORDS, NULL},
    {"allocateIdentity", (PyCFunction)PyDatabaseConnectionState::allocateIdentity, METH_VARARGS | METH_KEYWORDS, NULL},
    {"setSerializationContext", (PyCFunction)PyDatabaseConnectionState::setSerializationContext, METH_VARARGS | METH_KEYWORDS, NULL},
    {"setFieldId", (PyCFunction)PyDatabaseConnectionState::setFieldId, METH_VARARGS | METH_KEYWORDS, NULL},
    {"getMinTid", (PyCFunction)PyDatabaseConnectionState::getMinTid, METH_VARARGS | METH_KEYWORDS, NULL},
    {"incomingTransaction", (PyCFunction)PyDatabaseConnectionState::incomingTransaction, METH_VARARGS | METH_KEYWORDS, NULL},
    {"markTypeSubscribed", (PyCFunction)PyDatabaseConnectionState::markTypeSubscribed, METH_VARARGS | METH_KEYWORDS, NULL},
    {"markObjectSubscribed", (PyCFunction)PyDatabaseConnectionState::markObjectSubscribed, METH_VARARGS | METH_KEYWORDS, NULL},
    {"markObjectLazy", (PyCFunction)PyDatabaseConnectionState::markObjectLazy, METH_VARARGS | METH_KEYWORDS, NULL},
    {"markObjectNotLazy", (PyCFunction)PyDatabaseConnectionState::markObjectNotLazy, METH_VARARGS | METH_KEYWORDS, NULL},
    {"typeSubscriptionLowestTransaction", (PyCFunction)PyDatabaseConnectionState::typeSubscriptionLowestTransaction, METH_VARARGS | METH_KEYWORDS, NULL},
    {"objectSubscriptionLowestTransaction", (PyCFunction)PyDatabaseConnectionState::objectSubscriptionLowestTransaction, METH_VARARGS | METH_KEYWORDS, NULL},
    {"setTriggerLazyLoad", (PyCFunction)PyDatabaseConnectionState::setTriggerLazyLoad, METH_VARARGS | METH_KEYWORDS, NULL},
    {NULL}  /* Sentinel */
};


/* static */
void PyDatabaseConnectionState::dealloc(PyDatabaseConnectionState *self)
{
    self->state.~shared_ptr();

    Py_TYPE(self)->tp_free((PyObject*)self);
}

/* static */
PyObject* PyDatabaseConnectionState::new_(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    PyDatabaseConnectionState* self;

    self = (PyDatabaseConnectionState*) type->tp_alloc(type, 0);

    if (self != NULL) {
        new (&self->state) std::shared_ptr<DatabaseConnectionState>();
    }
    return (PyObject*)self;
}

/* static */
int PyDatabaseConnectionState::init(PyDatabaseConnectionState *self, PyObject *args, PyObject *kwargs)
{
    static const char *kwlist[] = {NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "", (char**)kwlist)) {
        return -1;
    }

    self->state.reset(
        new DatabaseConnectionState()
    );

    return 0;
}

/* static */
PyObject* PyDatabaseConnectionState::setSerializationContext(PyDatabaseConnectionState* self, PyObject* args, PyObject* kwargs)
{
    static const char *kwlist[] = {"serializationContext", NULL};

    PyObject* context;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", (char**)kwlist, &context)) {
        return NULL;
    }

    return translateExceptionToPyObject([&]() {
        if (!self->state) {
            throw std::runtime_error("Invalid PyDatabaseConnectionState (nullptr)");
        }

        self->state->setContext(
            std::shared_ptr<SerializationContext>(
                new PythonSerializationContext(context)
            )
        );

        return incref(Py_None);
    });
}

/* static */
PyObject* PyDatabaseConnectionState::setFieldId(PyDatabaseConnectionState* self, PyObject* args, PyObject* kwargs)
{
    static const char *kwlist[] = {"schemaname", "typename", "fieldname", "field_id", NULL};

    const char* schemaname;
    const char* type_name;
    const char* fieldname;
    field_id fieldId;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sssl", (char**)kwlist, &schemaname, &type_name, &fieldname, &fieldId)) {
        return NULL;
    }

    return translateExceptionToPyObject([&]() {
        self->state->setFieldId(SchemaAndTypeName(schemaname, type_name), fieldname, fieldId);

        return incref(Py_None);
    });
}

/* static */
PyObject* PyDatabaseConnectionState::getMinTid(PyDatabaseConnectionState* self, PyObject* args, PyObject* kwargs)
{
    static const char *kwlist[] = {NULL};


    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "", (char**)kwlist)) {
        return NULL;
    }

    return translateExceptionToPyObject([&]() {
        return PyLong_FromLong(self->state->getMinId());
    });
}

/* static */
PyObject* PyDatabaseConnectionState::setIdentityRoot(PyDatabaseConnectionState* self, PyObject* args, PyObject* kwargs)
{
    static const char *kwlist[] = {"id", NULL};

    transaction_id id;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "l", (char**)kwlist, &id)) {
        return NULL;
    }

    return translateExceptionToPyObject([&]() {
        self->state->setIdentityRoot(id);

        return incref(Py_None);
    });
}

/* static */
PyObject* PyDatabaseConnectionState::outstandingViewCount(PyDatabaseConnectionState* self, PyObject* args, PyObject* kwargs)
{
    static const char *kwlist[] = {NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "", (char**)kwlist)) {
        return NULL;
    }

    return translateExceptionToPyObject([&]() {
        return PyLong_FromLong(self->state->outstandingViewCount());
    });
}

/* static */
PyObject* PyDatabaseConnectionState::objectCount(PyDatabaseConnectionState* self, PyObject* args, PyObject* kwargs)
{
    static const char *kwlist[] = {NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "", (char**)kwlist)) {
        return NULL;
    }

    return translateExceptionToPyObject([&]() {
        return PyLong_FromLong(self->state->getVersionedObjects()->objectCount());
    });
}

/* static */
PyObject* PyDatabaseConnectionState::allocateIdentity(PyDatabaseConnectionState* self, PyObject* args, PyObject* kwargs)
{
    static const char *kwlist[] = {NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "", (char**)kwlist)) {
        return NULL;
    }

    return translateExceptionToPyObject([&]() {
        return PyLong_FromLong(self->state->allocateIdentity());
    });
}

/* static */
PyObject* PyDatabaseConnectionState::incomingTransaction(PyDatabaseConnectionState* self, PyObject* args, PyObject* kwargs)
{
    static const char *kwlist[] = {"transaction_id", "writes", "set_adds", "set_removes", NULL};

    transaction_id tid;
    PyObject* writes;
    PyObject* set_adds;
    PyObject* set_removes;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "lOOO", (char**)kwlist, &tid, &writes, &set_adds, &set_removes)) {
        return NULL;
    }

    return translateExceptionToPyObject([&]() {
        if (!self->state) {
            throw std::runtime_error("Invalid PyDatabaseConnectionState (nullptr)");
        }

        auto cd_writes = ConstDict<ObjectFieldId, OneOf<None, Bytes> >::fromPython(writes);
        auto cd_set_adds = ConstDict<IndexId, TupleOf<object_id> >::fromPython(set_adds);
        auto cd_set_removes = ConstDict<IndexId, TupleOf<object_id> >::fromPython(set_removes);

        self->state->incomingTransaction(tid, cd_writes, cd_set_adds, cd_set_removes);

        return incref(Py_None);
    });
}

/* static */
PyObject* PyDatabaseConnectionState::markTypeSubscribed(PyDatabaseConnectionState* self, PyObject* args, PyObject* kwargs) {
    static const char *kwlist[] = {"schema", "typename", "transaction_id", NULL};

    const char* schemaName;
    const char* typeName;
    transaction_id tid;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ssl", (char**)kwlist, &schemaName, &typeName, &tid)) {
        return NULL;
    }

    return translateExceptionToPyObject([&]() {
        if (!self->state) {
            throw std::runtime_error("Invalid PyDatabaseConnectionState (nullptr)");
        }

        self->state->markTypeSubscribed(SchemaAndTypeName(schemaName, typeName), tid);

        return incref(Py_None);
    });
}

/* static */
PyObject* PyDatabaseConnectionState::markObjectSubscribed(PyDatabaseConnectionState* self, PyObject* args, PyObject* kwargs) {
    static const char *kwlist[] = {"object_id", "transaction_id", NULL};

    object_id oid;
    transaction_id tid;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ll", (char**)kwlist, &oid, &tid)) {
        return NULL;
    }

    return translateExceptionToPyObject([&]() {
        if (!self->state) {
            throw std::runtime_error("Invalid PyDatabaseConnectionState (nullptr)");
        }

        self->state->markObjectSubscribed(oid, tid);

        return incref(Py_None);
    });
}

/* static */
PyObject* PyDatabaseConnectionState::typeSubscriptionLowestTransaction(PyDatabaseConnectionState* self, PyObject* args, PyObject* kwargs) {
    static const char *kwlist[] = {"schema", "typename", NULL};

    const char* schemaName;
    const char* typeName;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ss", (char**)kwlist, &schemaName, &typeName)) {
        return NULL;
    }

    return translateExceptionToPyObject([&]() {
        if (!self->state) {
            throw std::runtime_error("Invalid PyDatabaseConnectionState (nullptr)");
        }

        transaction_id tid = self->state->typeSubscriptionLowestTransaction(SchemaAndTypeName(schemaName, typeName));

        if (tid == NO_TRANSACTION) {
            return incref(Py_None);
        }

        return PyLong_FromLong(tid);
    });
}

/* static */
PyObject* PyDatabaseConnectionState::markObjectLazy(PyDatabaseConnectionState* self, PyObject* args, PyObject* kwargs) {
    static const char *kwlist[] = {"schema", "typename", "object_id", NULL};

    const char* schemaName;
    const char* typeName;
    object_id oid;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ssl", (char**)kwlist, &schemaName, &typeName, &oid)) {
        return NULL;
    }

    return translateExceptionToPyObject([&]() {
        if (!self->state) {
            throw std::runtime_error("Invalid PyDatabaseConnectionState (nullptr)");
        }

        self->state->markObjectLazy(SchemaAndTypeName(schemaName, typeName), oid);

        return incref(Py_None);
    });
}

/* static */
PyObject* PyDatabaseConnectionState::markObjectNotLazy(PyDatabaseConnectionState* self, PyObject* args, PyObject* kwargs) {
    static const char *kwlist[] = {"object_id", NULL};

    object_id oid;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "l", (char**)kwlist, &oid)) {
        return NULL;
    }

    return translateExceptionToPyObject([&]() {
        if (!self->state) {
            throw std::runtime_error("Invalid PyDatabaseConnectionState (nullptr)");
        }

        self->state->markObjectNotLazy(oid);

        return incref(Py_None);
    });
}

/* static */
PyObject* PyDatabaseConnectionState::objectSubscriptionLowestTransaction(PyDatabaseConnectionState* self, PyObject* args, PyObject* kwargs) {
    static const char *kwlist[] = {"object_id", NULL};

    object_id oid;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "l", (char**)kwlist, &oid)) {
        return NULL;
    }

    return translateExceptionToPyObject([&]() {
        if (!self->state) {
            throw std::runtime_error("Invalid PyDatabaseConnectionState (nullptr)");
        }

        transaction_id tid = self->state->objectSubscriptionLowestTransaction(oid);

        if (tid == NO_TRANSACTION) {
            return incref(Py_None);
        }

        return PyLong_FromLong(tid);
    });
}

/* static */
PyObject* PyDatabaseConnectionState::setTriggerLazyLoad(PyDatabaseConnectionState* self, PyObject* args, PyObject* kwargs) {
    static const char *kwlist[] = {"callback", NULL};

    PyObject* callback;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", (char**)kwlist, &callback)) {
        return NULL;
    }

    return translateExceptionToPyObject([&]() {
        if (!self->state) {
            throw std::runtime_error("Invalid PyDatabaseConnectionState (nullptr)");
        }

        self->state->setTriggerLazyLoad(callback);

        return incref(Py_None);
    });
}

PyTypeObject PyType_DatabaseConnectionState = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "DatabaseConnectionState",
    .tp_basicsize = sizeof(PyDatabaseConnectionState),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor) PyDatabaseConnectionState::dealloc,
    .tp_print = 0,
    .tp_getattr = 0,
    .tp_setattr = 0,
    .tp_as_async = 0,
    .tp_repr = 0,
    .tp_as_number = 0,
    .tp_as_sequence = 0,
    .tp_as_mapping = 0,
    .tp_hash = 0,
    .tp_call = 0,
    .tp_str = 0,
    .tp_getattro = 0,
    .tp_setattro = 0,
    .tp_as_buffer = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = 0,
    .tp_traverse = 0,
    .tp_clear = 0,
    .tp_richcompare = 0,
    .tp_weaklistoffset = 0,
    .tp_iter = 0,
    .tp_iternext = 0,
    .tp_methods = PyDatabaseConnectionState_methods,
    .tp_members = 0,
    .tp_getset = 0,
    .tp_base = 0,
    .tp_dict = 0,
    .tp_descr_get = 0,
    .tp_descr_set = 0,
    .tp_dictoffset = 0,
    .tp_init = (initproc) PyDatabaseConnectionState::init,
    .tp_alloc = 0,
    .tp_new = PyDatabaseConnectionState::new_,
    .tp_free = 0,
    .tp_is_gc = 0,
    .tp_bases = 0,
    .tp_mro = 0,
    .tp_cache = 0,
    .tp_subclasses = 0,
    .tp_weaklist = 0,
    .tp_del = 0,
    .tp_version_tag = 0,
    .tp_finalize = 0,
};

