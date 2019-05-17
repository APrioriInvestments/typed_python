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

#include "View.hpp"
#include "PyDatabaseConnectionState.hpp"
#include "ObjectFieldId.hpp"
#include "IndexId.hpp"
#include "../typed_python/SerializationBuffer.hpp"
#include "../typed_python/SerializationContext.hpp"
#include "../typed_python/PythonSerializationContext.hpp"


class PyView {
public:
    PyObject_HEAD;
    std::shared_ptr<View> state;

    static void dealloc(PyView *self)
    {
        self->state.~shared_ptr();
        Py_TYPE(self)->tp_free((PyObject*)self);
    }

    static PyObject *new_(PyTypeObject* type, PyObject* args, PyObject* kwargs)
    {
        PyView* self;

        self = (PyView*) type->tp_alloc(type, 0);

        if (self != NULL) {
            new (&self->state) std::shared_ptr<View>();
        }

        return (PyObject*)self;
    }

    static int init(PyView* self, PyObject* args, PyObject* kwargs)
    {
        static const char *kwlist[] = {"databaseConnectionState", "transaction_id", "allowWrites", NULL};

        PyObject* databaseConnectionState;
        transaction_id tid;
        bool allowWrites;

        if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Olb", (char**)kwlist, &databaseConnectionState, &tid, &allowWrites)) {
            return -1;
        }

        if (databaseConnectionState->ob_type != &PyType_DatabaseConnectionState) {
            PyErr_Format(PyExc_TypeError, "Expected a DatabaseConnectionState, got %S", databaseConnectionState->ob_type);
            return -1;
        }

        self->state.reset(new View(
            ((PyDatabaseConnectionState*)databaseConnectionState)->state,
            tid,
            allowWrites
        ));

        return 0;
    }

    static PyObject* enter(PyView* self, PyObject* args, PyObject* kwargs) {
        static const char *kwlist[] = {NULL};

        if (!PyArg_ParseTupleAndKeywords(args, kwargs, "", (char**)kwlist)) {
            return NULL;
        }

        self->state->enter();
        return incref(Py_None);
    }

    static PyObject* exit(PyView* self, PyObject* args, PyObject* kwargs) {
        static const char *kwlist[] = {NULL};

        if (!PyArg_ParseTupleAndKeywords(args, kwargs, "", (char**)kwlist)) {
            return NULL;
        }

        self->state->exit();
        return incref(Py_None);
    }


    static PyObject* extractReads(PyView* self, PyObject* args, PyObject* kwargs) {
        static const char *kwlist[] = {NULL};

        if (!PyArg_ParseTupleAndKeywords(args, kwargs, "", (char**)kwlist)) {
            return NULL;
        }

        ListOf<ObjectFieldId> out;

        for (auto key: self->state->getReadValues()) {
            out.append(ObjectFieldId(key.second, key.first, false));
        }

        return out.toPython();
    }

    static PyObject* extractWrites(PyView* self, PyObject* args, PyObject* kwargs) {
        static const char *kwlist[] = {NULL};

        if (!PyArg_ParseTupleAndKeywords(args, kwargs, "", (char**)kwlist)) {
            return NULL;
        }

        return translateExceptionToPyObject([&]() {
            Dict<ObjectFieldId, OneOf<None, Bytes> > out;

            {
                PyEnsureGilReleased releaseTheGil;

                auto context = self->state->getSerializationContext();

                for (auto keyAndCache: self->state->getWriteCache()) {
                    SerializationBuffer b(*context);

                    keyAndCache.second.type()->serialize(keyAndCache.second.data(), b, 0);

                    b.finalize();

                    out[ObjectFieldId(keyAndCache.first.second, keyAndCache.first.first, false)] = Bytes((const char*)b.buffer(), b.size());
                }

                for (auto keyAndCache: self->state->getDeleteCache()) {
                    out[ObjectFieldId(keyAndCache.second, keyAndCache.first, false)] = None();
                }
            }

            return out.toPython();
        });
    }

    static PyObject* extractIndexReads(PyView* self, PyObject* args, PyObject* kwargs) {
        static const char *kwlist[] = {NULL};

        if (!PyArg_ParseTupleAndKeywords(args, kwargs, "", (char**)kwlist)) {
            return NULL;
        }

        ListOf<IndexId> out;

        for (auto indexKey: self->state->getSetReads()) {
            out.append(IndexId(indexKey.fieldId(), indexKey.indexValue()));
        }

        return out.toPython();
    }

    static PyObject* extractSetAdds(PyView* self, PyObject* args, PyObject* kwargs) {
        static const char *kwlist[] = {NULL};

        if (!PyArg_ParseTupleAndKeywords(args, kwargs, "", (char**)kwlist)) {
            return NULL;
        }

        Dict<IndexId, ListOf<object_id> > out;

        for (auto indexKeyToOids: self->state->getSetAdds()) {
            ListOf<object_id> oids = out[IndexId(indexKeyToOids.first.fieldId(), indexKeyToOids.first.indexValue())];
            for (auto oid: indexKeyToOids.second) {
                oids.append(oid);
            }
        }

        return out.toPython();
    }

    static PyObject* extractSetRemoves(PyView* self, PyObject* args, PyObject* kwargs) {
        static const char *kwlist[] = {NULL};

        if (!PyArg_ParseTupleAndKeywords(args, kwargs, "", (char**)kwlist)) {
            return NULL;
        }

        Dict<IndexId, ListOf<object_id> > out;

        for (auto indexKeyToOids: self->state->getSetRemoves()) {
            ListOf<object_id> oids = out[IndexId(indexKeyToOids.first.fieldId(), indexKeyToOids.first.indexValue())];
            for (auto oid: indexKeyToOids.second) {
                oids.append(oid);
            }
        }

        return out.toPython();
    }

    static PyObject* setSerializationContext(PyView* self, PyObject* args, PyObject* kwargs)
    {
        static const char *kwlist[] = {"serializationContext", NULL};

        PyObject* context;

        if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", (char**)kwlist, &context)) {
            return NULL;
        }

        return translateExceptionToPyObject([&]() {
            if (!self->state) {
                throw std::runtime_error("Invalid PyView (nullptr)");
            }

            self->state->setSerializationContext(
                std::shared_ptr<SerializationContext>(
                    new PythonSerializationContext(context)
                )
            );

            return incref(Py_None);
        });
    }

};

extern PyTypeObject PyType_View;
