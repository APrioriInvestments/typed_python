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

#include "PyPyObjGraphSnapshot.hpp"


PyDoc_STRVAR(PyPyObjGraphSnapshot_doc,
    "A snapshot of a collection of python objects.\n\n"
);

PyMethodDef PyPyObjGraphSnapshot_methods[] = {
    {"extractTypes", (PyCFunction)PyPyObjGraphSnapshot::extractTypes, METH_VARARGS | METH_KEYWORDS, NULL},
    {"getObjects", (PyCFunction)PyPyObjGraphSnapshot::getObjects, METH_VARARGS | METH_KEYWORDS, NULL},
    {"internalize", (PyCFunction)PyPyObjGraphSnapshot::internalize, METH_VARARGS | METH_KEYWORDS, NULL},
    {"resolveForwards", (PyCFunction)PyPyObjGraphSnapshot::resolveForwards, METH_VARARGS | METH_KEYWORDS, NULL},
    {"hashToObject", (PyCFunction)PyPyObjGraphSnapshot::hashToObject, METH_VARARGS | METH_KEYWORDS, NULL},
    {NULL}  /* Sentinel */
};


/* static */
void PyPyObjGraphSnapshot::dealloc(PyPyObjGraphSnapshot *self)
{
    if (self->mOwnsSnapshot) {
        delete self->mGraphSnapshot;
    }

    Py_TYPE(self)->tp_free((PyObject*)self);
}

PyObject* PyPyObjGraphSnapshot::newPyObjGraphSnapshot(PyObjGraphSnapshot* g, bool ownsIt) {
    PyPyObjGraphSnapshot* self = (PyPyObjGraphSnapshot*)PyType_PyObjGraphSnapshot.tp_alloc(&PyType_PyObjGraphSnapshot, 0);
    self->mGraphSnapshot = g;
    self->mOwnsSnapshot = ownsIt;

    return (PyObject*)self;
}

/* static */
PyObject* PyPyObjGraphSnapshot::hashToObject(PyObject* graph, PyObject *args, PyObject *kwargs) {
    PyPyObjGraphSnapshot* self = (PyPyObjGraphSnapshot*)graph;

    return translateExceptionToPyObject([&]() {
        static const char *kwlist[] = {"hash", NULL};

        PyObject* hashPyobj = nullptr;

        if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", (char**)kwlist, &hashPyobj)) {
            throw PythonExceptionSet();
        }

        if (!PyUnicode_Check(hashPyobj)) {
            throw std::runtime_error("Invalid sha hash argument");
        }

        const char* hash = PyUnicode_AsUTF8(hashPyobj);

        ShaHash h = ShaHash::fromHexDigest(hash);

        PyObjSnapshot* s = self->mGraphSnapshot->snapshotForHash(h);

        if (!s) {
            return incref(Py_None);
        }

        return PyPyObjSnapshot::newPyObjSnapshot(s, graph);
    });
}

/* static */
PyObject* PyPyObjGraphSnapshot::internalize(PyObject* graph, PyObject *args, PyObject *kwargs) {
    PyPyObjGraphSnapshot* self = (PyPyObjGraphSnapshot*)graph;

    return translateExceptionToPyObject([&]() {
        static const char *kwlist[] = {"sourceGraph", NULL};

        PyObject* pySourceGraph = nullptr;

        if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", (char**)kwlist, &pySourceGraph)) {
            throw PythonExceptionSet();
        }

        if (pySourceGraph->ob_type != &PyType_PyObjGraphSnapshot) {
            throw std::runtime_error("Expected a PyObjGraphSnapshot for 'sourceGraph'");
        }

        PyPyObjGraphSnapshot* sourceGraph = (PyPyObjGraphSnapshot*)pySourceGraph;

        self->mGraphSnapshot->internalize(*sourceGraph->mGraphSnapshot, false);
        return incref(Py_None);
    });
}

/* static */
PyObject* PyPyObjGraphSnapshot::resolveForwards(PyObject* graph, PyObject *args, PyObject *kwargs) {
    PyPyObjGraphSnapshot* self = (PyPyObjGraphSnapshot*)graph;

    return translateExceptionToPyObject([&]() {
        static const char *kwlist[] = {NULL};

        if (!PyArg_ParseTupleAndKeywords(args, kwargs, "", (char**)kwlist)) {
            throw PythonExceptionSet();
        }

        self->mGraphSnapshot->resolveForwards();

        return incref(Py_None);
    });
}

/* static */
PyObject* PyPyObjGraphSnapshot::extractTypes(PyObject* graph, PyObject *args, PyObject *kwargs) {
    PyPyObjGraphSnapshot* self = (PyPyObjGraphSnapshot*)graph;

    return translateExceptionToPyObject([&]() {
        static const char *kwlist[] = {NULL};

        if (!PyArg_ParseTupleAndKeywords(args, kwargs, "", (char**)kwlist)) {
            throw PythonExceptionSet();
        }

        PyObjectStealer res(PySet_New(NULL));

        for (auto o: self->mGraphSnapshot->getObjectsByIndex()) {
            if (o->getType()) {
                PySet_Add(res, (PyObject*)PyInstance::typeObj(o->getType()));
            }
        }

        return incref(res);
    });
}

/* static */
PyObject* PyPyObjGraphSnapshot::getObjects(PyObject* graph, PyObject *args, PyObject *kwargs) {
    PyPyObjGraphSnapshot* self = (PyPyObjGraphSnapshot*)graph;

    return translateExceptionToPyObject([&]() {
        static const char *kwlist[] = {NULL};

        if (!PyArg_ParseTupleAndKeywords(args, kwargs, "", (char**)kwlist)) {
            throw PythonExceptionSet();
        }

        PyObjectStealer res(PyList_New(0));

        for (auto o: self->mGraphSnapshot->getObjectsByIndex()) {
            PyList_Append(res, PyPyObjSnapshot::newPyObjSnapshot(o, graph));
        }

        return incref(res);
    });
}

/* static */
PyObject* PyPyObjGraphSnapshot::new_(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    PyPyObjGraphSnapshot* self;

    self = (PyPyObjGraphSnapshot*)type->tp_alloc(type, 0);

    if (self != NULL) {
        self->mGraphSnapshot = nullptr;
        self->mOwnsSnapshot = false;
    }

    return (PyObject*)self;
}

PyObject* PyPyObjGraphSnapshot::tp_repr(PyObject *selfObj) {
    PyPyObjGraphSnapshot* pyGraph = (PyPyObjGraphSnapshot*)selfObj;

    return translateExceptionToPyObject([&]() {
        return PyUnicode_FromString(
            (
                "PyObjGraphSnapshot("
                + format(pyGraph->mGraphSnapshot->getObjects().size())
                + ")"
            ).c_str()
        );
    });
}


/* static */
int PyPyObjGraphSnapshot::init(PyPyObjGraphSnapshot *self, PyObject *args, PyObject *kwargs)
{
    self->mGraphSnapshot = new PyObjGraphSnapshot();
    self->mOwnsSnapshot = true;
    return 0;
}


PyTypeObject PyType_PyObjGraphSnapshot = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "PyObjGraph2Snapshot",
    .tp_basicsize = sizeof(PyPyObjGraphSnapshot),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor) PyPyObjGraphSnapshot::dealloc,
    #if PY_MINOR_VERSION < 8
    .tp_print = 0,
    #else
    .tp_vectorcall_offset = 0,                  // printfunc  (Changed to tp_vectorcall_offset in Python 3.8)
    #endif
    .tp_getattr = 0,
    .tp_setattr = 0,
    .tp_as_async = 0,
    .tp_repr = PyPyObjGraphSnapshot::tp_repr,
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
    .tp_doc = PyPyObjGraphSnapshot_doc,
    .tp_traverse = 0,
    .tp_clear = 0,
    .tp_richcompare = 0,
    .tp_weaklistoffset = 0,
    .tp_iter = 0,
    .tp_iternext = 0,
    .tp_methods = PyPyObjGraphSnapshot_methods,
    .tp_members = 0,
    .tp_getset = 0,
    .tp_base = 0,
    .tp_dict = 0,
    .tp_descr_get = 0,
    .tp_descr_set = 0,
    .tp_dictoffset = 0,
    .tp_init = (initproc) PyPyObjGraphSnapshot::init,
    .tp_alloc = 0,
    .tp_new = PyPyObjGraphSnapshot::new_,
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
