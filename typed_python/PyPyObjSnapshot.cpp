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

#include "PyPyObjSnapshot.hpp"
#include "PyPyObjGraphSnapshot.hpp"
#include "PyObjGraphSnapshot.hpp"

PyDoc_STRVAR(PyPyObjSnapshot_doc,
    "A single object in a snapshot graph.\n\n"
);

PyMethodDef PyPyObjSnapshot_methods[] = {
    {"create", (PyCFunction)PyPyObjSnapshot::create, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
    {NULL}  /* Sentinel */
};


/* static */
PyObject* PyPyObjSnapshot::create(PyObject* self, PyObject* args, PyObject* kwargs) {
    return translateExceptionToPyObject([&]() {
        PyObject* instance;
        int linkBack = 1;
        static const char *kwlist[] = {"instance", "linkBackToOriginalObject", NULL};

        if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|p", (char**)kwlist, &instance, &linkBack)) {
            throw PythonExceptionSet();
        }

        std::unordered_map<PyObject*, PyObjSnapshot*> constantMapCache;
        std::unordered_map<Type*, PyObjSnapshot*> constantMapCacheType;
        std::unordered_map<InstanceRef, PyObjSnapshot*> constantMapCacheInst;

        PyObjGraphSnapshot* graph = new PyObjGraphSnapshot();

        PyObjSnapshotMaker maker(
            constantMapCache,
            constantMapCacheType,
            constantMapCacheInst,
            graph,
            linkBack
        );

        PyObjSnapshot* object = maker.internalize(instance);

        PyObjectStealer graphObj(PyPyObjGraphSnapshot::newPyObjGraphSnapshot(graph, true));

        return PyPyObjSnapshot::newPyObjSnapshot(object, graphObj);
    });
}

/* static */
void PyPyObjSnapshot::dealloc(PyPyObjSnapshot *self)
{
    decref(self->mElements);
    decref(self->mKeys);
    decref(self->mGraph);
    decref(self->mByKey);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

PyObject* PyPyObjSnapshot::newPyObjSnapshot(PyObjSnapshot* g, PyObject* pyGraphPtr) {
    PyPyObjSnapshot* self = (PyPyObjSnapshot*)PyType_PyObjSnapshot.tp_alloc(&PyType_PyObjSnapshot, 0);

    self->mPyobj = g;
    self->mGraph = incref(pyGraphPtr);
    self->mElements = nullptr;
    self->mKeys = nullptr;
    self->mByKey = nullptr;

    return (PyObject*)self;
}

/* static */
PyObject* PyPyObjSnapshot::new_(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    PyPyObjSnapshot* self;

    self = (PyPyObjSnapshot*)type->tp_alloc(type, 0);

    if (self != NULL) {
        self->mPyobj = nullptr;
        self->mElements = nullptr;
        self->mKeys = nullptr;
        self->mGraph = nullptr;
        self->mByKey = nullptr;
    }

    return (PyObject*)self;
}

PyObject* PyPyObjSnapshot::tp_getattro(PyObject* selfObj, PyObject* attrName) {
    PyPyObjSnapshot* pySnap = (PyPyObjSnapshot*)selfObj;

    return translateExceptionToPyObject([&] {
        if (!PyUnicode_Check(attrName)) {
            throw std::runtime_error("Expected a string for attribute name");
        }

        std::string attr(PyUnicode_AsUTF8(attrName));

        PyObjSnapshot* obj = pySnap->mPyobj;

        auto it = obj->namedElements().find(attr);
        if (it != obj->namedElements().end()) {
            return PyPyObjSnapshot::newPyObjSnapshot(
                it->second,
                it->second->getGraph() == obj->getGraph() ? pySnap->mGraph : nullptr
            );
        }

        auto it2 = obj->namedInts().find(attr);
        if (it2 != obj->namedInts().end()) {
            return PyLong_FromLong(it2->second);
        }

        if (attr == "pyobj") {
            PyObject* o = obj->getPyObj();
            if (!o) {
                throw std::runtime_error("PyObjSnapshot of kind " + obj->kindAsString() + " has no pyobj");
            }
            return incref(o);
        }

        if (attr == "kind") {
            return PyUnicode_FromString(obj->kindAsString().c_str());
        }

        if (attr == "graph") {
            return incref(pySnap->mGraph ? pySnap->mGraph : Py_None);
        }

        if (attr == "shaHash" && obj->getGraph()) {
            ShaHash hash = obj->getGraph()->hashFor(obj);
            return PyUnicode_FromString(hash.digestAsHexString().c_str());
        }

        if (attr == "type" && obj->getType()) {
            return incref((PyObject*)PyInstance::typeObj(obj->getType()));
        }

        if (attr == "instance" && obj->isInstance()) {
            return PyInstance::fromInstance(obj->getInstance());
        }

        if (attr == "stringValue" && obj->isString()) {
            return PyUnicode_FromString(obj->getStringValue().c_str());
        }

        if (attr == "name") {
            return PyUnicode_FromString(obj->getName().c_str());
        }

        if (attr == "moduleName") {
            return PyUnicode_FromString(obj->getModuleName().c_str());
        }

        if (attr == "elements") {
            if (!pySnap->mElements) {
                pySnap->mElements = PyList_New(0);
                for (auto p: obj->elements()) {
                    PyList_Append(
                        pySnap->mElements,
                        PyPyObjSnapshot::newPyObjSnapshot(
                            p,
                            p->getGraph() == obj->getGraph() ? pySnap->mGraph : nullptr
                        )
                    );
                }
            }

            return incref(pySnap->mElements);
        }

        if (attr == "keys") {
            if (!pySnap->mKeys) {
                pySnap->mKeys = PyList_New(0);
                for (auto p: obj->keys()) {
                    PyList_Append(
                        pySnap->mKeys,
                        PyPyObjSnapshot::newPyObjSnapshot(
                            p,
                            p->getGraph() == obj->getGraph() ? pySnap->mGraph : nullptr
                        )
                    );
                }
            }

            return incref(pySnap->mKeys);
        }

        if (attr == "byKey") {
            if (!pySnap->mByKey) {
                pySnap->mByKey = PyDict_New();
                for (long k = 0; k < obj->keys().size(); k++) {
                    PyObjSnapshot* p = obj->elements()[k];

                    PyDict_SetItem(
                        pySnap->mByKey,
                        obj->keys()[k]->getPyObj(),
                        PyPyObjSnapshot::newPyObjSnapshot(
                            p,
                            p->getGraph() == obj->getGraph() ? pySnap->mGraph : nullptr
                        )
                    );
                }
            }

            return incref(pySnap->mByKey);
        }

        return PyObject_GenericGetAttr(selfObj, attrName);
    });
}

PyObject* PyPyObjSnapshot::tp_repr(PyObject *selfObj) {
    PyPyObjSnapshot* self = (PyPyObjSnapshot*)selfObj;

    return translateExceptionToPyObject([&]() {
        return PyUnicode_FromString(self->mPyobj->toString().c_str());
    });
}


/* static */
int PyPyObjSnapshot::init(PyPyObjSnapshot *self, PyObject *args, PyObject *kwargs)
{
    PyErr_Format(PyExc_RuntimeError, "PyObjSnapshot cannot be initialized directly");
    return -1;
}


PyTypeObject PyType_PyObjSnapshot = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "PyObjSnapshot",
    .tp_basicsize = sizeof(PyPyObjSnapshot),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor) PyPyObjSnapshot::dealloc,
    #if PY_MINOR_VERSION < 8
    .tp_print = 0,
    #else
    .tp_vectorcall_offset = 0,                  // printfunc  (Changed to tp_vectorcall_offset in Python 3.8)
    #endif
    .tp_getattr = 0,
    .tp_setattr = 0,
    .tp_as_async = 0,
    .tp_repr = PyPyObjSnapshot::tp_repr,
    .tp_as_number = 0,
    .tp_as_sequence = 0,
    .tp_as_mapping = 0,
    .tp_hash = 0,
    .tp_call = 0,
    .tp_str = 0,
    .tp_getattro = PyPyObjSnapshot::tp_getattro,
    .tp_setattro = 0,
    .tp_as_buffer = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = PyPyObjSnapshot_doc,
    .tp_traverse = 0,
    .tp_clear = 0,
    .tp_richcompare = 0,
    .tp_weaklistoffset = 0,
    .tp_iter = 0,
    .tp_iternext = 0,
    .tp_methods = PyPyObjSnapshot_methods,
    .tp_members = 0,
    .tp_getset = 0,
    .tp_base = 0,
    .tp_dict = 0,
    .tp_descr_get = 0,
    .tp_descr_set = 0,
    .tp_dictoffset = 0,
    .tp_init = (initproc) PyPyObjSnapshot::init,
    .tp_alloc = 0,
    .tp_new = PyPyObjSnapshot::new_,
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
