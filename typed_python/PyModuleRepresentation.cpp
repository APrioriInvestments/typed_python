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

#include "PyModuleRepresentation.hpp"


PyDoc_STRVAR(PyModuleRepresentation_doc,
    "Models the dict of a module, with facilities for deepcopying mutually recursive functions."
);

PyMethodDef PyModuleRepresentationInstance_methods[] = {
    {"addExternal", (PyCFunction)PyModuleRepresentation::addExternal, METH_VARARGS | METH_KEYWORDS, NULL},
    {"getDict", (PyCFunction)PyModuleRepresentation::getDict, METH_VARARGS | METH_KEYWORDS, NULL},
    {"update", (PyCFunction)PyModuleRepresentation::update, METH_VARARGS | METH_KEYWORDS, NULL},
    {"getExternalReferences", (PyCFunction)PyModuleRepresentation::getExternalReferences, METH_VARARGS | METH_KEYWORDS, NULL},
    {"getInternalReferences", (PyCFunction)PyModuleRepresentation::getInternalReferences, METH_VARARGS | METH_KEYWORDS, NULL},
    {"getVisibleNames", (PyCFunction)PyModuleRepresentation::getVisibleNames, METH_VARARGS | METH_KEYWORDS, NULL},
    {"copyInto", (PyCFunction)PyModuleRepresentation::copyInto, METH_VARARGS | METH_KEYWORDS, NULL},
    {NULL}  /* Sentinel */
};


/* static */
void PyModuleRepresentation::dealloc(PyModuleRepresentation *self)
{
    if (self->mModuleRepresentation) {
        delete self->mModuleRepresentation;
    }

    Py_TYPE(self)->tp_free((PyObject*)self);
}

/* static */
PyObject* PyModuleRepresentation::new_(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    PyModuleRepresentation* self;

    self = (PyModuleRepresentation*)type->tp_alloc(type, 0);

    if (self != NULL) {
        self->mModuleRepresentation = nullptr;
    }

    return (PyObject*)self;
}

/* static */
int PyModuleRepresentation::init(PyModuleRepresentation *self, PyObject *args, PyObject *kwargs)
{
    static const char* kwlist[] = {"name", NULL};

    const char* name;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s", (char**)kwlist, &name)) {
        return -1;
    }

    self->mModuleRepresentation = new ModuleRepresentation(name);

    return 0;
}

PyObject* PyModuleRepresentation::addExternal(PyModuleRepresentation* self, PyObject* args, PyObject* kwargs) {
    static const char* kwlist[] = {"name", "value", NULL};

    PyObject* name;
    PyObject* value;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", (char**)kwlist, &name, &value)) {
        return nullptr;
    }

    return translateExceptionToPyObject([&]() {
        if (!self->mModuleRepresentation) {
            throw std::runtime_error("Unpopulated ModuleRepresentation");
        }

        self->mModuleRepresentation->addExternal(name, value);

        return incref(Py_None);
    });
}

PyObject* PyModuleRepresentation::getInternalReferences(PyModuleRepresentation* self, PyObject* args, PyObject* kwargs) {
    static const char* kwlist[] = {"name", NULL};

    const char* name;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s", (char**)kwlist, &name)) {
        return nullptr;
    }

    return translateExceptionToPyObject([&]() {
        if (!self->mModuleRepresentation) {
            throw std::runtime_error("Unpopulated ModuleRepresentation");
        }

        PyObject* res = PyList_New(0);

        for (auto o: self->mModuleRepresentation->getInternalReferences(name)) {
            PyList_Append(res, o.pyobj());
        }

        return res;
    });
}

PyObject* PyModuleRepresentation::getExternalReferences(PyModuleRepresentation* self, PyObject* args, PyObject* kwargs) {
    static const char* kwlist[] = {"name", NULL};

    const char* name;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s", (char**)kwlist, &name)) {
        return nullptr;
    }

    return translateExceptionToPyObject([&]() {
        if (!self->mModuleRepresentation) {
            throw std::runtime_error("Unpopulated ModuleRepresentation");
        }

        PyObject* res = PyList_New(0);

        for (auto o: self->mModuleRepresentation->getExternalReferences(name)) {
            PyList_Append(res, o.pyobj());
        }

        return res;
    });
}

PyObject* PyModuleRepresentation::getVisibleNames(PyModuleRepresentation* self, PyObject* args, PyObject* kwargs) {
    static const char* kwlist[] = {"name", NULL};

    const char* name;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s", (char**)kwlist, &name)) {
        return nullptr;
    }

    return translateExceptionToPyObject([&]() {
        if (!self->mModuleRepresentation) {
            throw std::runtime_error("Unpopulated ModuleRepresentation");
        }

        PyObject* res = PyList_New(0);

        for (auto o: self->mModuleRepresentation->getVisibleNames(name)) {
            PyList_Append(res, PyUnicode_FromString(o.c_str()));
        }

        return res;
    });
}

PyObject* PyModuleRepresentation::getDict(PyModuleRepresentation* self, PyObject* args, PyObject* kwargs) {
    static const char* kwlist[] = {NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "", (char**)kwlist)) {
        return nullptr;
    }

    return translateExceptionToPyObject([&]() {
        if (!self->mModuleRepresentation) {
            throw std::runtime_error("Unpopulated ModuleRepresentation");
        }

        return incref(PyModule_GetDict(self->mModuleRepresentation->mModuleObject.get()));
    });
}

PyObject* PyModuleRepresentation::update(PyModuleRepresentation* self, PyObject* args, PyObject* kwargs) {
    static const char* kwlist[] = {NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "", (char**)kwlist)) {
        return nullptr;
    }

    return translateExceptionToPyObject([&]() {
        if (!self->mModuleRepresentation) {
            throw std::runtime_error("Unpopulated ModuleRepresentation");
        }

        self->mModuleRepresentation->update();

        return incref(Py_None);
    });
}

PyObject* PyModuleRepresentation::copyInto(PyModuleRepresentation* self, PyObject* args, PyObject* kwargs) {
    static const char* kwlist[] = {"other", "names", NULL};

    PyObject* other;
    PyObject* names;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", (char**)kwlist, &other, &names)) {
        return nullptr;
    }

    return translateExceptionToPyObject([&]() {
        if (!self->mModuleRepresentation) {
            throw std::runtime_error("Unpopulated ModuleRepresentation");
        }

        if (other->ob_type != &PyType_ModuleRepresentation) {
            throw std::runtime_error("'other' must be a ModuleRepresentation");
        }

        PyModuleRepresentation* otherAsModule = (PyModuleRepresentation*)other;

        if (!otherAsModule->mModuleRepresentation) {
            throw std::runtime_error("Unpopulated ModuleRepresentation");
        }

        std::set<std::string> namesAsStrings;

        iterate(names, [&](PyObject* aName) {
            if (!PyUnicode_Check(aName)) {
                throw std::runtime_error("Expected 'names' to contain strings");
            }

            namesAsStrings.insert(PyUnicode_AsUTF8(aName));
        });


        self->mModuleRepresentation->copyInto(*otherAsModule->mModuleRepresentation, namesAsStrings);

        return incref(Py_None);
    });
}


PyTypeObject PyType_ModuleRepresentation = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "ModuleRepresentation",
    .tp_basicsize = sizeof(PyModuleRepresentation),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor) PyModuleRepresentation::dealloc,
    #if PY_MINOR_VERSION < 8
    .tp_print = 0,
    #else
    .tp_vectorcall_offset = 0,                  // printfunc  (Changed to tp_vectorcall_offset in Python 3.8)
    #endif
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
    .tp_doc = PyModuleRepresentation_doc,
    .tp_traverse = 0,
    .tp_clear = 0,
    .tp_richcompare = 0,
    .tp_weaklistoffset = 0,
    .tp_iter = 0,
    .tp_iternext = 0,
    .tp_methods = PyModuleRepresentationInstance_methods,
    .tp_members = 0,
    .tp_getset = 0,
    .tp_base = 0,
    .tp_dict = 0,
    .tp_descr_get = 0,
    .tp_descr_set = 0,
    .tp_dictoffset = 0,
    .tp_init = (initproc) PyModuleRepresentation::init,
    .tp_alloc = 0,
    .tp_new = PyModuleRepresentation::new_,
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

    #if PY_MINOR_VERSION >= 8
    .tp_vectorcall = 0,
    #endif

    #if PY_MINOR_VERSION == 8
    .tp_print = 0,
    #endif

};
