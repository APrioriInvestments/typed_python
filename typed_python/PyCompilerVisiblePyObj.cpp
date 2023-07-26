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

#include "PyCompilerVisiblePyObj.hpp"


PyDoc_STRVAR(PyCompilerVisiblePyObj_doc,
    "A single overload of a Function type object.\n\n"
);

PyMethodDef PyCompilerVisiblePyObj_methods[] = {
    {"create", (PyCFunction)PyCompilerVisiblePyObj::create, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
    {NULL}  /* Sentinel */
};


/* static */
PyObject* PyCompilerVisiblePyObj::create(PyObject* self, PyObject* args, PyObject* kwargs) {
    return translateExceptionToPyObject([&]() {
        PyObject* instance;
        int linkBack = 1;
        static const char *kwlist[] = {"instance", "linkBackToOriginalObject", NULL};

        if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|p", (char**)kwlist, &instance, &linkBack)) {
            throw PythonExceptionSet();
        }

        std::unordered_map<PyObject*, CompilerVisiblePyObj*> constantMapCache;
        std::map<::Type*, ::Type*> groupMap;

        CompilerVisiblePyObj* object = CompilerVisiblePyObj::internalizePyObj(
            instance,
            constantMapCache,
            groupMap,
            linkBack
        );

        return PyCompilerVisiblePyObj::newPyCompilerVisiblePyObj(object);
    });
}

/* static */
void PyCompilerVisiblePyObj::dealloc(PyCompilerVisiblePyObj *self)
{
    decref(self->mElements);
    decref(self->mKeys);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

PyObject* PyCompilerVisiblePyObj::newPyCompilerVisiblePyObj(CompilerVisiblePyObj* g) {
    static std::unordered_map<CompilerVisiblePyObj*, PyObject*> memo;

    auto it = memo.find(g);
    if (it != memo.end()) {
        return incref(it->second);
    }

    PyCompilerVisiblePyObj* self = (PyCompilerVisiblePyObj*)PyType_CompilerVisiblePyObj.tp_alloc(&PyType_CompilerVisiblePyObj, 0);
    self->mPyobj = g;
    self->mElements = nullptr;
    self->mKeys = nullptr;
    self->mByKey = nullptr;

    memo[g] = incref((PyObject*)self);

    return (PyObject*)self;
}

/* static */
PyObject* PyCompilerVisiblePyObj::new_(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    PyCompilerVisiblePyObj* self;

    self = (PyCompilerVisiblePyObj*)type->tp_alloc(type, 0);

    if (self != NULL) {
        self->mPyobj = nullptr;
        self->mElements = nullptr;
        self->mKeys = nullptr;
        self->mByKey = nullptr;
    }

    return (PyObject*)self;
}

PyObject* PyCompilerVisiblePyObj::tp_getattro(PyObject* selfObj, PyObject* attrName) {
    PyCompilerVisiblePyObj* pyCVPO = (PyCompilerVisiblePyObj*)selfObj;

    return translateExceptionToPyObject([&] {
        if (!PyUnicode_Check(attrName)) {
            throw std::runtime_error("Expected a string for attribute name");
        }

        std::string attr(PyUnicode_AsUTF8(attrName));

        CompilerVisiblePyObj* obj = pyCVPO->mPyobj;

        auto it = obj->namedElements().find(attr);
        if (it != obj->namedElements().end()) {
            return PyCompilerVisiblePyObj::newPyCompilerVisiblePyObj(it->second);
        }

        auto it2 = obj->namedInts().find(attr);
        if (it2 != obj->namedInts().end()) {
            return PyLong_FromLong(it2->second);
        }

        if (attr == "pyobj") {
            PyObject* o = obj->getPyObj();
            if (!o) {
                throw std::runtime_error("CVPO of kind " + obj->kindAsString() + " has no pyobj");
            }
            return incref(o);
        }

        if (attr == "kind") {
            return PyUnicode_FromString(obj->kindAsString().c_str());
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
            if (!pyCVPO->mElements) {
                pyCVPO->mElements = PyList_New(0);
                for (auto p: obj->elements()) {
                    PyList_Append(pyCVPO->mElements, PyCompilerVisiblePyObj::newPyCompilerVisiblePyObj(p));
                }
            }

            return incref(pyCVPO->mElements);
        }

        if (attr == "keys") {
            if (!pyCVPO->mKeys) {
                pyCVPO->mKeys = PyList_New(0);
                for (auto p: obj->keys()) {
                    PyList_Append(pyCVPO->mKeys, PyCompilerVisiblePyObj::newPyCompilerVisiblePyObj(p));
                }
            }

            return incref(pyCVPO->mKeys);
        }

        if (attr == "byKey") {
            if (!pyCVPO->mByKey) {
                pyCVPO->mByKey = PyDict_New();
                for (long k = 0; k < obj->keys().size(); k++) {
                    PyDict_SetItem(
                        pyCVPO->mByKey,
                        obj->keys()[k]->getPyObj(),
                        PyCompilerVisiblePyObj::newPyCompilerVisiblePyObj(obj->elements()[k])
                    );
                }
            }

            return incref(pyCVPO->mByKey);
        }

        return PyObject_GenericGetAttr(selfObj, attrName);
    });
}

PyObject* PyCompilerVisiblePyObj::tp_repr(PyObject *selfObj) {
    PyCompilerVisiblePyObj* self = (PyCompilerVisiblePyObj*)selfObj;

    return translateExceptionToPyObject([&]() {
        return PyUnicode_FromString(self->mPyobj->toString().c_str());
    });
}


/* static */
int PyCompilerVisiblePyObj::init(PyCompilerVisiblePyObj *self, PyObject *args, PyObject *kwargs)
{
    PyErr_Format(PyExc_RuntimeError, "CompilerVisiblePyObj cannot be initialized directly");
    return -1;
}


PyTypeObject PyType_CompilerVisiblePyObj = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "CompilerVisiblePyObj",
    .tp_basicsize = sizeof(PyCompilerVisiblePyObj),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor) PyCompilerVisiblePyObj::dealloc,
    #if PY_MINOR_VERSION < 8
    .tp_print = 0,
    #else
    .tp_vectorcall_offset = 0,                  // printfunc  (Changed to tp_vectorcall_offset in Python 3.8)
    #endif
    .tp_getattr = 0,
    .tp_setattr = 0,
    .tp_as_async = 0,
    .tp_repr = PyCompilerVisiblePyObj::tp_repr,
    .tp_as_number = 0,
    .tp_as_sequence = 0,
    .tp_as_mapping = 0,
    .tp_hash = 0,
    .tp_call = 0,
    .tp_str = 0,
    .tp_getattro = PyCompilerVisiblePyObj::tp_getattro,
    .tp_setattro = 0,
    .tp_as_buffer = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = PyCompilerVisiblePyObj_doc,
    .tp_traverse = 0,
    .tp_clear = 0,
    .tp_richcompare = 0,
    .tp_weaklistoffset = 0,
    .tp_iter = 0,
    .tp_iternext = 0,
    .tp_methods = PyCompilerVisiblePyObj_methods,
    .tp_members = 0,
    .tp_getset = 0,
    .tp_base = 0,
    .tp_dict = 0,
    .tp_descr_get = 0,
    .tp_descr_set = 0,
    .tp_dictoffset = 0,
    .tp_init = (initproc) PyCompilerVisiblePyObj::init,
    .tp_alloc = 0,
    .tp_new = PyCompilerVisiblePyObj::new_,
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
