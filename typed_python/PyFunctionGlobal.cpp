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

#include "PyFunctionGlobal.hpp"


PyDoc_STRVAR(PyFunctionGlobal_doc,
    "A single overload of a Function type object.\n\n"
);

PyMethodDef PyFunctionGlobal_methods[] = {
    {"isUnresolved", (PyCFunction)PyFunctionGlobal::isUnresolved, METH_VARARGS | METH_KEYWORDS, NULL},
    {"getValue", (PyCFunction)PyFunctionGlobal::getValue, METH_VARARGS | METH_KEYWORDS, NULL},
    {NULL}  /* Sentinel */
};


PyObject* PyFunctionGlobal::getValue(PyFunctionGlobal* self, PyObject* args, PyObject* kwargs) {
    static const char* kwlist[] = {NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "", (char**)kwlist)) {
        return NULL;
    }

    return translateExceptionToPyObject([&]() {
        PyObject* result = self->getGlobal().getValueAsPyobj();
        if (!result) {
            // TODO: should this throw a NameError?
            throw std::runtime_error("Global is not resolved");
        }
        return incref(result);
    });
}

PyObject* PyFunctionGlobal::isUnresolved(PyFunctionGlobal* self, PyObject* args, PyObject* kwargs) {
    static const char* kwlist[] = {"insistForwardsResolved", NULL};

    int insistForwardsResolved = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|p", (char**)kwlist, &insistForwardsResolved)) {
        return NULL;
    }

    return translateExceptionToPyObject([&]() {
        return incref(self->getGlobal().isUnresolved(insistForwardsResolved) ? Py_True : Py_False);
    });
}

FunctionGlobal& PyFunctionGlobal::getGlobal() {
    if (!mFuncType) {
        throw std::runtime_error("FunctionGlobal is missing a function type");
    }

    if (!mName) {
        throw std::runtime_error("FunctionGlobal is missing a name");
    }

    if (mOverloadIx < 0 || mOverloadIx >= mFuncType->getOverloads().size()){
        throw std::runtime_error("FunctionGlobal overload is out of bounds");
    }

    FunctionOverload& o = mFuncType->getOverloads()[mOverloadIx];

    auto it = o.getGlobals().find(*mName);
    if (it == o.getGlobals().end()) {
        throw std::runtime_error("FunctionGlobal has no global named " + *mName);
    }

    return it->second;
}

/* static */
void PyFunctionGlobal::dealloc(PyFunctionGlobal *self)
{
    if (self->mName) {
        delete self->mName;
    }
    Py_TYPE(self)->tp_free((PyObject*)self);
}

PyObject* PyFunctionGlobal::newPyFunctionGlobal(Function* func, long overloadIx, std::string globalName) {
    PyFunctionGlobal* self = (PyFunctionGlobal*)PyType_FunctionGlobal.tp_alloc(&PyType_FunctionGlobal, 0);
    self->mFuncType = func;
    self->mOverloadIx = overloadIx;
    self->mName = new std::string(globalName);
    return (PyObject*)self;
}

/* static */
PyObject* PyFunctionGlobal::new_(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    PyFunctionGlobal* self;

    self = (PyFunctionGlobal*)type->tp_alloc(type, 0);

    if (self != NULL) {
        self->mFuncType = nullptr;
        self->mOverloadIx = 0;
        self->mName = nullptr;
    }

    return (PyObject*)self;
}

/* static */
int PyFunctionGlobal::init(PyFunctionGlobal *self, PyObject *args, PyObject *kwargs)
{
    PyErr_Format(PyExc_RuntimeError, "FunctionGlobal cannot be initialized directly");
    return -1;
}

PyTypeObject PyType_FunctionGlobal = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "FunctionGlobal",
    .tp_basicsize = sizeof(PyFunctionGlobal),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor) PyFunctionGlobal::dealloc,
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
    .tp_doc = PyFunctionGlobal_doc,
    .tp_traverse = 0,
    .tp_clear = 0,
    .tp_richcompare = 0,
    .tp_weaklistoffset = 0,
    .tp_iter = 0,
    .tp_iternext = 0,
    .tp_methods = PyFunctionGlobal_methods,
    .tp_members = 0,
    .tp_getset = 0,
    .tp_base = 0,
    .tp_dict = 0,
    .tp_descr_get = 0,
    .tp_descr_set = 0,
    .tp_dictoffset = 0,
    .tp_init = (initproc) PyFunctionGlobal::init,
    .tp_alloc = 0,
    .tp_new = PyFunctionGlobal::new_,
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
