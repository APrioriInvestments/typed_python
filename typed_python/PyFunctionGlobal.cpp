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
    {NULL}  /* Sentinel */
};

/* static */
void PyFunctionGlobal::dealloc(PyFunctionGlobal *self)
{
    Py_TYPE(self)->tp_free((PyObject*)self);
}

PyObject* PyFunctionGlobal::newPyFunctionGlobal(FunctionGlobal* g) {
    PyFunctionGlobal* self = (PyFunctionGlobal*)PyType_FunctionGlobal.tp_alloc(&PyType_FunctionGlobal, 0);
    self->mGlobal = g;
    return (PyObject*)self;
}

/* static */
PyObject* PyFunctionGlobal::new_(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    PyFunctionGlobal* self;

    self = (PyFunctionGlobal*)type->tp_alloc(type, 0);

    if (self != NULL) {
        self->mGlobal = nullptr;
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
