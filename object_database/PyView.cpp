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

#include "PyView.hpp"

PyMethodDef PyView_methods[] = {
    {"enter", (PyCFunction)PyView::enter, METH_VARARGS | METH_KEYWORDS, NULL},
    {"exit", (PyCFunction)PyView::exit, METH_VARARGS | METH_KEYWORDS, NULL},
    {"setSerializationContext", (PyCFunction)PyView::setSerializationContext, METH_VARARGS | METH_KEYWORDS, NULL},
    {"extractReads", (PyCFunction)PyView::extractReads, METH_VARARGS | METH_KEYWORDS, NULL},
    {"extractWrites", (PyCFunction)PyView::extractWrites, METH_VARARGS | METH_KEYWORDS, NULL},
    {"extractIndexReads", (PyCFunction)PyView::extractIndexReads, METH_VARARGS | METH_KEYWORDS, NULL},
    {"extractSetAdds", (PyCFunction)PyView::extractSetAdds, METH_VARARGS | METH_KEYWORDS, NULL},
    {"extractSetRemoves", (PyCFunction)PyView::extractSetRemoves, METH_VARARGS | METH_KEYWORDS, NULL},
    {NULL}  /* Sentinel */
};

PyTypeObject PyType_View = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "View",
    .tp_basicsize = sizeof(PyView),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor) PyView::dealloc,
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
    .tp_methods = PyView_methods,
    .tp_members = 0,
    .tp_getset = 0,
    .tp_base = 0,
    .tp_dict = 0,
    .tp_descr_get = 0,
    .tp_descr_set = 0,
    .tp_dictoffset = 0,
    .tp_init = (initproc) PyView::init,
    .tp_alloc = 0,
    .tp_new = PyView::new_,
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

