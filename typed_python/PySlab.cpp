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

#include "PySlab.hpp"


PyDoc_STRVAR(PySlab_doc,
    "Models a contiguous block of memory backing some TP objects.\n\n"
    "Slabs get created by using the 'deepcopyContiguous' method to copy\n"
    "a python object graph into a Slab.  Python objects don't go in the Slab\n"
    "since their memory is controlled by the interpreter, but all \n"
    "typed_python instances get duplicated into the slab. The slab will be\n"
    "released when the last object in the slab is released.\n\n"
    "The primary purpose of a Slab is to allow us to indicate that a group\n"
    "of objects have the same lifetime, so that they can be allocated in \n"
    "one large allocation from the operating system. This prevents the severe\n"
    "fragmentation that can occur in big systems with may large objects each\n"
    "consisting of small ones.\n\n"
    "If you construct the slab with trackInternalReferences=True then we will\n"
    "maintain a list of each allocation, whether it's alive, and Type it was.\n"
    "This can be used to get diagnostics if your Slabs are leaking."
);

PyDoc_STRVAR(PySlab_refcount_doc,
    "Slab.refcount() -> int\n\n"
    "Return the refcount of the Slab. Each living reference in the slab is\n"
    "a refcount, as well as the python object you're calling from."
);

PyDoc_STRVAR(PySlab_bytecount_doc,
    "Slab.bytecount() -> int\n\n"
    "Return the total number of bytes allocated to the slab."
);

PyDoc_STRVAR(PySlab_allocCount_doc,
    "Slab.allocCount() -> int\n\n"
    "Return the total number of distinct allocations made in the slab.\n"
    "This will be 0 if the slab is not tracking internal allocations.\n"
);

PyDoc_STRVAR(PySlab_liveAllocCount_doc,
    "Slab.liveAllocCount() -> int\n\n"
    "Return the total number of allocations in the slab that are still active.\n"
    "This will be 0 if the slab is not tracking internal allocations.\n"
);

PyDoc_STRVAR(PySlab_extractObject_doc,
    "Slab.extractObject(index) -> None or object\n\n"
    "Given an integer identifying the index of the allocation, return the object\n"
    "that's allocated there. This will be None if the object is released. Note that\n"
    "this may segfault if you release the object from another thread just as you\n"
    "call this method, so use this for diagnostic purposes only."
);

PyDoc_STRVAR(PySlab_allocIsAlive_doc,
    "Slab.allocIsAlive(index) -> bool\n\n"
    "Return True if the alloc at index `index` is not freed yet."
);

PyDoc_STRVAR(PySlab_allocType_doc,
    "Slab.allocType(index) -> Type or None\n\n"
    "Return the Type of the alloc at `index`. This will be None if the object is\n"
    "not the root of an object (for instance, internals of hash tables are not associated\n"
    "with the Dict object that holds them). This will be None if we're not tracking\n"
    "internals."
);

PyDoc_STRVAR(PySlab_allocRefcount_doc,
    "Slab.allocRefcount(index) -> int\n\n"
    "Return the current refcount of the object allocated at index `index`. If\n"
    "we're not tracking internals, this will be 0."
);

PyDoc_STRVAR(PySlab_slabPtr_doc,
    "Slab.slabPtr() -> int\n\n"
    "Get the root address of the memory slab as an integer."
);

PyMethodDef PySlabInstance_methods[] = {
    {"refcount", (PyCFunction)PySlab::refcount, METH_VARARGS | METH_KEYWORDS, PySlab_refcount_doc},
    {"bytecount", (PyCFunction)PySlab::bytecount, METH_VARARGS | METH_KEYWORDS, PySlab_bytecount_doc},
    {"allocCount", (PyCFunction)PySlab::allocCount, METH_VARARGS | METH_KEYWORDS, PySlab_allocCount_doc},
    {"liveAllocCount", (PyCFunction)PySlab::liveAllocCount, METH_VARARGS | METH_KEYWORDS, PySlab_liveAllocCount_doc},
    {"extractObject", (PyCFunction)PySlab::extractObject, METH_VARARGS | METH_KEYWORDS, PySlab_extractObject_doc},
    {"allocIsAlive", (PyCFunction)PySlab::allocIsAlive, METH_VARARGS | METH_KEYWORDS, PySlab_allocIsAlive_doc},
    {"allocType", (PyCFunction)PySlab::allocType, METH_VARARGS | METH_KEYWORDS, PySlab_allocType_doc},
    {"allocRefcount", (PyCFunction)PySlab::allocRefcount, METH_VARARGS | METH_KEYWORDS, PySlab_allocRefcount_doc},
    {"slabPtr", (PyCFunction)PySlab::slabPtr, METH_VARARGS | METH_KEYWORDS, PySlab_slabPtr_doc},
    {"getTag", (PyCFunction)PySlab::getTag, METH_VARARGS | METH_KEYWORDS, NULL},
    {"setTag", (PyCFunction)PySlab::setTag, METH_VARARGS | METH_KEYWORDS, NULL},
    {NULL}  /* Sentinel */
};

PyObject* PySlab::getTag(PySlab* self, PyObject* args, PyObject* kwargs) {
    PyObject* o = self->mSlab->getTag();
    if (!o) {
        return incref(Py_None);
    }
    return incref(o);
}

PyObject* PySlab::setTag(PySlab* self, PyObject* args, PyObject* kwargs) {
    static const char *kwlist[] = {"tag", NULL};

    PyObject* tag;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", (char**)kwlist, &tag)) {
        return NULL;
    }

    self->mSlab->setTag(tag);

    return incref(Py_None);
}

PyObject* PySlab::slabPtr(PySlab* self, PyObject* args, PyObject* kwargs)
{
    static const char *kwlist[] = {NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "", (char**)kwlist)) {
        return NULL;
    }

    return translateExceptionToPyObject([&]() {
        return PyLong_FromLong((size_t)self->mSlab);
    });
}


PyObject* PySlab::refcount(PySlab* self, PyObject* args, PyObject* kwargs)
{
    static const char *kwlist[] = {NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "", (char**)kwlist)) {
        return NULL;
    }

    return translateExceptionToPyObject([&]() {
        return PyLong_FromLong(self->mSlab->refcount());
    });
}


PyObject* PySlab::bytecount(PySlab* self, PyObject* args, PyObject* kwargs)
{
    static const char *kwlist[] = {NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "", (char**)kwlist)) {
        return NULL;
    }

    return translateExceptionToPyObject([&]() {
        return PyLong_FromLong(self->mSlab->getBytecount());
    });
}



/* static */
void PySlab::dealloc(PySlab *self)
{
    if (self->mSlab) {
        self->mSlab->decref();
    }

    Py_TYPE(self)->tp_free((PyObject*)self);
}

PyObject* PySlab::newPySlab(Slab* s) {
    PySlab* self = (PySlab*)PyType_Slab.tp_alloc(&PyType_Slab, 0);
    self->mSlab = s;
    self->mSlab->incref();
    return (PyObject*)self;
}

/* static */
PyObject* PySlab::new_(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    PySlab* self;

    self = (PySlab*)type->tp_alloc(type, 0);

    if (self != NULL) {
        self->mSlab = nullptr;
    }

    return (PyObject*)self;
}

/* static */
int PySlab::init(PySlab *self, PyObject *args, PyObject *kwargs)
{
    PyErr_Format(PyExc_RuntimeError, "Slab cannot be initialized directly");
    return -1;
}

PyObject* PySlab::allocCount(PySlab* self, PyObject* args, PyObject* kwargs) {
    static const char* kwlist[] = {NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "", (char**)kwlist)) {
        return NULL;
    }

    return PyLong_FromLong(self->mSlab->allocCount());
}

PyObject* PySlab::liveAllocCount(PySlab* self, PyObject* args, PyObject* kwargs) {
    static const char* kwlist[] = {NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "", (char**)kwlist)) {
        return NULL;
    }

    return PyLong_FromLong(self->mSlab->liveAllocCount());
}

PyObject* PySlab::extractObject(PySlab* self, PyObject* args, PyObject* kwargs) {
    static const char* kwlist[] = {"index", NULL};

    long index;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "l", (char**)kwlist, &index)) {
        return NULL;
    }

    return translateExceptionToPyObject([&]() {
        if (!self->mSlab->allocIsAlive(index)) {
            throw std::runtime_error("Object was destroyed.");
        }

        void* ptr = self->mSlab->allocPointer(index);
        Type* t = self->mSlab->allocType(index);

        if (!ptr || !t) {
            return incref(Py_None);
        }

        if (t->isDict() || t->isListOf() || t->isClass()
                || t->isTupleOf() || t->isConstDict() || t->isAlternative()
                || t->isConcreteAlternative() || t->isSet() || t->isString() || t->isBytes()
                || t->isPythonObjectOfType()
            ) {
            return PyInstance::extractPythonObject(
                (instance_ptr)&ptr,
                t
            );
        }

        return incref(Py_None);
    });
}

PyObject* PySlab::allocIsAlive(PySlab* self, PyObject* args, PyObject* kwargs) {
    static const char* kwlist[] = {"index", NULL};

    long index;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "l", (char**)kwlist, &index)) {
        return NULL;
    }

    return translateExceptionToPyObject([&]() {
        return incref(self->mSlab->allocIsAlive(index) ? Py_True : Py_False);
    });
}

PyObject* PySlab::allocType(PySlab* self, PyObject* args, PyObject* kwargs) {
    static const char* kwlist[] = {"index", NULL};

    long index;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "l", (char**)kwlist, &index)) {
        return NULL;
    }

    return translateExceptionToPyObject([&]() {
        Type* t = self->mSlab->allocType(index);

        if (!t) {
            return incref(Py_None);
        }

        return incref((PyObject*)PyInstance::typeObj(t));
    });
}


PyObject* PySlab::allocRefcount(PySlab* self, PyObject* args, PyObject* kwargs) {
    static const char* kwlist[] = {"index", NULL};

    long index;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "l", (char**)kwlist, &index)) {
        return NULL;
    }

    return translateExceptionToPyObject([&]() {
        return PyLong_FromLong(self->mSlab->allocRefcount(index));
    });
}


PyTypeObject PyType_Slab = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "Slab",
    .tp_basicsize = sizeof(PySlab),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor) PySlab::dealloc,
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
    .tp_doc = PySlab_doc,
    .tp_traverse = 0,
    .tp_clear = 0,
    .tp_richcompare = 0,
    .tp_weaklistoffset = 0,
    .tp_iter = 0,
    .tp_iternext = 0,
    .tp_methods = PySlabInstance_methods,
    .tp_members = 0,
    .tp_getset = 0,
    .tp_base = 0,
    .tp_dict = 0,
    .tp_descr_get = 0,
    .tp_descr_set = 0,
    .tp_dictoffset = 0,
    .tp_init = (initproc) PySlab::init,
    .tp_alloc = 0,
    .tp_new = PySlab::new_,
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
