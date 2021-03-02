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

#include "PyTemporaryReferenceTracer.hpp"


PyDoc_STRVAR(PyTemporaryReferenceTracer_doc,
    "A 'function tracer' (in the sys.settrace sense) for temporary reference conversion\n\n"
    "When you write 'x = l[ix]' and 'l' contains a reference to a HeldClass, or \n"
    "any time you write 'x = someExpression' where 'x' results in a temporary ref\n"
    "we use a PyTemporaryReferenceTracker to turn it into a copy of the object\n"
    "when the current expression has finished executing.\n"
);

PyObject* PyTemporaryReferenceTracer::call(PyObject* o, PyObject* a, PyObject* kwargs) {
    // we're being triggered as a function-level trace. that means the caller has a
    // refcount on us.
    PyTemporaryReferenceTracer* self = (PyTemporaryReferenceTracer*)o;

    try {
        self->triggerSelf();
    } catch(PythonExceptionSet& e) {
        return NULL;
    }

    // if we are the global tracer, then simply remove us from the trace list
    PyThreadState *tstate = PyThreadState_GET();
    if (tstate->c_traceobj == (PyObject*)self) {
        // we are the current global trace function, but we're never going to do anything,
        // so just untrigger ourself
        PyEval_SetTrace(self->existingGlobalTracer, self->existingGlobalTracerArg);
    }

    if (self->existingFrameTracer) {
        // make sure to reinstate the existing frame tracer.
        PyObject* res = self->existingFrameTracer;
        self->existingFrameTracer = nullptr;

        PyObject* existingTraceRes = PyObject_Call(res, a, kwargs);

        // drop our refcount on 'res' since we set it to null in the object itself.
        decref(res);

        return existingTraceRes;
    }

    // we have no existing frame tracer, so this simply sets it back.
    return incref(Py_None);
}

void PyTemporaryReferenceTracer::triggerSelf() {
    // make sure we don't double trigger
    if (!toTrigger) {
        return;
    }

    Type* targetType = ((PyInstance*)toTrigger)->mContainingInstance.type();
    instance_ptr data = ((PyInstance*)toTrigger)->mContainingInstance.data();

    if (!targetType->isRefTo()) {
        PyErr_Format(
            PyExc_RuntimeError,
            "Internal error: somehow a PyTemporaryReferenceTracer is pointing at a non-ref."
        );

        throw PythonExceptionSet();
    }

    Type* actualType = ((RefTo*)targetType)->getEltType();
    instance_ptr actualData = *(instance_ptr*)data;

    PyTypeObject* actualTypeObj = PyInstance::typeObj(actualType);

    // this should work because all of our type objects expect the
    // object itself to be of the same binary format.
    toTrigger->ob_type = actualTypeObj;

    ((PyInstance*)toTrigger)->mContainingInstance = Instance::create(actualType, actualData);
    decref(toTrigger);
    toTrigger = nullptr;
}

int PyTemporaryReferenceTracer::globalTraceFun(PyObject* obj, PyFrameObject* frame, int what, PyObject* arg) {
    PyTemporaryReferenceTracer* self = (PyTemporaryReferenceTracer*)obj;

    try {
        self->triggerSelf();
    } catch (PythonExceptionSet& e) {
        return -1;
    }

    // we are the global trace fun. make sure we reinstate and then call the existing trace fun
    PyObject* tracerArg = self->existingGlobalTracerArg;
    Py_tracefunc tracer = self->existingGlobalTracer;

    PyEval_SetTrace(tracer, tracerArg);

    // call the original tracer
    if (tracer) {
        return tracer(tracerArg, frame, what, arg);
    }

    return 0;
}

void PyTemporaryReferenceTracer::dealloc(PyTemporaryReferenceTracer *self) {
    decref(self->toTrigger);
    decref(self->existingGlobalTracerArg);
    decref(self->existingFrameTracer);

    Py_TYPE(self)->tp_free((PyObject*)self);
}

PyObject *PyTemporaryReferenceTracer::new_(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
    PyTemporaryReferenceTracer* self = (PyTemporaryReferenceTracer*)type->tp_alloc(type, 0);

    self->toTrigger = nullptr;
    self->existingGlobalTracer = nullptr;
    self->existingGlobalTracerArg = nullptr;
    self->existingFrameTracer = nullptr;

    return (PyObject*)self;
}


PyTypeObject PyType_TemporaryReferenceTracer = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "TemporaryReferenceTracer",
    .tp_basicsize = sizeof(PyTemporaryReferenceTracer),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor) PyTemporaryReferenceTracer::dealloc,
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
    .tp_call = PyTemporaryReferenceTracer::call,
    .tp_str = 0,
    .tp_getattro = 0,
    .tp_setattro = 0,
    .tp_as_buffer = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = PyTemporaryReferenceTracer_doc,
    .tp_traverse = 0,
    .tp_clear = 0,
    .tp_richcompare = 0,
    .tp_weaklistoffset = 0,
    .tp_iter = 0,
    .tp_iternext = 0,
    .tp_methods = 0,
    .tp_members = 0,
    .tp_getset = 0,
    .tp_base = 0,
    .tp_dict = 0,
    .tp_descr_get = 0,
    .tp_descr_set = 0,
    .tp_dictoffset = 0,
    .tp_init = 0,
    .tp_alloc = 0,
    .tp_new = 0,
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
