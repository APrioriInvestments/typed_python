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

#pragma once

#include "PyInstance.hpp"

class PyTemporaryReferenceTracer {
public:
    PyObject_HEAD

    // the object we need to turn into a concrete ref on the next python instruction
    PyObject* toTrigger;

    // if a tracer was already on, this was the trace function. We'll reinstate it
    // after we've triggered
    Py_tracefunc existingGlobalTracer;

    // the arg to the existing global tracer.
    PyObject* existingGlobalTracerArg;

    // the existing frame tracer, if its on.
    PyObject* existingFrameTracer;

    // turn 'toTrigger' from a Ref to an actual HeldClass
    void triggerSelf();

    // the trace function we install when we are the 'global' trace function. We retained
    // the prior one, so we just trigger that one (and replace it in the current frame
    // for future calls). This allows us to chain ourselves in when other tracers are being
    // used.
    static int globalTraceFun(PyObject* obj, PyFrameObject* frame, int what, PyObject* arg);

    static void dealloc(PyTemporaryReferenceTracer *self);

    static PyObject *new_(PyTypeObject *type, PyObject *args, PyObject *kwargs);

    static PyObject* call(PyObject* self, PyObject* args, PyObject* kwargs);
};

extern PyTypeObject PyType_TemporaryReferenceTracer;
