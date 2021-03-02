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


int PyTemporaryReferenceTracer::globalTraceFun(PyObject* obj, PyFrameObject* frame, int what, PyObject* arg) {
    if (frame != globalTracer.mostRecentEmptyFrame) {
        auto it = globalTracer.frameToHandles.find(frame);

        if (it == globalTracer.frameToHandles.end()) {
            globalTracer.mostRecentEmptyFrame = frame;
        } else {
            globalTracer.mostRecentEmptyFrame = nullptr;

            for (auto obj: it->second) {
                ((PyInstance*)obj)->resolveTemporaryReference();
                decref(obj);
            }

            globalTracer.frameToHandles.erase(it);
        }
    }

    int res = 0;
    if (globalTracer.priorTraceFunc) {
        res = globalTracer.priorTraceFunc(
            globalTracer.priorTraceFuncArg, frame, what, arg
        );
    }

    if (globalTracer.frameToHandles.size() == 0) {
        // uninstall ourself
        PyEval_SetTrace(globalTracer.priorTraceFunc, globalTracer.priorTraceFuncArg);
        decref(globalTracer.priorTraceFuncArg);

        globalTracer.priorTraceFunc = nullptr;
        globalTracer.priorTraceFuncArg = nullptr;
    }

    return res;
}


void PyTemporaryReferenceTracer::traceObject(PyObject* o, PyFrameObject* f) {
    // mark that we're going to trace
    globalTracer.frameToHandles[f].push_back(incref(o));

    // reset the
    if (globalTracer.mostRecentEmptyFrame == f) {
        globalTracer.mostRecentEmptyFrame = nullptr;
    }

    PyThreadState *tstate = PyThreadState_GET();

    // this swallows the reference we're holding on 'tracer' into the function itself
    if (tstate->c_tracefunc != globalTraceFun) {
        globalTracer.priorTraceFunc = tstate->c_tracefunc;
        globalTracer.priorTraceFuncArg = incref(tstate->c_traceobj);

        PyEval_SetTrace(PyTemporaryReferenceTracer::globalTraceFun, nullptr);
    }
}

PyTemporaryReferenceTracer PyTemporaryReferenceTracer::globalTracer;
