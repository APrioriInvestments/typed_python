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

bool PyTemporaryReferenceTracer::isLineNewStatement(PyObject* code, int line) {
    auto it = codeObjectToExpressionLines.find(code);

    if (it != codeObjectToExpressionLines.end()) {
        return it->second.find(line) != it->second.end();
    }

    // this permanently memoizes this code object in this global object
    // this should be OK because there are (usually) a small and finite number of
    // code objects in a given program.
    incref(code);
    auto& lineNumbers = codeObjectToExpressionLines[code];

    static PyObject* internals = internalsModule();

    PyObjectStealer res(
        PyObject_CallMethod(internals, "extractCodeObjectNewStatementLineNumbers", "O", code, NULL)
    );

    if (!res) {
        PyErr_Clear();
    } else {
        iterate((PyObject*)res, [&](PyObject* lineNo) {
            if (PyLong_Check(lineNo)) {
                lineNumbers.insert(PyLong_AsLong(lineNo));
            }
        });
    }

    return lineNumbers.find(line) != lineNumbers.end();
}


int PyTemporaryReferenceTracer::globalTraceFun(PyObject* dummyObj, PyFrameObject* frame, int what, PyObject* arg) {
    if (frame != globalTracer.mostRecentEmptyFrame) {
        bool shouldProcess = true;

        if (what == PyTrace_LINE) {
            shouldProcess = globalTracer.isLineNewStatement(
                (PyObject*)frame->f_code,
                frame->f_lineno
            );
        }

        if (shouldProcess) {
            auto it = globalTracer.frameToHandles.find(frame);

            if (it == globalTracer.frameToHandles.end()) {
                globalTracer.mostRecentEmptyFrame = frame;
            } else {
                globalTracer.mostRecentEmptyFrame = nullptr;

                for (auto objAndAction: it->second) {
                    if (objAndAction.second == TraceAction::ConvertTemporaryReference) {
                        ((PyInstance*)objAndAction.first)->resolveTemporaryReference();
                    }

                    decref(objAndAction.first);
                }

                globalTracer.frameToHandles.erase(it);
            }
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

void PyTemporaryReferenceTracer::installGlobalTraceHandlerIfNecessary() {
    // ensure that the global trace handler is installed
    PyThreadState *tstate = PyThreadState_GET();

    // this swallows the reference we're holding on 'tracer' into the function itself
    if (tstate->c_tracefunc != globalTraceFun) {
        globalTracer.priorTraceFunc = tstate->c_tracefunc;
        globalTracer.priorTraceFuncArg = incref(tstate->c_traceobj);

        PyEval_SetTrace(PyTemporaryReferenceTracer::globalTraceFun, nullptr);
    }
}

void PyTemporaryReferenceTracer::traceObject(PyObject* o, PyFrameObject* f) {
    // mark that we're going to trace
    globalTracer.frameToHandles[f].push_back(std::make_pair(incref(o), TraceAction::ConvertTemporaryReference));

    if (globalTracer.mostRecentEmptyFrame == f) {
        globalTracer.mostRecentEmptyFrame = nullptr;
    }

    installGlobalTraceHandlerIfNecessary();
}

void PyTemporaryReferenceTracer::keepaliveForCurrentInstruction(PyObject* o, PyFrameObject* f) {
    // mark that we're going to trace
    globalTracer.frameToHandles[f].push_back(std::make_pair(incref(o), TraceAction::Decref));

    if (globalTracer.mostRecentEmptyFrame == f) {
        globalTracer.mostRecentEmptyFrame = nullptr;
    }

    installGlobalTraceHandlerIfNecessary();
}

void PyTemporaryReferenceTracer::traceObject(PyObject* o) {
    PyThreadState *tstate = PyThreadState_GET();
    PyFrameObject *f = tstate->frame;

    if (f) {
        PyTemporaryReferenceTracer::traceObject(o, f);
    }
}

void PyTemporaryReferenceTracer::keepaliveForCurrentInstruction(PyObject* o) {
    PyThreadState *tstate = PyThreadState_GET();
    PyFrameObject *f = tstate->frame;

    if (f) {
        PyTemporaryReferenceTracer::keepaliveForCurrentInstruction(o, f);
    }
}


PyTemporaryReferenceTracer PyTemporaryReferenceTracer::globalTracer;
