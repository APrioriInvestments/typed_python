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

#include "PyTemporaryReferenceTracer.hpp"

bool PyTemporaryReferenceTracer::isLineNewStatement(PyObject* code, int line) {
    PyErrorStasher stashCurrentException;

    auto it = codeObjectToExpressionLines.find(code);

    if (it != codeObjectToExpressionLines.end()) {
        return it->second.find(line) != it->second.end();
    }

    // this permanently memoizes this code object in this global object
    // this should be OK because there are (usually) a small and finite number of
    // code objects in a given program.
    incref(code);
    auto& lineNumbers = codeObjectToExpressionLines[code];

    PyObject* internals = internalsModule();

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
    if (frame != globalTracer.mostRecentEmptyFrame &&
        globalTracer.frameToActions.find(frame) != globalTracer.frameToActions.end()) {
        // always process exception and return statements
        bool forceProcess = (
            what == PyTrace_EXCEPTION ||
            what == PyTrace_RETURN
        );

        // we process any statement on a line that's a new statement
        bool shouldProcess = globalTracer.isLineNewStatement(
            (PyObject*)frame->f_code,
            PyFrame_GetLineNumber(frame)
        );

        if (shouldProcess || forceProcess) {
            auto it = globalTracer.frameToActions.find(frame);

            if (it == globalTracer.frameToActions.end()) {
                globalTracer.mostRecentEmptyFrame = frame;
            } else {
                globalTracer.mostRecentEmptyFrame = nullptr;

                std::vector<FrameAction> persistingActions;

                for (auto& frameAction: it->second) {
                    if (frameAction.lineNumber != PyFrame_GetLineNumber(frame) || forceProcess) {
                        if (frameAction.action == TraceAction::ConvertTemporaryReference) {
                            ((PyInstance*)frameAction.obj)->resolveTemporaryReference();
                            decref(frameAction.obj);
                        }
                        else if (frameAction.action == TraceAction::Decref) {
                            decref(frameAction.obj);
                        }
                        else if (frameAction.action == TraceAction::Autoresolve) {
                            PyErrorStasher stashCurrentException;

                            // attempt to autoresolve the type.  Note that if we
                            // fail to do so, we just swallow the exception, which
                            // is not great, but we can't be throwing exceptions
                            // in here...
                            try {
                                frameAction.typ->tryToAutoresolve();
                            } catch(std::exception& e) {
                                // just swallow it
                            } catch(PythonExceptionSet& e) {
                                PyErr_Clear();
                            }
                        }
                    } else {
                        persistingActions.push_back(frameAction);
                    }
                }

                if (persistingActions.size()) {
                    it->second = persistingActions;
                } else {
                    globalTracer.frameToActions.erase(it);
                }
            }
        }
    }

    int res = 0;
    if (globalTracer.priorTraceFunc) {
        res = globalTracer.priorTraceFunc(
            globalTracer.priorTraceFuncArg, frame, what, arg
        );
    }

    if (globalTracer.frameToActions.size() == 0) {
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
    globalTracer.frameToActions[f].push_back(
        FrameAction(
            incref(o),
            TraceAction::ConvertTemporaryReference,
            PyFrame_GetLineNumber(f)
        )
    );

    if (globalTracer.mostRecentEmptyFrame == f) {
        globalTracer.mostRecentEmptyFrame = nullptr;
    }

    installGlobalTraceHandlerIfNecessary();
}

void PyTemporaryReferenceTracer::keepaliveForCurrentInstruction(PyObject* o, PyFrameObject* f) {
    // mark that we're going to trace
    globalTracer.frameToActions[f].push_back(
        FrameAction(
            incref(o),
            TraceAction::Decref,
            PyFrame_GetLineNumber(f)
        )
    );

    if (globalTracer.mostRecentEmptyFrame == f) {
        globalTracer.mostRecentEmptyFrame = nullptr;
    }

    installGlobalTraceHandlerIfNecessary();
}

void PyTemporaryReferenceTracer::autoresolveOnNextInstruction(Type* t, PyFrameObject* f) {
    // mark that we're going to trace
    globalTracer.frameToActions[f].push_back(
        FrameAction(
            t,
            TraceAction::Autoresolve,
            PyFrame_GetLineNumber(f)
        )
    );

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

Type* PyTemporaryReferenceTracer::autoresolveOnNextInstruction(Type* o) {
    if (!o->isForwardDefined() || o->isResolved()) {
        return o;
    }

    PyThreadState *tstate = PyThreadState_GET();
    PyFrameObject *f = tstate->frame;

    // search upwards until we find a frame that's not running in typed_python.internals
    auto isInternals = [&]() {
        if (!PyUnicode_Check(f->f_code->co_filename)) {
            return false;
        }

        return endsWith(PyUnicode_AsUTF8(f->f_code->co_filename), "typed_python/internals.py");
    };

    while (isInternals()) {
        f = PyFrame_GetBack(f);
        if (!f) {
            throw std::runtime_error("Frame failed");
        }
    }

    if (f) {
        PyTemporaryReferenceTracer::autoresolveOnNextInstruction(o, f);
    }

    return o;
}

void PyTemporaryReferenceTracer::keepaliveForCurrentInstruction(PyObject* o) {
    PyThreadState *tstate = PyThreadState_GET();
    PyFrameObject *f = tstate->frame;

    if (f) {
        PyTemporaryReferenceTracer::keepaliveForCurrentInstruction(o, f);
    }
}


PyTemporaryReferenceTracer PyTemporaryReferenceTracer::globalTracer;
