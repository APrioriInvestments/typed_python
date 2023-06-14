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
#include <unordered_map>

enum class TraceAction {
    ConvertTemporaryReference,
    Decref,
    Autoresolve
};

class PyTemporaryReferenceTracer {
public:
    PyTemporaryReferenceTracer() :
        mostRecentEmptyFrame(nullptr),
        priorTraceFunc(nullptr),
        priorTraceFuncArg(nullptr)
    {}

    // perform an action on the first instruction where a
    // frame goes out of scope or where it is no longer on the
    // given line number
    class FrameAction {
    public:
        FrameAction(PyObject* inO, TraceAction inA, int inLine) :
            obj(inO),
            typ(nullptr),
            action(inA),
            lineNumber(inLine)
        {
        }

        FrameAction(Type* inT, TraceAction inA, int inLine) :
            obj(nullptr),
            typ(inT),
            action(inA),
            lineNumber(inLine)
        {
        }

        PyObject* obj;
        Type* typ;
        TraceAction action;
        int lineNumber;
    };

    std::unordered_map<PyFrameObject*, std::vector<FrameAction> > frameToActions;

    std::unordered_map<PyObject*, std::set<int> > codeObjectToExpressionLines;

    // the most recent frame we touched that has nothing in it
    PyFrameObject* mostRecentEmptyFrame;

    Py_tracefunc priorTraceFunc;

    PyObject* priorTraceFuncArg;

    bool isLineNewStatement(PyObject* code, int line);

    static PyTemporaryReferenceTracer globalTracer;

    static int globalTraceFun(PyObject* obj, PyFrameObject* frame, int what, PyObject* arg);

    // the next time we have an instruction in 'frame', trigger 'o' to become
    // a non-temporary reference
    static void traceObject(PyObject* o, PyFrameObject* frame);

    static void traceObject(PyObject* o);

    // on the next instruction of 'frame', attempt to autoresolve 't'
    static void autoresolveOnNextInstruction(Type* t, PyFrameObject* frame);

    // on the next instruction of 'frame', attempt to autoresolve 't'
    static Type* autoresolveOnNextInstruction(Type* t);

    static void installGlobalTraceHandlerIfNecessary();

    static void keepaliveForCurrentInstruction(PyObject* o);

    static void keepaliveForCurrentInstruction(PyObject* o, PyFrameObject* frame);
};
