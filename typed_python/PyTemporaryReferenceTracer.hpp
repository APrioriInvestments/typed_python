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

class PyTemporaryReferenceTracer {
public:
    PyTemporaryReferenceTracer() :
        mostRecentEmptyFrame(nullptr),
        priorTraceFunc(nullptr),
        priorTraceFuncArg(nullptr)
    {}

    std::unordered_map<PyFrameObject*, std::vector<PyObject*> > frameToHandles;

    // the most recent frame we touched that has nothing in it
    PyFrameObject* mostRecentEmptyFrame;

    Py_tracefunc priorTraceFunc;

    PyObject* priorTraceFuncArg;

    static PyTemporaryReferenceTracer globalTracer;

    static int globalTraceFun(PyObject* obj, PyFrameObject* frame, int what, PyObject* arg);

    // the next time we have an instruction in 'frame', trigger 'o' to become
    // a non-temporary reference
    static void traceObject(PyObject* o, PyFrameObject* frame);
};
