/******************************************************************************
   Copyright 2017-2024 typed_python Authors

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

#include <Python.h>
#include <frameobject.h>

// Compatibility shims for Python 3.10 vs 3.11+.
//
// Python 3.11 made PyFrameObject and PyCodeObject opaque, removing
// direct field access. This header provides inline accessors that
// work across versions.
//
// Functions returning PyObject*/PyFrameObject* return NEW references
// on 3.11+ and BORROWED references on 3.10. Callers that only read
// the result transiently can use the RAII guard PyObjBorrower below
// to Py_DECREF when needed.

namespace PyCompat {

// ── Frame access ────────────────────────────────────────────────

// Get the current frame from a thread state.
// 3.11+: returns a NEW reference (caller must decref).
// 3.10:  returns a BORROWED reference.
inline PyFrameObject* getFrame(PyThreadState* tstate) {
#if PY_VERSION_HEX >= 0x030b0000
    return PyThreadState_GetFrame(tstate);
#else
    return tstate->frame;
#endif
}

// Get the globals dict from a frame.
// 3.11+: returns a NEW reference.
// 3.10:  returns a BORROWED reference.
inline PyObject* getFrameGlobals(PyFrameObject* frame) {
#if PY_VERSION_HEX >= 0x030b0000
    return PyFrame_GetGlobals(frame);
#else
    return frame->f_globals;
#endif
}

// ── Code object field access ────────────────────────────────────
//
// In 3.11+ all PyCodeObject fields are private.  The getters below
// return NEW references on 3.11+ and BORROWED references on 3.10.

inline PyObject* codeGetCode(PyCodeObject* co) {
#if PY_VERSION_HEX >= 0x030b0000
    return PyCode_GetCode(co);       // new ref
#else
    return co->co_code;              // borrowed
#endif
}

inline PyObject* codeGetConsts(PyCodeObject* co) {
#if PY_VERSION_HEX >= 0x030b0000
    // PyCode_GetConsts is not available; use the generic attr accessor
    return PyObject_GetAttrString((PyObject*)co, "co_consts");
#else
    return co->co_consts;
#endif
}

inline PyObject* codeGetNames(PyCodeObject* co) {
#if PY_VERSION_HEX >= 0x030b0000
    return PyObject_GetAttrString((PyObject*)co, "co_names");
#else
    return co->co_names;
#endif
}

inline PyObject* codeGetVarnames(PyCodeObject* co) {
#if PY_VERSION_HEX >= 0x030b0000
    return PyObject_GetAttrString((PyObject*)co, "co_varnames");
#else
    return co->co_varnames;
#endif
}

inline PyObject* codeGetFreevars(PyCodeObject* co) {
#if PY_VERSION_HEX >= 0x030b0000
    return PyObject_GetAttrString((PyObject*)co, "co_freevars");
#else
    return co->co_freevars;
#endif
}

inline PyObject* codeGetCellvars(PyCodeObject* co) {
#if PY_VERSION_HEX >= 0x030b0000
    return PyObject_GetAttrString((PyObject*)co, "co_cellvars");
#else
    return co->co_cellvars;
#endif
}

inline PyObject* codeGetName(PyCodeObject* co) {
#if PY_VERSION_HEX >= 0x030b0000
    return PyObject_GetAttrString((PyObject*)co, "co_name");
#else
    return co->co_name;
#endif
}

inline PyObject* codeGetLinetable(PyCodeObject* co) {
#if PY_VERSION_HEX >= 0x030b0000
    return PyObject_GetAttrString((PyObject*)co, "co_linetable");
#elif PY_VERSION_HEX >= 0x030a0000
    return co->co_linetable;
#else
    return co->co_lnotab;
#endif
}

inline int codeGetArgcount(PyCodeObject* co) {
#if PY_VERSION_HEX >= 0x030b0000
    return co->co_argcount;  // still public in 3.11
#else
    return co->co_argcount;
#endif
}

inline int codeGetKwonlyargcount(PyCodeObject* co) {
#if PY_VERSION_HEX >= 0x030b0000
    return co->co_kwonlyargcount;
#else
    return co->co_kwonlyargcount;
#endif
}

inline int codeGetNlocals(PyCodeObject* co) {
#if PY_VERSION_HEX >= 0x030b0000
    return co->co_nlocals;
#else
    return co->co_nlocals;
#endif
}

inline int codeGetStacksize(PyCodeObject* co) {
#if PY_VERSION_HEX >= 0x030b0000
    return co->co_stacksize;
#else
    return co->co_stacksize;
#endif
}

inline int codeGetFirstlineno(PyCodeObject* co) {
#if PY_VERSION_HEX >= 0x030b0000
    return co->co_firstlineno;
#else
    return co->co_firstlineno;
#endif
}

// ── RAII guard for version-dependent ref semantics ──────────────
//
// On 3.11+ the getters above return new references that must be
// decref'd.  On 3.10 they return borrowed references and must NOT
// be decref'd.  This guard handles both cases.
//
// Usage:
//   PyObject* names = PyCompat::codeGetNames(co);
//   PyCompat::NewRefIf311 guard(names);
//   // use names ...
//   // guard decrefs on destruction (3.11+) or does nothing (3.10)

struct NewRefIf311 {
    PyObject* obj;
    NewRefIf311(PyObject* o) : obj(o) {}
    ~NewRefIf311() {
#if PY_VERSION_HEX >= 0x030b0000
        Py_XDECREF(obj);
#endif
    }
    NewRefIf311(const NewRefIf311&) = delete;
    NewRefIf311& operator=(const NewRefIf311&) = delete;
};

} // namespace PyCompat
