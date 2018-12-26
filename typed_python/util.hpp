#pragma once

static_assert(PY_MAJOR_VERSION >= 3, "nativepython is a python3 project only");
static_assert(PY_MINOR_VERSION >= 6, "nativepython is a python3.6 project only");

// thread-local counter for how many times we've disabled native dispatch
extern thread_local int64_t native_dispatch_disabled;


inline PyObject* incref(PyObject* o) {
    Py_INCREF(o);
    return o;
}
