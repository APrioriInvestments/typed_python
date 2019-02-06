#pragma once

#include <Python.h>
#include <vector>
#include <string>

class Type;
class Instance;

static_assert(PY_MAJOR_VERSION >= 3, "nativepython is a python3 project only");
static_assert(PY_MINOR_VERSION >= 6, "nativepython is a python3.6 project only");

// thread-local counter for how many times we've disabled native dispatch
extern thread_local int64_t native_dispatch_disabled;

// indicates that we're unwinding the C stack with a python exception
// already set. Invoking code can return NULL to the python layer.
class PythonExceptionSet {};

inline PyObject* incref(PyObject* o) {
    Py_INCREF(o);
    return o;
}

class PyObjectHolder {
protected:
    PyObjectHolder(PyObject* p, bool ifPresentThenStealReference) : m_ptr(p) {
    }

public:
    PyObjectHolder() : m_ptr(nullptr) {
    }

    explicit PyObjectHolder(PyObject* p) : m_ptr(incref(p)) {
    }

    PyObjectHolder(const PyObjectHolder& other) {
        if (other.m_ptr) {
            m_ptr = incref(other.m_ptr);
        } else {
            m_ptr = nullptr;
        }
    }

    PyObjectHolder& operator=(const PyObjectHolder& other) {
        if (other.m_ptr) {
            incref(other.m_ptr);
        }
        if (m_ptr) {
            Py_DECREF(m_ptr);
        }

        m_ptr = other.m_ptr;

        return *this;
    }

    ~PyObjectHolder() {
        if (m_ptr) {
            Py_DECREF(m_ptr);
        }
    }

    operator bool() const {
        return m_ptr;
    }

    operator PyObject*() const {
        return m_ptr;
    }

    PyObject* operator->() const {
        return m_ptr;
    }

protected:
    PyObject* m_ptr;
};

class PyObjectStealer : public PyObjectHolder {
public:
    explicit PyObjectStealer(PyObject* p) : PyObjectHolder(p, true) {
    }
    PyObjectStealer(PyObjectHolder& other) = delete;
};

bool unpackTupleToTypes(PyObject* tuple, std::vector<Type*>& out);

bool unpackTupleToStringAndTypes(PyObject* tuple, std::vector<std::pair<std::string, Type*> >& out);

bool unpackTupleToStringTypesAndValues(PyObject* tuple, std::vector<std::tuple<std::string, Type*, Instance> >& out);

bool unpackTupleToStringAndObjects(PyObject* tuple, std::vector<std::pair<std::string, PyObject*> >& out);

