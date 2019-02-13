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

//given a python richcompare flag, such as Py_LT, and a c-style 'cmp' result, compute
//the resulting boolean value
inline bool cmpResultToBoolForPyOrdering(int pyComparisonOp, char cmpResult) {
    switch (pyComparisonOp) {
        case Py_EQ: return cmpResult == 0;
        case Py_NE: return cmpResult != 0;
        case Py_LT: return cmpResult < 0;
        case Py_GT: return cmpResult > 0;
        case Py_LE: return cmpResult <= 0;
        case Py_GE: return cmpResult >= 0;
    }

    throw std::logic_error("Invalid pyComparisonOp");
}

// thread-local counter for the currently released threadstate
extern thread_local PyThreadState* curPyThreadState;

//scoped object to ensure we're not holding the GIL. If we've
//already released it, this is a no-op. Upon destruction, we
//reacquire it.
class PyEnsureGilReleased {
public:
    PyEnsureGilReleased() :
        m_should_reaquire(false)
    {
        if (curPyThreadState == nullptr) {
            curPyThreadState = PyEval_SaveThread();
            m_should_reaquire = true;
        }
    }

    ~PyEnsureGilReleased() {
        if (m_should_reaquire) {
            PyEval_RestoreThread(curPyThreadState);
            curPyThreadState = nullptr;
        }
    }

private:
    bool m_should_reaquire;
};

//scoped object to ensure we're holding the GIL. If
//we released it in a thread above, this should reacquire it.
//if we already hold it, it should be a no-op
class PyEnsureGilAcquired {
public:
    PyEnsureGilAcquired() :
        m_should_rerelease(false)
    {
        if (curPyThreadState) {
            PyEval_RestoreThread(curPyThreadState);
            m_should_rerelease = true;
            curPyThreadState = nullptr;
        }
    }

    ~PyEnsureGilAcquired() {
        if (m_should_rerelease) {
            curPyThreadState = PyEval_SaveThread();
        }
    }

private:
    bool m_should_rerelease;
};
