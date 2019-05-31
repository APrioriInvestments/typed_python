/******************************************************************************
   Copyright 2017-2019 Nativepython Authors

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
#include "PyGilState.hpp"
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
    assertHoldingTheGil();

    if (o) {
        Py_INCREF(o);
    }
    return o;
}


inline PyTypeObject* incref(PyTypeObject* o) {
    return (PyTypeObject*)incref((PyObject*)o);
}


inline PyObject* decref(PyObject* o) {
    assertHoldingTheGil();

    if (o) {
        Py_DECREF(o);
    }
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
            decref(m_ptr);
        }

        m_ptr = other.m_ptr;

        return *this;
    }

    PyObjectHolder& steal(PyObject* other) {
        if (m_ptr) {
            decref(m_ptr);
        }

        m_ptr = other;

        return *this;
    }

    ~PyObjectHolder() {
        if (m_ptr) {
            decref(m_ptr);
        }
    }

    operator bool() const {
        return m_ptr;
    }

    operator PyObject*() const {
        return m_ptr;
    }

    operator PyTypeObject*() const {
        return (PyTypeObject*)m_ptr;
    }

    PyObject* operator->() const {
        return m_ptr;
    }

    void set(PyObject* val) {
        if (val == m_ptr) {
            return;
        }
        if (val) {
            incref(val);
        }
        if (m_ptr) {
            decref(m_ptr);
        }

        m_ptr = val;
    }

    //pull the pointer out without decrefing it
    PyObject* extract() {
        PyObject* result = m_ptr;
        m_ptr = nullptr;
        return result;
    }

    //reset the pointer to null and decref existing
    PyObject* release() {
        PyObject* result = m_ptr;

        if (m_ptr) {
            decref(m_ptr);
            m_ptr = nullptr;
        }

        return result;
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

inline PyObject* incref(const PyObjectHolder& o) {
    return incref((PyObject*)o);
}

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

/******
Call 'f', which must return PyObject*, in a block that guards against
exceptions returning nakedly to the python interpreter. This is meant
to guard the barrier between Python C callbacks and our internals which
may use exceptions.
******/
template<class func_type>
PyObject* translateExceptionToPyObject(func_type f) {
    try {
        return f();
    } catch(PythonExceptionSet& e) {
        return nullptr;
    } catch(std::exception& e) {
        PyErr_SetString(PyExc_TypeError, e.what());
        return nullptr;
    }
}


/******
Call 'f', which must return void, in a block that guards against
exceptions returning nakedly to the python interpreter. This is meant
to guard the barrier between Python C callbacks and our internals which
may use exceptions. Returns -1 on failure, 0 on success.
******/
template<class func_type>
int translateExceptionToPyObjectReturningInt(func_type f) {
    try {
        f();
        return 0;
    } catch(PythonExceptionSet& e) {
        return -1;
    } catch(std::exception& e) {
        PyErr_SetString(PyExc_TypeError, e.what());
        return -1;
    }
}

/********
Iterate over 'o' calling 'f' with each PyObject encountered.

May throw, so use in conjunction with 'translateExceptionToPyObject'
********/
template<class func_type>
void iterate(PyObject* o, func_type f) {
    PyObject *iterator = PyObject_GetIter(o);
    PyObject *item;

    if (iterator == NULL) {
        throw PythonExceptionSet();
    }

    while ((item = PyIter_Next(iterator))) {
        try {
            f(item);
            Py_DECREF(item);
        } catch(...) {
            Py_DECREF(item);
            Py_DECREF(iterator);
            throw;
        }
    }

    Py_DECREF(iterator);

    if (PyErr_Occurred()) {
        throw PythonExceptionSet();
    }
}
