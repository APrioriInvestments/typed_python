/******************************************************************************
   Copyright 2017-2020 typed_python Authors

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
class MemberDefinition;

static_assert(PY_MAJOR_VERSION >= 3, "typed_python is a python3 project only");
static_assert(PY_MINOR_VERSION >= 6, "typed_python is a python3.6 project only");

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

    bool operator<(const PyObjectHolder& other) const {
        return m_ptr < other.m_ptr;
    }

    bool operator>(const PyObjectHolder& other) const {
        return m_ptr > other.m_ptr;
    }

    bool operator==(const PyObjectHolder& other) const {
        return m_ptr == other.m_ptr;
    }

    bool operator<(PyObject* other) const {
        return m_ptr < other;
    }

    bool operator>(PyObject* other) const {
        return m_ptr > other;
    }

    bool operator==(PyObject* other) const {
        return m_ptr == other;
    }

    PyObject* operator->() const {
        return m_ptr;
    }

    PyObject* get() const {
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

bool unpackTupleToMemberDefinition(PyObject* tuple, std::vector<MemberDefinition>& out);

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

//given a python richcompare flag, such as Py_LT,
// return a string naming the appropriate magic method implementing this comparison
inline const char* pyCompareFlagToMethod(int pyComparisonOp) {
    switch (pyComparisonOp) {
        case Py_EQ: return "__eq__";
        case Py_NE: return "__ne__";
        case Py_LT: return "__lt__";
        case Py_GT: return "__gt__";
        case Py_LE: return "__le__";
        case Py_GE: return "__ge__";
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
Call 'f', which must return int, in a block that guards against
exceptions returning nakedly to the python interpreter. This is meant
to guard the barrier between Python C callbacks and our internals which
may use exceptions. Returns -1 on failure, the function value on success.
******/
template<class func_type>
int translateExceptionToPyObjectReturningInt(func_type f) {
    try {
        return f();
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
    PyObjectStealer iterator(PyObject_GetIter(o));
    PyObjectHolder item;

    if (!iterator) {
        throw PythonExceptionSet();
    }

    while (item.steal(PyIter_Next(iterator))) {
        f((PyObject*)item);
    }

    if (PyErr_Occurred()) {
        throw PythonExceptionSet();
    }
}

/********
Iterate over 'o' calling 'f' with each PyObject encountered.

if 'f' return false, exit early.

May throw, so use in conjunction with 'translateExceptionToPyObject'
********/
template<class func_type>
void iterateWithEarlyExit(PyObject* o, func_type f) {
    PyObjectStealer iterator(PyObject_GetIter(o));
    PyObjectHolder item;

    if (!iterator) {
        throw PythonExceptionSet();
    }

    bool exitEarly = false;

    while ((item.steal(PyIter_Next(iterator))) && !exitEarly) {
        if (!f((PyObject*)item)) {
            exitEarly = true;
        }
    }

    if (PyErr_Occurred()) {
        throw PythonExceptionSet();
    }
}


// Removes trailing zeros from char* representation of a floating-point number
// in the style of the python str representation.
// Modifies buffer in place.
// 1.0 -> 1.0
// 1.00 -> 1.0
// 1.00000 -> 1.0
// 1.20 -> 1.2
// 1.200 -> 1.2
// 1.200001 -> 1.200001
// 1.500e7 -> 1.5e7
// 1.000e7 -> 1e7
inline void remove_trailing_zeros_pystyle(char *s) {
    char *cur = s;
    char *decimal = 0;
    char *firstzero = 0;
    while (*cur) {
        if (*cur == '.') {
            decimal = cur;
            firstzero = 0;
        }
        else if (*cur == '0' && decimal) {
            if (!firstzero)
                firstzero = cur;
        }
        else if (*cur == 'e' && firstzero)
            break;
        else
            firstzero = 0;
        cur++;
    }
    // *cur is 'e' or \x00
    if (firstzero) {
        if (!*cur) {
            if (firstzero - decimal == 1)
                firstzero++;
            if (firstzero < cur)
                *firstzero = 0;
        }
        else {
            if (firstzero - decimal == 1)
                firstzero--;
            while (*cur)
                *firstzero++ = *cur++;
            *firstzero = 0;
        }
    }
}

//if a name has a '.' in it, strip out everything but the last part
std::string qualname_to_name(std::string n) {
    size_t ix = n.rfind(".");
    if (ix == std::string::npos) {
        return n;
    }

    return n.substr(ix + 1);
}


inline PyObject* internalsModule() {
    static PyObject* module = PyImport_ImportModule("typed_python.internals");
    return module;
}


inline PyObject* runtimeModule() {
    static PyObject* module = PyImport_ImportModule("typed_python.compiler.runtime");
    return module;
}

inline PyObject* builtinsModule() {
    static PyObject* module = PyImport_ImportModule("builtins");
    return module;
}

inline PyObject* sysModule() {
    static PyObject* module = PyImport_ImportModule("sys");
    return module;
}

inline PyObject* weakrefModule() {
    static PyObject* module = PyImport_ImportModule("weakref");
    return module;
}

inline PyObject* osModule() {
    static PyObject* module = PyImport_ImportModule("os");
    return module;
}


inline bool startsWith(std::string name, std::string prefix) {
    if (name.size() < prefix.size()) {
        return false;
    }

    return name.substr(0, prefix.size()) == prefix;
}

