#pragma once

#include "PyInstance.hpp"

class PyTupleOrListOfInstance : public PyInstance {
public:
    TupleOrListOf* type();

    PyObject* sq_concat_concrete(PyObject* rhs);
};

class PyListOfInstance : public PyTupleOrListOfInstance {
public:
    ListOf* type();

    static PyObject* listAppend(PyObject* o, PyObject* args);

    static PyObject* listResize(PyObject* o, PyObject* args);

    static PyObject* listReserve(PyObject* o, PyObject* args);

    static PyObject* listClear(PyObject* o, PyObject* args);

    static PyObject* listReserved(PyObject* o, PyObject* args);

    static PyObject* listPop(PyObject* o, PyObject* args);

    static PyObject* listSetSizeUnsafe(PyObject* o, PyObject* args);

    static PyObject* listPointerUnsafe(PyObject* o, PyObject* args);
};

class PyTupleOfInstance : public PyTupleOrListOfInstance {
public:
    TupleOf* type();
};
