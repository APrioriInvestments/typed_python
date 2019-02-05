#pragma once

#include "PyInstance.hpp"

class PyPointerToInstance : public PyInstance {
public:
    typedef PointerTo modeled_type;

    PointerTo* type();

    static PyObject* pointerInitialize(PyObject* o, PyObject* args);

    static PyObject* pointerSet(PyObject* o, PyObject* args);

    static PyObject* pointerGet(PyObject* o, PyObject* args);

    static PyObject* pointerCast(PyObject* o, PyObject* args);

    PyObject* pyOperatorConcrete(PyObject* rhs, const char* op, const char* opErr);
};
