#pragma once

#include "PyInstance.hpp"

class PyPointerToInstance : public PyInstance {
public:
    PointerTo* type();

    static PyObject* pointerInitialize(PyObject* o, PyObject* args);

    static PyObject* pointerSet(PyObject* o, PyObject* args);

    static PyObject* pointerGet(PyObject* o, PyObject* args);

    static PyObject* pointerCast(PyObject* o, PyObject* args);
};
