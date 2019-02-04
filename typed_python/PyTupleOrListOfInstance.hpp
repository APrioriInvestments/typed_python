#pragma once

#include "PyInstance.hpp"

class PyTupleOrListOfInstance : public PyInstance {
public:
    TupleOrListOf* type();

    PyObject* sq_concat_concrete(PyObject* rhs);

    PyObject* sq_item_concrete(Py_ssize_t ix);

    Py_ssize_t mp_and_sq_length_concrete();

    PyObject* mp_subscript_concrete(PyObject* item);

    static void copyConstructFromPythonInstance(TupleOrListOf* tupT, instance_ptr tgt, PyObject* pyRepresentation);

    static PyObject* toArray(PyObject* o, PyObject* args);
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

    int mp_ass_subscript_concrete(PyObject* item, PyObject* value);
};

class PyTupleOfInstance : public PyTupleOrListOfInstance {
public:
    TupleOf* type();
};
