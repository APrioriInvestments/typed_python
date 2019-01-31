#pragma once

#include "PyInstance.hpp"

class PyConstDictInstance : public PyInstance {
public:
    ConstDict* type();

    PyObject* sq_concat_concrete(PyObject* rhs);

    PyObject* tp_iter_concrete();

    PyObject* tp_iternext_concrete();

    Py_ssize_t mp_and_sq_length_concrete();

    int sq_contains_concrete(PyObject* item);

    PyObject* pyOperatorConcrete(PyObject* rhs, const char* op, const char* opErr);

    static PyObject* constDictItems(PyObject *o);

    static PyObject* constDictKeys(PyObject *o);

    static PyObject* constDictValues(PyObject *o);

    static PyObject* constDictGet(PyObject* o, PyObject* args);
};



