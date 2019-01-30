#pragma once

#include "PyInstance.hpp"

class PyConstDictInstance : public PyInstance {
public:
    ConstDict* type();

    PyObject* sq_concat_concrete(PyObject* rhs);

    PyObject* tp_iter_concrete();

    PyObject* tp_iternext_concrete();

    static PyObject* constDictItems(PyObject *o);

    static PyObject* constDictKeys(PyObject *o);

    static PyObject* constDictValues(PyObject *o);

    static PyObject* constDictGet(PyObject* o, PyObject* args);
};



