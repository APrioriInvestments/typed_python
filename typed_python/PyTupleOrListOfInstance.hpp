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
};

class PyTupleOfInstance : public PyTupleOrListOfInstance {
public:
    TupleOf* type();
};
