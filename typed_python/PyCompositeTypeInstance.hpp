#pragma once

#include "PyInstance.hpp"

class PyCompositeTypeInstance : public PyInstance {
public:
    CompositeType* type();

    PyObject* sq_item_concrete(Py_ssize_t ix);

    Py_ssize_t mp_and_sq_length();
};

class PyTupleInstance : public PyCompositeTypeInstance {
public:
    Tuple* type();
};

class PyNamedTupleInstance : public PyCompositeTypeInstance {
public:
    NamedTuple* type();
};
