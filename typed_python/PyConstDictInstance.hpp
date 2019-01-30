#pragma once

#include "PyInstance.hpp"

class PyConstDictInstance : public PyInstance {
public:
    ConstDict* type();

    PyObject* sq_concat_concrete(PyObject* rhs);
};



