#pragma once

#include "PyInstance.hpp"

class PyAlternativeInstance : public PyInstance {
public:
    Alternative* type();

    PyObject* pyOperatorConcrete(PyObject* rhs, const char* op, const char* opErr);
};

class PyConcreteAlternativeInstance : public PyInstance {
public:
    ConcreteAlternative* type();

    PyObject* pyOperatorConcrete(PyObject* rhs, const char* op, const char* opErr);
};
