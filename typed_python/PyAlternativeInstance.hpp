#pragma once

#include "PyInstance.hpp"

class PyAlternativeInstance : public PyInstance {
public:
    typedef Alternative modeled_type;

    Alternative* type();

    PyObject* pyOperatorConcrete(PyObject* rhs, const char* op, const char* opErr);
};

class PyConcreteAlternativeInstance : public PyInstance {
public:
    typedef ConcreteAlternative modeled_type;

    ConcreteAlternative* type();

    PyObject* pyOperatorConcrete(PyObject* rhs, const char* op, const char* opErr);
};
