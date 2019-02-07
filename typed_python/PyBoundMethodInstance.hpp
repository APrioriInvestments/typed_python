#pragma once

#include "PyInstance.hpp"

class PyBoundMethodInstance : public PyInstance {
public:
    typedef BoundMethod modeled_type;

    BoundMethod* type();

    PyObject* tp_call_concrete(PyObject* args, PyObject* kwargs);

    static bool pyValCouldBeOfTypeConcrete(modeled_type* type, PyObject* pyRepresentation) {
        return true;
    }
};
