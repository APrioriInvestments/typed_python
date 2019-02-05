#pragma once

#include "PyInstance.hpp"

class PyNoneInstance : public PyInstance {
public:
    typedef None modeled_type;

    static void copyConstructFromPythonInstanceConcrete(None* oneOf, instance_ptr tgt, PyObject* pyRepresentation) {
        if (pyRepresentation == Py_None) {
            return;
        }
        throw std::logic_error("Can't initialize a None from an instance of " +
            std::string(pyRepresentation->ob_type->tp_name));
    }
};

